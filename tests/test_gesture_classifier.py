# tests/test_gesture_classifier.py
import numpy as np
import pytest
from src.vision.gesture_classifier import (
    FeatureExtractor, GestureDetector, GestureClassifier,
    Gesture, GestureState
)
from src.vision.hand_tracker import HandData


def _make_landmarks(
    index_up=True, middle_up=False, ring_up=False,
    pinky_up=False, thumb_out=True, pinch=False
) -> np.ndarray:
    """
    Build a synthetic landmark array for testing.
    
    We place landmarks at realistic positions so the feature extractor
    can compute meaningful values without a real camera.
    
    Coordinate system: (0,0) = top-left, (1,1) = bottom-right.
    Wrist is at the bottom, fingertips at the top.
    """
    lm = np.zeros((21, 3))
    
    # Wrist at bottom center
    lm[0] = [0.5, 0.85, 0.0]
    
    # Thumb — extend it sideways when thumb_out=True
    lm[1] = [0.55, 0.75, 0.0]  # CMC
    lm[2] = [0.60, 0.70, 0.0]  # MCP
    lm[3] = [0.65, 0.65, 0.0]  # IP
    lm[4] = [0.70, 0.60, 0.0] if thumb_out else [0.55, 0.68, 0.0]  # TIP
    
    # Index MCP (knuckle) — base reference
    lm[5] = [0.45, 0.65, 0.0]  # MCP
    lm[6] = [0.45, 0.55, 0.0]  # PIP
    lm[7] = [0.45, 0.45, 0.0]  # DIP
    lm[8] = [0.45, 0.35, 0.0] if index_up else [0.45, 0.58, 0.0]  # TIP
    
    # Middle finger
    lm[9]  = [0.50, 0.63, 0.0]  # MCP
    lm[10] = [0.50, 0.52, 0.0]  # PIP
    lm[11] = [0.50, 0.42, 0.0]  # DIP
    lm[12] = [0.50, 0.32, 0.0] if middle_up else [0.50, 0.55, 0.0]  # TIP
    
    # Ring finger
    lm[13] = [0.55, 0.63, 0.0]
    lm[14] = [0.55, 0.54, 0.0]
    lm[15] = [0.55, 0.45, 0.0]
    lm[16] = [0.55, 0.36, 0.0] if ring_up else [0.55, 0.57, 0.0]
    
    # Pinky finger
    lm[17] = [0.60, 0.65, 0.0]
    lm[18] = [0.60, 0.57, 0.0]
    lm[19] = [0.60, 0.50, 0.0]
    lm[20] = [0.60, 0.42, 0.0] if pinky_up else [0.60, 0.59, 0.0]
    
    # Pinch: move thumb tip close to index tip
    if pinch:
        lm[4] = lm[8].copy()  # Thumb tip overlaps index tip
        lm[4][0] += 0.02       # Tiny offset (not perfectly overlapping)
    
    return lm


class TestFeatureExtractor:
    
    def test_extended_index_detected(self):
        lm = _make_landmarks(index_up=True)
        f = FeatureExtractor().extract(lm)
        assert f['index_extended'] is True
    
    def test_folded_index_detected(self):
        lm = _make_landmarks(index_up=False)
        f = FeatureExtractor().extract(lm)
        assert f['index_extended'] is False
    
    def test_open_palm_all_fingers_up(self):
        lm = _make_landmarks(index_up=True, middle_up=True,
                              ring_up=True, pinky_up=True, thumb_out=True)
        f = FeatureExtractor().extract(lm)
        assert f['palm_open'] is True
        assert f['fingers_up'] == 4
    
    def test_fist_no_fingers_up(self):
        lm = _make_landmarks(index_up=False, middle_up=False,
                              ring_up=False, pinky_up=False, thumb_out=False)
        f = FeatureExtractor().extract(lm)
        assert f['fist'] is True
        assert f['fingers_up'] == 0
    
    def test_pinch_distance_small_when_pinching(self):
        lm = _make_landmarks(pinch=True)
        f = FeatureExtractor().extract(lm)
        assert f['pinch_thumb_index_norm'] < 0.2  # Should be very small


class TestGestureDetector:
    
    def test_pointing_scores_high_for_index_only(self):
        lm = _make_landmarks(index_up=True, middle_up=False)
        f  = FeatureExtractor().extract(lm)
        scores = GestureDetector().score_all(f)
        assert scores[Gesture.POINTING] > 0.6
    
    def test_freeze_scores_high_for_open_palm(self):
        lm = _make_landmarks(index_up=True, middle_up=True,
                              ring_up=True, pinky_up=True, thumb_out=True)
        f  = FeatureExtractor().extract(lm)
        scores = GestureDetector().score_all(f)
        assert scores[Gesture.FREEZE_CURSOR] > 0.8
    
    def test_click_scores_high_when_pinching(self):
        lm = _make_landmarks(pinch=True)
        f  = FeatureExtractor().extract(lm)
        scores = GestureDetector().score_all(f)
        assert scores[Gesture.LEFT_CLICK] > 0.7
    
    def test_pause_scores_high_for_fist(self):
        lm = _make_landmarks(index_up=False, middle_up=False,
                              ring_up=False, pinky_up=False, thumb_out=False)
        f  = FeatureExtractor().extract(lm)
        scores = GestureDetector().score_all(f)
        assert scores[Gesture.PAUSE_TRACKING] > 0.8


class TestGestureClassifier:
    
    def _make_hand_data(self, **kwargs):
        hd = HandData()
        hd.hand_detected = True
        hd.landmarks_raw    = _make_landmarks(**kwargs)
        hd.landmarks_smooth = hd.landmarks_raw.copy()
        return hd
    
    def test_no_gesture_when_hand_absent(self):
        clf = GestureClassifier()
        clf.start()
        hd = HandData()  # hand_detected=False by default
        result = clf.process(hd)
        assert result.gesture == Gesture.NONE
    
    def test_pointing_gesture_eventually_fires(self):
        """Feed several frames of pointing — should become ACTIVE."""
        clf = GestureClassifier({'gestures': {'stability_frames': 2, 'confidence_threshold': 0.5}})
        clf.start()
        hd = self._make_hand_data(index_up=True, middle_up=False)
        result = None
        for _ in range(5):  # Feed 5 frames
            result = clf.process(hd)
        assert result.gesture == Gesture.POINTING
        assert result.state == GestureState.ACTIVE
    
    def test_freeze_gesture_fires_for_open_palm(self):
        clf = GestureClassifier({'gestures': {'stability_frames': 2, 'confidence_threshold': 0.5}})
        clf.start()
        hd = self._make_hand_data(index_up=True, middle_up=True,
                                   ring_up=True, pinky_up=True, thumb_out=True)
        result = None
        for _ in range(5):
            result = clf.process(hd)
        assert result.gesture == Gesture.FREEZE_CURSOR