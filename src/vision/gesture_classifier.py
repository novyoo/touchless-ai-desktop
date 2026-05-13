# src/vision/gesture_classifier.py

import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from enum import Enum, auto

log = logging.getLogger(__name__)

class Gesture(Enum):
    NONE         = auto()
    POINTING     = auto()   # Right hand: index only
    PALM         = auto()   # Right hand: all 5 fingers open — pause toggle
    LEFT_CLICK   = auto()   # Left hand: thumb+index tap
    RIGHT_CLICK  = auto()   # Left hand: thumb+middle tap
    SCROLL_UP    = auto()   # Both hands: index only pointing simultaneously
    SCROLL_DOWN  = auto()   # Both hands: peace sign (index+middle) simultaneously
    DRAG         = auto()   # Right pointing + left pinch hold

class GestureState(Enum):
    IDLE    = auto()
    HOLDING = auto()
    ACTIVE  = auto()

@dataclass
class HandGestureResult:
    gesture:        Gesture      = Gesture.NONE
    confidence:     float        = 0.0
    state:          GestureState = GestureState.IDLE
    hold_duration:  float        = 0.0
    just_triggered: bool         = False
    finger_states:  dict         = field(default_factory=dict)
    pinch_ti_dist:  float        = 1.0
    pinch_tm_dist:  float        = 1.0

@dataclass
class GestureResult:
    right:           HandGestureResult = field(default_factory=HandGestureResult)
    left:            HandGestureResult = field(default_factory=HandGestureResult)
    scroll_dy:       float = 0.0
    is_dragging:     bool  = False
    tracking_paused: bool  = False
    # Hold progress for two-hand scroll gestures (0.0 → 1.0)
    scroll_up_progress:   float = 0.0   # fills while both hands point
    scroll_down_progress: float = 0.0   # fills while both hands peace

# Landmark indices 
_WRIST      = 0
_THUMB_TIP  = 4;  _THUMB_MCP  = 2
_INDEX_TIP  = 8;  _INDEX_PIP  = 6;  _INDEX_MCP  = 5
_MIDDLE_TIP = 12; _MIDDLE_PIP = 10; _MIDDLE_MCP = 9
_RING_TIP   = 16; _RING_PIP   = 14
_PINKY_TIP  = 20; _PINKY_PIP  = 18

class FeatureExtractor:

    EXTEND_THRESH = 0.035

    def extract(self, lm: np.ndarray) -> dict:
        f = {}
        hand_size = max(self._dist2d(lm[_WRIST], lm[_MIDDLE_MCP]), 0.01)
        f['hand_size']       = hand_size
        f['index_extended']  = lm[_INDEX_TIP][1]  < lm[_INDEX_PIP][1]  - self.EXTEND_THRESH
        f['middle_extended'] = lm[_MIDDLE_TIP][1] < lm[_MIDDLE_PIP][1] - self.EXTEND_THRESH
        f['ring_extended']   = lm[_RING_TIP][1]   < lm[_RING_PIP][1]   - self.EXTEND_THRESH
        f['pinky_extended']  = lm[_PINKY_TIP][1]  < lm[_PINKY_PIP][1]  - self.EXTEND_THRESH
        f['thumb_extended']  = (
            self._dist2d(lm[_THUMB_TIP], lm[_INDEX_MCP]) / hand_size
        ) > 0.45
        f['fingers_up']    = sum([
            f['index_extended'], f['middle_extended'],
            f['ring_extended'],  f['pinky_extended']
        ])
        f['pinch_ti_norm'] = self._dist2d(lm[_THUMB_TIP], lm[_INDEX_TIP])  / hand_size
        f['pinch_tm_norm'] = self._dist2d(lm[_THUMB_TIP], lm[_MIDDLE_TIP]) / hand_size

        # Pure pointing: ONLY index up, all others down, no pinch
        f['is_pointing'] = (
            f['index_extended'] and
            not f['middle_extended'] and
            not f['ring_extended'] and
            not f['pinky_extended'] and
            f['pinch_ti_norm'] > 0.30   # Not pinching
        )

        # Peace sign: index + middle up, ring + pinky down
        f['is_peace'] = (
            f['index_extended'] and
            f['middle_extended'] and
            not f['ring_extended'] and
            not f['pinky_extended']
        )

        # Open palm: all 4 fingers + thumb extended
        f['is_palm'] = (
            f['fingers_up'] >= 4 and
            f['thumb_extended']
        )

        return f

    @staticmethod
    def _dist2d(a, b) -> float:
        dx = a[0] - b[0]; dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

class RightHandClassifier:
    """
    Right hand only.
    Detects: POINTING, PALM (pause toggle).
    Cursor moves ONLY when POINTING is active.
    """

    def __init__(self, config: dict = None):
        cfg              = (config or {}).get('gestures', {})
        self.pause_secs  = cfg.get('pause_hold_duration', 1.0)
        self.stab_frames = cfg.get('stability_frames', 4)
        self._extractor  = FeatureExtractor()

        # Stability buffers
        self._point_buf: deque = deque(maxlen=self.stab_frames)
        self._peace_buf: deque = deque(maxlen=self.stab_frames)

        # Palm hold state
        self._palm_since:  Optional[float] = None
        self._palm_fired:  bool = False

    def process(self, lm: np.ndarray) -> HandGestureResult:
        result = HandGestureResult()
        f      = self._extractor.extract(lm)

        result.finger_states = {
            'index':  f['index_extended'],
            'middle': f['middle_extended'],
            'ring':   f['ring_extended'],
            'pinky':  f['pinky_extended'],
            'thumb':  f['thumb_extended'],
        }
        result.pinch_ti_dist = f['pinch_ti_norm']

        now = time.perf_counter()

        # Palm detection (pause toggle) 
        # Palm detection (pause toggle)
        if f['is_palm']:
            if self._palm_since is None:
                self._palm_since = now
                self._palm_fired = False

            held = now - self._palm_since
            result.hold_duration = held

            if held >= self.pause_secs and not self._palm_fired:
                result.gesture        = Gesture.PALM
                result.state          = GestureState.ACTIVE
                result.just_triggered = True
                result.hold_duration  = held
                self._palm_fired      = True
                self._point_buf.clear()
                return result
            else:
                result.gesture    = Gesture.PALM
                result.state      = GestureState.HOLDING
                result.hold_duration = held
                return result
        else:
            # KEY FIX: reset BOTH _palm_since AND _palm_fired when hand opens
            # Previously _palm_fired stayed True — blocked second palm trigger
            self._palm_since = None
            self._palm_fired = False   # ← this line was missing

        # Pointing detection 
        # Strict: index ONLY. Any other finger up = not pointing.
        self._point_buf.append(f['is_pointing'])
        if (len(self._point_buf) >= self.stab_frames and
                all(self._point_buf)):
            result.gesture    = Gesture.POINTING
            result.confidence = 0.95
            result.state      = GestureState.ACTIVE

        # Peace sign detection (for scroll down) 
        self._peace_buf.append(f['is_peace'])

        return result

    def is_peace_stable(self) -> bool:
        """Returns True if right hand has been showing peace sign for stab_frames."""
        return (len(self._peace_buf) >= self.stab_frames and
                all(self._peace_buf))

    def reset(self):
        self._point_buf.clear()
        self._peace_buf.clear()
        self._palm_since = None
        self._palm_fired = False

class LeftHandClassifier:
    """
    Left hand only.
    Detects: LEFT_CLICK, RIGHT_CLICK (tap = quick pinch release).
    Scroll is handled by GestureClassifier combining both hands.
    """
    PINCH_DOWN = 0.28
    PINCH_UP   = 0.38   # Hysteresis — must open wider than pinch threshold to release

    def __init__(self, config: dict = None):
        cfg              = (config or {}).get('gestures', {})
        self.scroll_hold = cfg.get('scroll_hold_duration', 0.3)
        self._extractor  = FeatureExtractor()
        self.stab_frames = cfg.get('stability_frames', 4)

        # Thumb-index state
        self._ti_down:        bool  = False
        self._ti_since:       float = 0.0
        self._ti_scroll_mode: bool  = False  # True once hold exceeds scroll_hold

        # Thumb-middle state
        self._tm_down:        bool  = False
        self._tm_since:       float = 0.0
        self._tm_scroll_mode: bool  = False

        # Peace sign buffer (for scroll down detection)
        self._peace_buf: deque = deque(maxlen=self.stab_frames)

    def process(self, lm: np.ndarray) -> HandGestureResult:
        result = HandGestureResult()
        f      = self._extractor.extract(lm)

        result.finger_states = {
            'index':  f['index_extended'],
            'middle': f['middle_extended'],
            'ring':   f['ring_extended'],
            'pinky':  f['pinky_extended'],
            'thumb':  f['thumb_extended'],
        }
        result.pinch_ti_dist = f['pinch_ti_norm']
        result.pinch_tm_dist = f['pinch_tm_norm']

        # Track peace sign for scroll down
        self._peace_buf.append(f['is_peace'])

        now = time.perf_counter()
        ti  = f['pinch_ti_norm']
        tm  = f['pinch_tm_norm']

        # Thumb-index state machine 
        if not self._ti_down and ti < self.PINCH_DOWN:
            self._ti_down        = True
            self._ti_since       = now
            self._ti_scroll_mode = False

        elif self._ti_down and ti > self.PINCH_UP:
            held = now - self._ti_since
            self._ti_down = False
            if not self._ti_scroll_mode:
                # Quick release = click
                result.gesture        = Gesture.LEFT_CLICK
                result.state          = GestureState.ACTIVE
                result.just_triggered = True
                result.confidence     = 0.95
                result.hold_duration  = held
                return result
            self._ti_scroll_mode = False

        if self._ti_down:
            held = now - self._ti_since
            result.hold_duration = held
            if held >= self.scroll_hold:
                self._ti_scroll_mode = True

        # Thumb-middle state machine 
        if not self._tm_down and tm < self.PINCH_DOWN:
            # Only enter if NOT already in ti pinch (avoid ambiguity)
            if not self._ti_down:
                self._tm_down        = True
                self._tm_since       = now
                self._tm_scroll_mode = False

        elif self._tm_down and tm > self.PINCH_UP:
            held = now - self._tm_since
            self._tm_down = False
            if not self._tm_scroll_mode:
                result.gesture        = Gesture.RIGHT_CLICK
                result.state          = GestureState.ACTIVE
                result.just_triggered = True
                result.confidence     = 0.95
                result.hold_duration  = held
                return result
            self._tm_scroll_mode = False

        if self._tm_down:
            held = now - self._tm_since
            if held >= self.scroll_hold:
                self._tm_scroll_mode = True

        return result

    def is_peace_stable(self) -> bool:
        return (len(self._peace_buf) >= self.stab_frames and
                all(self._peace_buf))

    def is_pointing_stable(self, lm: np.ndarray) -> bool:
        f = self._extractor.extract(lm)
        return f['is_pointing']

    @property
    def ti_in_scroll_mode(self) -> bool:
        return self._ti_down and self._ti_scroll_mode

    @property
    def tm_in_scroll_mode(self) -> bool:
        return self._tm_down and self._tm_scroll_mode

    def reset(self):
        self._ti_down        = False
        self._tm_down        = False
        self._ti_scroll_mode = False
        self._tm_scroll_mode = False
        self._peace_buf.clear()

class GestureClassifier:
    """
    Two-hand classifier.

    RIGHT hand  → cursor (POINTING only) + pause (PALM)
    LEFT hand   → clicks (pinch tap) 
    BOTH hands  → scroll up (both pointing) / scroll down (both peace)
    """

    def __init__(self, config: dict = None):
        self._config  = config or {}
        self._right   = RightHandClassifier(config)
        self._left    = LeftHandClassifier(config)
        self._tracking_paused = False

        cfg = self._config.get('gestures', {})
        # Scroll speed: units per frame at 30fps
        # This fires every frame while gesture is held → smooth continuous scroll
        self._scroll_speed = cfg.get('scroll_sensitivity', 5.0)
        self._scroll_hold_duration = cfg.get('scroll_hold_duration', 2.0)

        self._scroll_up_since = None
        self._scroll_down_since = None

    def start(self):
        log.info("GestureClassifier ready — two-hand v3")

    def process(self, two_hand_data) -> GestureResult:
        result = GestureResult(tracking_paused=self._tracking_paused)

        rh = two_hand_data.right_hand
        lh = two_hand_data.left_hand

        # Right hand 
        if rh.hand_detected:
            result.right = self._right.process(rh.landmarks_smooth)
        else:
            self._right.reset()

        # Palm toggle for pause/resume

        if result.right.gesture == Gesture.PALM and result.right.just_triggered:

            self._tracking_paused = not self._tracking_paused

            # Reset all temporary gesture states
            self._right.reset()
            self._left.reset()

            # Reset scroll timers too
            self._scroll_up_since = None
            self._scroll_down_since = None

            log.info(
                f"Tracking {'PAUSED' if self._tracking_paused else 'RESUMED'}"
            )

            result.tracking_paused = self._tracking_paused

            # IMPORTANT:
            # Stop processing immediately after pause toggle
            return result

        result.tracking_paused = self._tracking_paused
        if self._tracking_paused:
            return result

        # Left hand 
        if lh.hand_detected:
            result.left = self._left.process(lh.landmarks_smooth)
        else:
            self._left.reset()

        # Two-hand scroll detection 
        # Scroll UP:   both hands index-only pointing simultaneously
        # Scroll DOWN: both hands peace sign simultaneously
        # These override individual hand gestures when both hands are present.

        both_present = rh.hand_detected and lh.hand_detected

        if both_present:
            right_pointing = result.right.gesture == Gesture.POINTING
            right_peace    = self._right.is_peace_stable()
            left_pointing  = lh.hand_detected and self._left.is_pointing_stable(lh.landmarks_smooth)
            left_peace     = self._left.is_peace_stable()

            now = time.perf_counter()

            # SCROLL UP (both pointing)
            if right_pointing and left_pointing:
                if self._scroll_up_since is None:
                    self._scroll_up_since = now
                held = now - self._scroll_up_since
                result.scroll_up_progress = min(1.0, held / self._scroll_hold_duration)
                if held >= self._scroll_hold_duration:
                    result.scroll_dy     = -self._scroll_speed
                    result.right.gesture = Gesture.SCROLL_UP
                    result.left.gesture  = Gesture.SCROLL_UP
            else:
                self._scroll_up_since    = None
                result.scroll_up_progress = 0.0

            # SCROLL DOWN (both peace)
            if right_peace and left_peace:
                if self._scroll_down_since is None:
                    self._scroll_down_since = now
                held = now - self._scroll_down_since
                result.scroll_down_progress = min(1.0, held / self._scroll_hold_duration)
                if held >= self._scroll_hold_duration:
                    result.scroll_dy     = self._scroll_speed
                    result.right.gesture = Gesture.SCROLL_DOWN
                    result.left.gesture  = Gesture.SCROLL_DOWN
            else:
                self._scroll_down_since     = None
                result.scroll_down_progress = 0.0

        # Drag: right pointing + left pinch held 
        right_pointing_active = (result.right.gesture == Gesture.POINTING and
                                 result.right.state    == GestureState.ACTIVE)
        left_drag_held = (lh.hand_detected and
                          self._left._ti_down and
                          self._left._ti_scroll_mode)

        result.is_dragging = right_pointing_active and left_drag_held

        return result

    def stop(self):
        log.info("GestureClassifier stopped")