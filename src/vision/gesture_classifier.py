# src/vision/gesture_classifier.py
# Redesigned for real-world usability.
# Key changes:
#   - Right click: index + middle BOTH touch thumb (three-finger pinch)
#   - Scroll: index + middle extended AND spread apart (width > threshold)
#   - Freeze and Pause are TOGGLES, not one-direction states
#   - Better confidence scoring with real-hand-tested thresholds
#   - Clearer gesture separation so gestures don't bleed into each other

import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from enum import Enum, auto

log = logging.getLogger(__name__)


class Gesture(Enum):
    NONE           = auto()
    POINTING       = auto()   # Index only — default cursor mode
    LEFT_CLICK     = auto()   # Thumb + index pinch tap
    RIGHT_CLICK    = auto()   # Thumb + index + middle three-finger pinch
    DRAG_START     = auto()   # Thumb + index pinch HOLD (0.5s)
    SCROLL         = auto()   # Index + middle spread, move hand
    PRECISION_MODE = auto()   # Peace sign — slow precise cursor
    FREEZE_CURSOR  = auto()   # Open palm TOGGLE
    PAUSE_TRACKING = auto()   # Fist hold TOGGLE


class GestureState(Enum):
    IDLE     = auto()
    HOLDING  = auto()
    ACTIVE   = auto()
    RELEASED = auto()


@dataclass
class GestureResult:
    gesture: Gesture = Gesture.NONE
    confidence: float = 0.0
    state: GestureState = GestureState.IDLE
    scroll_dx: float = 0.0
    scroll_dy: float = 0.0
    hold_duration: float = 0.0
    just_triggered: bool = False
    # System-level toggle states (persist across frames)
    cursor_frozen: bool = False
    tracking_paused: bool = False
    finger_states: dict = field(default_factory=dict)
    pinch_distance: float = 1.0
    debug_scores: dict = field(default_factory=dict)


# ── Landmark indices ──────────────────────────────────────────────────────────
_WRIST      = 0
_THUMB_TIP  = 4;  _THUMB_IP   = 3;  _THUMB_MCP  = 2
_INDEX_TIP  = 8;  _INDEX_PIP  = 6;  _INDEX_MCP  = 5
_MIDDLE_TIP = 12; _MIDDLE_PIP = 10; _MIDDLE_MCP = 9
_RING_TIP   = 16; _RING_PIP   = 14
_PINKY_TIP  = 20; _PINKY_PIP  = 18


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:

    EXTEND_THRESH = 0.035

    def extract(self, lm: np.ndarray) -> dict:
        f = {}

        # Hand size reference (wrist → middle MCP)
        hand_size = max(self._dist2d(lm[_WRIST], lm[_MIDDLE_MCP]), 0.01)
        f['hand_size'] = hand_size

        # Finger extended states
        f['index_extended']  = lm[_INDEX_TIP][1]  < lm[_INDEX_PIP][1]  - self.EXTEND_THRESH
        f['middle_extended'] = lm[_MIDDLE_TIP][1] < lm[_MIDDLE_PIP][1] - self.EXTEND_THRESH
        f['ring_extended']   = lm[_RING_TIP][1]   < lm[_RING_PIP][1]   - self.EXTEND_THRESH
        f['pinky_extended']  = lm[_PINKY_TIP][1]  < lm[_PINKY_PIP][1]  - self.EXTEND_THRESH
        f['thumb_extended']  = self._thumb_extended(lm, hand_size)

        f['fingers_up'] = sum([
            f['index_extended'], f['middle_extended'],
            f['ring_extended'],  f['pinky_extended']
        ])

        # Pinch distances (normalized by hand size)
        f['pinch_ti_norm'] = self._dist2d(lm[_THUMB_TIP], lm[_INDEX_TIP])  / hand_size
        f['pinch_tm_norm'] = self._dist2d(lm[_THUMB_TIP], lm[_MIDDLE_TIP]) / hand_size
        f['pinch_im_norm'] = self._dist2d(lm[_INDEX_TIP], lm[_MIDDLE_TIP]) / hand_size

        # Three-finger pinch: average of thumb-index AND thumb-middle distances
        f['pinch_3finger'] = (f['pinch_ti_norm'] + f['pinch_tm_norm']) / 2.0

        # Spread between index and middle tips (for scroll detection)
        # Large spread = peace/scroll gesture. Small spread = scroll ready.
        f['im_spread'] = self._dist2d(lm[_INDEX_TIP], lm[_MIDDLE_TIP]) / hand_size

        # Palm openness
        f['palm_open'] = f['fingers_up'] >= 4 and f['thumb_extended']

        # Fist
        f['fist'] = f['fingers_up'] == 0 and not f['thumb_extended']

        return f

    def _thumb_extended(self, lm, hand_size) -> bool:
        dist = self._dist2d(lm[_THUMB_TIP], lm[_INDEX_MCP])
        return (dist / hand_size) > 0.45

    @staticmethod
    def _dist2d(a, b) -> float:
        dx = a[0] - b[0]; dy = a[1] - b[1]
        return (dx*dx + dy*dy) ** 0.5


# ── Gesture Detector ──────────────────────────────────────────────────────────

class GestureDetector:
    """
    Tuned thresholds based on real-hand testing.
    All pinch distances are normalized by hand size so they work
    regardless of how far the hand is from the camera.
    """

    # How close fingers must be to count as "pinching"
    # 0.25 = ~25% of the wrist-to-knuckle distance. Comfortable pinch.
    PINCH_2_THRESH  = 0.28   # Thumb + index (left click)
    PINCH_3_THRESH  = 0.32   # Three-finger avg (right click) — slightly easier
    SCROLL_SPREAD   = 0.30   # Index+middle must be at least this far apart to scroll
    SCROLL_TOGETHER = 0.22   # Below this = too close, probably not scroll intent

    def score_all(self, f: dict) -> dict:
        return {
            Gesture.POINTING:       self._score_pointing(f),
            Gesture.LEFT_CLICK:     self._score_left_click(f),
            Gesture.RIGHT_CLICK:    self._score_right_click(f),
            Gesture.DRAG_START:     self._score_left_click(f),
            Gesture.SCROLL:         self._score_scroll(f),
            Gesture.PRECISION_MODE: self._score_precision(f),
            Gesture.FREEZE_CURSOR:  self._score_freeze(f),
            Gesture.PAUSE_TRACKING: self._score_pause(f),
        }

    def _score_pointing(self, f) -> float:
        """Index only up, others down, no pinch."""
        s = 0.0
        if f['index_extended']:       s += 0.45
        if not f['middle_extended']:  s += 0.20
        if not f['ring_extended']:    s += 0.15
        if not f['pinky_extended']:   s += 0.20
        # Penalize if it's a pinch (don't confuse with click)
        if f['pinch_ti_norm'] < self.PINCH_2_THRESH:
            s *= 0.3
        return max(0.0, min(1.0, s))

    def _score_left_click(self, f) -> float:
        """Thumb + index pinch. Middle finger should be up or neutral."""
        pinch = f['pinch_ti_norm']
        if pinch > self.PINCH_2_THRESH * 1.4:
            return 0.0
        # Ramp: at threshold = 0.5, at half threshold = 1.0
        s = 1.0 - (pinch / self.PINCH_2_THRESH)
        s = max(0.0, min(1.0, s))
        # Penalize if middle is ALSO pinched (that's a right click)
        if f['pinch_tm_norm'] < self.PINCH_3_THRESH:
            s *= 0.4
        return s

    def _score_right_click(self, f) -> float:
        """
        Three-finger pinch: thumb touches BOTH index AND middle simultaneously.
        Much easier to do reliably than thumb-middle only.
        Bring all three fingertips together like holding a small grape.
        """
        avg = f['pinch_3finger']
        if avg > self.PINCH_3_THRESH * 1.4:
            return 0.0
        s = 1.0 - (avg / self.PINCH_3_THRESH)
        s = max(0.0, min(1.0, s))
        # Both index AND middle must be close to thumb
        if f['pinch_ti_norm'] > self.PINCH_3_THRESH * 1.1:
            s *= 0.5
        if f['pinch_tm_norm'] > self.PINCH_3_THRESH * 1.1:
            s *= 0.5
        return s

    def _score_scroll(self, f) -> float:
        """
        Index + middle both extended AND spread apart (like a peace sign
        but fingers spread wide). Move the whole hand to scroll.
        The spread requirement separates this from precision mode.
        """
        s = 0.0
        if f['index_extended']:      s += 0.30
        if f['middle_extended']:     s += 0.30
        if not f['ring_extended']:   s += 0.15
        if not f['pinky_extended']:  s += 0.15
        # Spread check: fingers should be apart
        spread = f['im_spread']
        if spread >= self.SCROLL_SPREAD:
            s += 0.10
        elif spread < self.SCROLL_TOGETHER:
            s *= 0.4   # Too close together — not a scroll
        # Not a pinch
        if f['pinch_ti_norm'] < self.PINCH_2_THRESH:
            s *= 0.2
        return max(0.0, min(1.0, s))

    def _score_precision(self, f) -> float:
        """
        Peace sign with fingers close together (not spread).
        Index + middle up, ring + pinky down, no pinch.
        """
        s = 0.0
        if f['index_extended']:      s += 0.30
        if f['middle_extended']:     s += 0.30
        if not f['ring_extended']:   s += 0.20
        if not f['pinky_extended']:  s += 0.20
        # Penalize if spread wide (that's scroll)
        if f['im_spread'] >= self.SCROLL_SPREAD:
            s *= 0.5
        return max(0.0, min(1.0, s))

    def _score_freeze(self, f) -> float:
        """Open palm — all 5 extended. Penalize if pinching."""
        s = 0.0
        if f['index_extended']:  s += 0.20
        if f['middle_extended']: s += 0.20
        if f['ring_extended']:   s += 0.20
        if f['pinky_extended']:  s += 0.20
        if f['thumb_extended']:  s += 0.20

        # Can't be an open palm if thumb is pinching index
        if f['pinch_ti_norm'] < 0.45:
            s *= 0.2

        return max(0.0, min(1.0, s))

    def _score_pause(self, f) -> float:
        """Closed fist — ALL fingers folded AND no pinch happening."""
        s = 0.0
        if not f['index_extended']:  s += 0.25
        if not f['middle_extended']: s += 0.25
        if not f['ring_extended']:   s += 0.25
        if not f['pinky_extended']:  s += 0.25

        # Critical: if any pinch is happening, this is NOT a fist.
        # When someone pinches, other fingers curl — that looks like a fist.
        # We break the tie by requiring the hand to be genuinely closed.
        if f['pinch_ti_norm'] < 0.45:   # Thumb near index = pinch, not fist
            s *= 0.1
        if f['pinch_tm_norm'] < 0.45:   # Thumb near middle = pinch, not fist
            s *= 0.1
        if f['pinch_3finger'] < 0.40:   # Three-finger pinch = definitely not fist
            s *= 0.1

        return max(0.0, min(1.0, s))


# ── Confidence Guard ──────────────────────────────────────────────────────────

class ConfidenceGuard:

    def __init__(
        self,
        min_confidence: float = 0.72,
        stability_frames: int = 4,
        click_hold_seconds: float = 0.5,
        pause_hold_seconds: float = 1.5,
    ):
        self.min_confidence    = min_confidence
        self.stability_frames  = stability_frames
        self.click_hold_secs   = click_hold_seconds
        self.pause_hold_secs   = pause_hold_seconds
        self._history: deque   = deque(maxlen=stability_frames)
        self._hold_start: Optional[float] = None
        self._hold_gesture: Gesture = Gesture.NONE
        self._last_triggered: float = 0.0

    def evaluate(self, scores: dict, now: float) -> GestureResult:
        result = GestureResult()

        # Layer 1 — threshold (higher bar for destructive/disruptive gestures)
        GESTURE_THRESHOLDS = {
            Gesture.PAUSE_TRACKING: 0.92,   # Must be very deliberate
            Gesture.FREEZE_CURSOR:  0.88,   # Must be clearly open palm
            Gesture.DRAG_START:     0.78,
        }

        best, best_score = Gesture.NONE, 0.0
        for g, s in scores.items():
            required = GESTURE_THRESHOLDS.get(g, self.min_confidence)
            if s >= required and s > best_score:
                best, best_score = g, s

        # Layer 2 — stability
        self._history.append(best)
        if len(self._history) < self.stability_frames or \
                not all(g == best for g in self._history):
            result.gesture    = Gesture.NONE
            result.confidence = best_score * 0.5
            return result

        # Layer 3 — timing
        if self._hold_gesture != best:
            self._hold_gesture = best
            self._hold_start   = now

        hold = now - (self._hold_start or now)
        result.hold_duration = hold

        required = 0.0
        if best == Gesture.DRAG_START:   required = self.click_hold_secs
        elif best == Gesture.PAUSE_TRACKING: required = self.pause_hold_secs

        if hold < required:
            result.gesture    = best
            result.confidence = best_score
            result.state      = GestureState.HOLDING
            return result

        result.gesture    = best
        result.confidence = best_score
        result.state      = GestureState.ACTIVE

        cooldown = 0.4
        if now - self._last_triggered > cooldown:
            result.just_triggered  = True
            self._last_triggered   = now

        return result

    def reset(self):
        self._history.clear()
        self._hold_start   = None
        self._hold_gesture = Gesture.NONE


# ── Scroll Tracker ────────────────────────────────────────────────────────────

class ScrollTracker:

    def __init__(self, sensitivity: float = 3.0):
        self.sensitivity = sensitivity
        self._prev: Optional[tuple] = None

    def update(self, lm: np.ndarray) -> tuple:
        ix, iy = lm[_INDEX_TIP][0],  lm[_INDEX_TIP][1]
        mx, my = lm[_MIDDLE_TIP][0], lm[_MIDDLE_TIP][1]
        mid_x  = (ix + mx) / 2.0
        mid_y  = (iy + my) / 2.0

        if self._prev is None:
            self._prev = (mid_x, mid_y)
            return (0.0, 0.0)

        dx = (mid_x - self._prev[0]) * self.sensitivity
        dy = (mid_y - self._prev[1]) * self.sensitivity
        self._prev = (mid_x, mid_y)
        return (dx, dy)

    def reset(self):
        self._prev = None


# ── Main Classifier ───────────────────────────────────────────────────────────

class GestureClassifier:

    def __init__(self, config: dict = None):
        cfg = config or {}
        g   = cfg.get('gestures', {})

        self._extractor = FeatureExtractor()
        self._detector  = GestureDetector()
        self._guard     = ConfidenceGuard(
            min_confidence     = g.get('confidence_threshold', 0.72),
            stability_frames   = g.get('stability_frames', 4),
            click_hold_seconds = g.get('hold_duration', 0.5),
            pause_hold_seconds = g.get('fist_pause_duration', 1.5),
        )
        self._scroll = ScrollTracker(sensitivity=g.get('scroll_sensitivity', 3.0))

        # Toggle states — persist between frames
        self._cursor_frozen    = False
        self._tracking_paused  = False

        self._prev_result = GestureResult()

    def start(self):
        log.info("GestureClassifier ready — redesigned gesture set v2")

    def process(self, hand_data) -> GestureResult:

        if self._tracking_paused:
            # Still watch for fist-hold to RESUME
            if not hand_data.hand_detected:
                return GestureResult(tracking_paused=True)

            lm       = hand_data.landmarks_smooth
            features = self._extractor.extract(lm)
            scores   = self._detector.score_all(features)
            raw      = self._guard.evaluate(scores, time.perf_counter())

            if raw.gesture == Gesture.PAUSE_TRACKING and raw.just_triggered:
                self._tracking_paused = False
                self._guard.reset()
                log.info("Tracking RESUMED via fist toggle")

            result = GestureResult(tracking_paused=True, cursor_frozen=self._cursor_frozen)
            result.gesture = raw.gesture
            return result

        if not hand_data.hand_detected:
            self._guard.reset()
            self._scroll.reset()
            return GestureResult(
                cursor_frozen=self._cursor_frozen,
                tracking_paused=self._tracking_paused
            )

        lm       = hand_data.landmarks_smooth
        features = self._extractor.extract(lm)
        scores   = self._detector.score_all(features)
        result   = self._guard.evaluate(scores, time.perf_counter())

        # ── Handle toggles ────────────────────────────────────────────────
        if result.gesture == Gesture.FREEZE_CURSOR and result.just_triggered:
            self._cursor_frozen = not self._cursor_frozen
            log.info(f"Cursor freeze toggled → {self._cursor_frozen}")

        if result.gesture == Gesture.PAUSE_TRACKING and result.just_triggered:
            self._tracking_paused = True
            self._guard.reset()
            log.info("Tracking PAUSED via fist toggle")

        # ── Scroll direction ──────────────────────────────────────────────
        if result.gesture == Gesture.SCROLL and result.state == GestureState.ACTIVE:
            result.scroll_dx, result.scroll_dy = self._scroll.update(lm)
        else:
            self._scroll.reset()

        # ── Attach persistent states ──────────────────────────────────────
        result.cursor_frozen   = self._cursor_frozen
        result.tracking_paused = self._tracking_paused

        # ── Attach debug info ─────────────────────────────────────────────
        result.finger_states = {
            'index':  features.get('index_extended',  False),
            'middle': features.get('middle_extended', False),
            'ring':   features.get('ring_extended',   False),
            'pinky':  features.get('pinky_extended',  False),
            'thumb':  features.get('thumb_extended',  False),
        }
        result.pinch_distance = features.get('pinch_ti_norm', 1.0)

        self._prev_result = result
        return result

    def stop(self):
        log.info("GestureClassifier stopped")