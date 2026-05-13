import cv2
import mediapipe as mp
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

WRIST        = 0
THUMB_TIP    = 4;  THUMB_IP  = 3;  THUMB_MCP = 2
INDEX_TIP    = 8;  INDEX_PIP = 6;  INDEX_MCP = 5
MIDDLE_TIP   = 12; MIDDLE_PIP = 10
RING_TIP     = 16; RING_PIP  = 14
PINKY_TIP    = 20; PINKY_PIP = 18

@dataclass
class HandData:
    landmarks_raw:    np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    landmarks_smooth: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    index_tip_norm:   tuple = (0.5, 0.5)
    index_tip_px:     tuple = (0, 0)
    detection_confidence: float = 0.0
    hand_detected:    bool = False
    # "Left" or "Right" — from MediaPipe's perspective (mirrored from user's)
    # After we flip the frame: MediaPipe "Right" = user's RIGHT hand
    handedness:       str = "Unknown"

@dataclass
class TwoHandData:
    # Container for both hands. right_hand controls cursor, left_hand controls clicks.
    right_hand: HandData = field(default_factory=HandData)
    left_hand:  HandData = field(default_factory=HandData)

class HandTracker:

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence:  float = 0.5,
        model_complexity: int = 1,
    ):
        self.max_num_hands            = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        self.model_complexity         = model_complexity
        self._hands       = None
        self._mp_hands    = mp.solutions.hands
        self._mp_drawing  = mp.solutions.drawing_utils

    def start(self):
        log.info("Initializing MediaPipe Hands (2-hand mode)...")
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        log.info("MediaPipe Hands initialized")

    def process(self, frame_bgr: np.ndarray) -> TwoHandData:
        """
        Process one frame. Returns TwoHandData with right_hand and left_hand.

        IMPORTANT — handedness after flip:
        We flip the frame horizontally before processing.
        MediaPipe labels hands from the camera's perspective.
        After flipping: MediaPipe "Right" = the user's actual right hand.
        """
        result = TwoHandData()
        if self._hands is None:
            return result

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        mp_result = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not mp_result.multi_hand_landmarks:
            return result

        for i, hand_landmarks in enumerate(mp_result.multi_hand_landmarks):
            if i >= 2:
                break

            # Get handedness label
            label = "Unknown"
            confidence = 0.0
            if mp_result.multi_handedness and i < len(mp_result.multi_handedness):
                cls        = mp_result.multi_handedness[i].classification[0]
                label      = cls.label       # "Left" or "Right"
                confidence = cls.score

            # Build landmark array
            raw = np.array([[lm.x, lm.y, lm.z]
                            for lm in hand_landmarks.landmark])

            hd = HandData(
                landmarks_raw         = raw,
                landmarks_smooth      = raw.copy(),
                index_tip_norm        = (raw[INDEX_TIP][0], raw[INDEX_TIP][1]),
                detection_confidence  = confidence,
                hand_detected         = True,
                handedness            = label,
            )

            # After flip: MediaPipe "Right" = user's right hand
            if label == "Right":
                result.right_hand = hd
            else:
                result.left_hand = hd

        return result

    def draw_landmarks(self, frame: np.ndarray, hand_data: HandData,
                       color_tip=(0, 255, 255)) -> np.ndarray:
        # Draw skeleton for one HandData onto frame.
        if not hand_data.hand_detected:
            return frame

        fh, fw = frame.shape[:2]
        pts = [(int(lm[0] * fw), int(lm[1] * fh))
               for lm in hand_data.landmarks_smooth]

        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        for s, e in CONNECTIONS:
            cv2.line(frame, pts[s], pts[e], (180,180,180), 2, cv2.LINE_AA)

        for i, (px, py) in enumerate(pts):
            r = 6 if i in (4,8,12,16,20) else 4
            cv2.circle(frame, (px,py), r, (255,255,255), -1)
            cv2.circle(frame, (px,py), r, (100,100,100), 1)

        # Index tip highlight
        ix, iy = pts[8]
        cv2.circle(frame, (ix,iy), 12, color_tip, -1)
        cv2.circle(frame, (ix,iy), 14, (0,180,180), 2)

        return frame

    def stop(self):
        if self._hands:
            self._hands.close()
        log.info("HandTracker stopped")