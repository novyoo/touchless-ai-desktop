# HandTracker — wraps MediaPipe Hands and provides clean landmark data.
# MediaPipe Hands is a machine learning model trained on millions of hand images.
# It takes an RGB image frame and outputs up to N sets of 21 3D landmarks.
# We wrap it to get a clean, simple interface.

import cv2
import mediapipe as mp   # Google's ML framework for hand/pose/face detection
import numpy as np       # NumPy: the Python library for fast array math
import logging
from dataclasses import dataclass, field
from typing import Optional, List

log = logging.getLogger(__name__)

# Landmark index constants
# MediaPipe gives 21 landmarks numbered 0-20. These constants give them names
# so our code reads like English instead of magic numbers.
# Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

WRIST           = 0
THUMB_CMC       = 1   # Thumb base (carpometacarpal joint)
THUMB_MCP       = 2   # Thumb knuckle
THUMB_IP        = 3   # Thumb middle joint
THUMB_TIP       = 4   # Thumb tip ← used for click detection

INDEX_MCP       = 5   # Index knuckle
INDEX_PIP       = 6   # Index middle joint
INDEX_DIP       = 7   # Index upper joint
INDEX_TIP       = 8   # Index tip ← controls the cursor

MIDDLE_MCP      = 9
MIDDLE_PIP      = 10
MIDDLE_DIP      = 11
MIDDLE_TIP      = 12  # Middle tip ← used for right-click

RING_MCP        = 13
RING_PIP        = 14
RING_DIP        = 15
RING_TIP        = 16

PINKY_MCP       = 17
PINKY_PIP       = 18
PINKY_DIP       = 19
PINKY_TIP       = 20


@dataclass
class HandData:
    """
    A clean data container for everything the Vision Engine knows about the hand.
    
    WHY a dataclass?
    A dataclass is a Python class that automatically creates __init__, __repr__,
    and other methods from the field definitions. It's perfect for data containers
    like this — less boilerplate, more clarity.
    """
    
    # Raw landmark positions as normalized coordinates (0.0 to 1.0)
    # Shape: (21, 3) — 21 landmarks, each with x, y, z
    landmarks_raw: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    
    # Smoothed landmark positions (after EMA/Kalman filtering)
    landmarks_smooth: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    
    # The index fingertip position in NORMALIZED coordinates (0.0 to 1.0)
    # This is the primary cursor input
    index_tip_norm: tuple = (0.5, 0.5)
    
    # The index fingertip in PIXEL coordinates (screen pixels)
    # This is what we pass to PyAutoGUI to move the cursor
    index_tip_px: tuple = (0, 0)
    
    # How confident is MediaPipe that it found a real hand? (0.0 to 1.0)
    # Low values mean the hand is occluded, far away, or in bad lighting.
    detection_confidence: float = 0.0
    
    # Was a hand detected in this frame?
    hand_detected: bool = False
    
    # "Left" or "Right" — MediaPipe can tell which hand it's seeing.
    # Note: MediaPipe labels from the camera's perspective, not the user's.
    handedness: str = "Unknown"


class HandTracker:
    """
    Detects and tracks one hand in video frames using MediaPipe.
    
    Example usage:
        tracker = HandTracker(min_detection_confidence=0.7)
        tracker.start()
        hand_data = tracker.process(frame_bgr)
        tracker.stop()
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Args:
            max_num_hands: Maximum hands to track simultaneously. We use 1 for
                          performance — tracking 2 hands doubles the CPU cost.
            min_detection_confidence: How sure MediaPipe must be before declaring
                          "I found a hand." Range: 0.0–1.0. Lower = more detections
                          but more false positives. 0.7 is the sweet spot.
            min_tracking_confidence: How sure it must be to continue tracking.
                          Lower than detection threshold is intentional — once found,
                          it's easier to keep tracking than to detect fresh.
            model_complexity: 0 = fastest (lite model), 1 = balanced, 2 = most accurate.
                          On most laptops, 1 gives ~30fps.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        
        # These are initialized in start() — MediaPipe requires explicit initialization
        self._hands = None
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

    def start(self):
        """Initialize the MediaPipe Hands model. Call this before process()."""
        log.info("Initializing MediaPipe Hands model...")
        
        # mp.solutions.hands.Hands() loads the neural network model from disk.
        # static_image_mode=False means it's optimized for video (uses tracking
        # between frames instead of detecting from scratch every frame).
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        log.info("MediaPipe Hands initialized")

    def process(self, frame_bgr: np.ndarray) -> HandData:
        """
        Process one video frame and extract hand landmark data.
        
        Args:
            frame_bgr: A BGR image as a numpy array (from OpenCV).
                      Shape should be (height, width, 3).
        
        Returns:
            HandData object with landmark positions and detection info.
        """
        hand_data = HandData()  # Start with default (no hand detected)
        
        if self._hands is None:
            log.error("HandTracker.start() must be called before process()")
            return hand_data
        
        # Step 1: Get image dimensions for coordinate conversion
        # frame_bgr.shape returns (height, width, channels)
        frame_h, frame_w = frame_bgr.shape[:2]
        
        # Step 2: Flip the frame horizontally (mirror effect)
        # WHY? Without flipping, moving your right hand moves the cursor left.
        # With flipping, it feels like a touchscreen mirror — natural.
        frame_bgr = cv2.flip(frame_bgr, 1)
        # flipCode=1 means flip around the y-axis (horizontal flip)
        
        # Step 3: Convert BGR to RGB
        # MediaPipe expects RGB. cv2.cvtColor converts between color spaces.
        # cv2.COLOR_BGR2RGB is a constant = "convert BGR to RGB"
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Step 4: Tell OpenCV not to write to this array.
        # MediaPipe internally may try to modify the frame. Making it not-writeable
        # forces a copy, which is actually FASTER because it avoids reference tracking.
        frame_rgb.flags.writeable = False
        
        # Step 5: Run the MediaPipe model
        # This is where the neural network actually runs. It takes ~5-15ms.
        results = self._hands.process(frame_rgb)
        
        # Allow writing again for drawing later
        frame_rgb.flags.writeable = True
        
        # Step 6: Extract landmark data if a hand was found
        # results.multi_hand_landmarks is None if no hands detected.
        # If hands ARE detected, it's a list of hand landmark objects.
        if results.multi_hand_landmarks:
            # We only care about the first hand (max_num_hands=1 anyway)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # results.multi_handedness is a list of handedness classifications
            if results.multi_handedness:
                handedness_info = results.multi_handedness[0]
                hand_data.handedness = handedness_info.classification[0].label
                hand_data.detection_confidence = handedness_info.classification[0].score
            
            # Extract all 21 landmarks into a NumPy array
            # landmark.x, .y, .z are normalized coordinates (0.0 to 1.0)
            raw = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ])
            # raw.shape == (21, 3) — 21 landmarks, each with x, y, z
            
            hand_data.landmarks_raw = raw
            hand_data.landmarks_smooth = raw.copy()  # Smoother will update this
            hand_data.hand_detected = True
            
            # Extract index fingertip (landmark 8) in normalized coords
            # Remember: after flipping, x=0 is left (camera right), x=1 is right
            idx_x = raw[INDEX_TIP][0]
            idx_y = raw[INDEX_TIP][1]
            hand_data.index_tip_norm = (idx_x, idx_y)
        
        return hand_data

    def draw_landmarks(self, frame: np.ndarray, hand_data: HandData) -> np.ndarray:
        """
        Draw the hand skeleton and landmark dots onto a frame.
        Uses pure OpenCV drawing — no dependency on MediaPipe's internal proto format.
        """
        if not hand_data.hand_detected:
            return frame

        frame_h, frame_w = frame.shape[:2]

        # Convert all normalized landmarks to pixel coordinates
        # landmark[i] = (x_norm, y_norm, z) → multiply by frame size to get pixels
        pts = []
        for lm in hand_data.landmarks_smooth:
            px = int(lm[0] * frame_w)
            py = int(lm[1] * frame_h)
            pts.append((px, py))

        # Draw connections (the "bones" of the hand skeleton)
        # MediaPipe defines which landmarks connect to which.
        # We hardcode the same connections here.
        CONNECTIONS = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17),
        ]

        for start_idx, end_idx in CONNECTIONS:
            cv2.line(
                frame,
                pts[start_idx],
                pts[end_idx],
                (180, 180, 180),   # Light gray lines for the skeleton
                2,                 # Line thickness in pixels
                cv2.LINE_AA        # Anti-aliased (smoother lines)
            )

        # Draw all 21 landmark dots
        for i, (px, py) in enumerate(pts):
            # Fingertips (4, 8, 12, 16, 20) get a slightly larger dot
            is_tip = i in (4, 8, 12, 16, 20)
            radius = 6 if is_tip else 4
            cv2.circle(frame, (px, py), radius, (255, 255, 255), -1)   # White fill
            cv2.circle(frame, (px, py), radius, (100, 100, 100), 1)    # Gray outline

        # Highlight index fingertip (landmark 8) the cursor point 
        ix, iy = pts[8]
        cv2.circle(frame, (ix, iy), 12, (0, 255, 255), -1)    # Solid cyan fill
        cv2.circle(frame, (ix, iy), 14, (0, 200, 200), 2)     # Darker cyan ring

        return frame

    def stop(self):
        # Clean up MediaPipe resources.
        if self._hands:
            self._hands.close()
            log.info("MediaPipe Hands closed")