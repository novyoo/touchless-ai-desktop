# DebugRunner — runs the Vision Engine and shows a live annotated preview.
# This is your Phase 1 test: run this file and you should see your webcam
# feed with hand landmarks drawn on it.
# Controls (when the preview window is active):
#   Q or ESC  → quit
#   S         → toggle skeleton drawing on/off
#   D         → toggle debug info overlay

import cv2
import time
import logging
import sys
import os

# Add project root to Python path so we can import src.*
# WHY? When you run `python src/vision/debug_runner.py`, Python's path starts
# at src/vision/. Adding the project root lets us do `from src.vision.capture import...`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.vision.capture import WebcamCapture
from src.vision.hand_tracker import HandTracker, INDEX_TIP
from src.vision.landmark_smoother import IndexTipSmoother
from src.utils.fps_counter import FPSCounter
from src.config.settings import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("debug_runner")


def run_vision_debug():
    """Main debug loop for the Vision Engine."""
    
    # Load configuration from defaults.yaml
    config = load_config()
    cam_cfg = config.get('camera', {})
    mp_cfg  = config.get('mediapipe', {})
    
    log.info("Starting Vision Engine debug mode...")
    
    # Initialize all components
    cap = WebcamCapture(
        device_id=cam_cfg.get('device_id', 0),
        width=cam_cfg.get('width', 1280),
        height=cam_cfg.get('height', 720),
        fps=cam_cfg.get('fps', 30),
    )
    
    tracker = HandTracker(
        max_num_hands=mp_cfg.get('max_num_hands', 1),
        min_detection_confidence=mp_cfg.get('min_detection_confidence', 0.7),
        min_tracking_confidence=mp_cfg.get('min_tracking_confidence', 0.5),
        model_complexity=mp_cfg.get('mediapipe_complexity', 1),
    )
    
    smoother = IndexTipSmoother(min_cutoff=1.0, beta=0.007)
    fps_counter = FPSCounter(window=30)
    
    # Start components
    if not cap.start():
        log.error("Could not open webcam. Check that no other app is using it.")
        return
    
    tracker.start()
    
    # UI state
    show_skeleton = True
    show_debug    = True
    
    log.info("Vision Engine running. Press Q or ESC in the preview window to quit.")
    
    try:
        while True:
            # Get frame 
            frame = cap.read()
            if frame is None:
                log.warning("Empty frame — skipping")
                continue
            
            # Process hand 
            t_start = time.perf_counter()
            hand_data = tracker.process(frame)
            t_track = (time.perf_counter() - t_start) * 1000  # Convert to ms
            
            # Apply smoothing to index fingertip
            if hand_data.hand_detected:
                raw_x, raw_y = hand_data.index_tip_norm
                sx, sy = smoother.smooth(raw_x, raw_y, time.perf_counter())
                hand_data.index_tip_norm = (sx, sy)
                hand_data.landmarks_smooth = hand_data.landmarks_raw.copy()
                # (Full landmark smoothing added in Phase 2 when we need it)
            else:
                smoother.reset()  # Reset when hand disappears
            
            fps_counter.tick()
            
            # Draw on frame
            # Flip the display frame to match the processed frame
            display = cv2.flip(frame, 1)
            
            if show_skeleton and hand_data.hand_detected:
                display = tracker.draw_landmarks(display, hand_data)
            
            if show_debug:
                _draw_debug_overlay(display, hand_data, fps_counter, t_track)
            
            # Show window 
            cv2.imshow("Touchless AI — Vision Engine Debug [Q=quit]", display)
            
            # Key handling 
            # cv2.waitKey(1) waits 1ms for a keypress. Returns -1 if no key.
            # We use & 0xFF to get the last byte (handles some platform quirks).
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                log.info("Quit key pressed")
                break
            elif key == ord('s'):
                show_skeleton = not show_skeleton
                log.info(f"Skeleton display: {show_skeleton}")
            elif key == ord('d'):
                show_debug = not show_debug
    
    finally:
        # ALWAYS clean up, even if an exception occurs
        tracker.stop()
        cap.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows
        log.info("Vision Engine debug stopped cleanly")


def _draw_debug_overlay(frame: object, hand_data, fps_counter, latency_ms: float):
    """
    Draw the debug HUD onto the frame.
    
    This shows all the raw metrics — exactly what the Explainability HUD
    will show in the final overlay.
    """
    h, w = frame.shape[:2]
    
    # Background panel 
    # Draw a semi-transparent dark rectangle so text is readable
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 200), (0, 0, 0), -1)
    # Blend: alpha=0.6 means 60% overlay + 40% original frame = semi-transparent
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Helper: draw text with outline 
    def put_text(text, y, color=(255, 255, 255)):
        # Draw black outline first (for contrast on any background)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        # Then draw white text on top
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    
    # Metrics 
    status_color = (0, 255, 100) if hand_data.hand_detected else (100, 100, 255)
    
    put_text(f"FPS: {fps_counter.fps:.1f}", 28, (0, 255, 200))
    put_text(f"Latency: {latency_ms:.1f}ms", 52)
    
    if hand_data.hand_detected:
        ix, iy = hand_data.index_tip_norm
        put_text(f"Hand: {hand_data.handedness} ({hand_data.detection_confidence:.2f})", 76, (0, 255, 100))
        put_text(f"Index tip: ({ix:.3f}, {iy:.3f})", 100)
        put_text(f"Index px: {hand_data.index_tip_px}", 124)
        put_text(f"Landmarks: 21 detected", 148, (200, 200, 200))
    else:
        put_text("Hand: NOT DETECTED", 76, (100, 100, 255))
        put_text("Show your hand to the camera", 100, (150, 150, 150))
    
    # Corner status indicator 
    # Big colored dot: green = tracking, blue = not detected
    dot_color = (0, 255, 100) if hand_data.hand_detected else (100, 100, 255)
    cv2.circle(frame, (w - 25, 25), 10, dot_color, -1)
    
    # Key hints at bottom
    cv2.putText(frame, "S=skeleton  D=debug  Q=quit",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


if __name__ == "__main__":
    run_vision_debug()