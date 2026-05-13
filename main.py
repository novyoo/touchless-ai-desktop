import sys
import logging
import time
import cv2
from src.vision.gesture_classifier import Gesture, GestureState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("touchless.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("main")

def main():
    log.info("Touchless AI Desktop starting...")

    from src.config.settings            import load_config
    from src.vision.capture             import WebcamCapture
    from src.vision.hand_tracker        import HandTracker
    from src.vision.landmark_smoother   import IndexTipSmoother
    from src.vision.gesture_classifier  import GestureClassifier
    from src.control.system_controller  import SystemController
    from src.utils.fps_counter          import FPSCounter

    config  = load_config()
    cam_cfg = config.get('camera',    {})
    mp_cfg  = config.get('mediapipe', {})

    cap = WebcamCapture(
        device_id = cam_cfg.get('device_id', 0),
        width     = cam_cfg.get('width', 1280),
        height    = cam_cfg.get('height', 720),
        fps       = cam_cfg.get('fps', 30),
    )
    tracker = HandTracker(
        max_num_hands            = mp_cfg.get('max_num_hands', 2),
        min_detection_confidence = mp_cfg.get('min_detection_confidence', 0.7),
        min_tracking_confidence  = mp_cfg.get('min_tracking_confidence', 0.5),
        model_complexity         = mp_cfg.get('mediapipe_complexity', 1),
    )

    # Two smoothers — one per hand
    smoother_r = IndexTipSmoother(min_cutoff=1.0, beta=0.007)
    smoother_l = IndexTipSmoother(min_cutoff=1.0, beta=0.007)

    classifier = GestureClassifier(config)
    # Read pause duration for HUD display
    PAUSE_HOLD_SECS = config.get('gestures', {}).get('pause_hold_duration', 1.0)
    SCROLL_HOLD_SECS = config.get('gestures', {}).get('scroll_hold_duration', 2.0)
    controller = SystemController(config)
    fps_ctr    = FPSCounter(window=30)

    if not cap.start():
        log.error("Cannot open webcam. Exiting.")
        sys.exit(1)

    tracker.start()
    classifier.start()
    controller.start()

    log.info("Running. Q=quit in debug window.")
    show_debug = True

    try:
        while True:
            frame = cap.read()
            if frame is None:
                continue

            # Vision — returns TwoHandData
            two_hand = tracker.process(frame)

            # Smooth each hand's index tip independently
            t = time.perf_counter()
            if two_hand.right_hand.hand_detected:
                rx, ry = two_hand.right_hand.index_tip_norm
                rx, ry = smoother_r.smooth(rx, ry, t)
                two_hand.right_hand.index_tip_norm   = (rx, ry)
                two_hand.right_hand.landmarks_smooth = two_hand.right_hand.landmarks_raw.copy()
            else:
                smoother_r.reset()

            if two_hand.left_hand.hand_detected:
                lx, ly = two_hand.left_hand.index_tip_norm
                lx, ly = smoother_l.smooth(lx, ly, t)
                two_hand.left_hand.index_tip_norm   = (lx, ly)
                two_hand.left_hand.landmarks_smooth = two_hand.left_hand.landmarks_raw.copy()
            else:
                smoother_l.reset()

            # Gesture classification
            gesture_result = classifier.process(two_hand)

            # System control
            controller.update(two_hand, gesture_result)

            fps_ctr.tick()

            # Debug window
            if show_debug:
                display = cv2.flip(frame, 1)
                # Draw right hand in cyan, left hand in orange
                display = tracker.draw_landmarks(
                    display, two_hand.right_hand, color_tip=(0, 255, 255))
                display = tracker.draw_landmarks(
                    display, two_hand.left_hand, color_tip=(0, 165, 255))
                _draw_hud(display, fps_ctr, gesture_result, two_hand, PAUSE_HOLD_SECS, SCROLL_HOLD_SECS)
                cv2.imshow("Touchless AI [Q=quit, D=HUD]", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == ord('d'):
                show_debug = not show_debug

    finally:
        controller.stop()
        tracker.stop()
        cap.release()
        cv2.destroyAllWindows()
        log.info("Clean shutdown.")

def _draw_hud(frame, fps_ctr, gesture_result, two_hand, pause_secs=1.0, scroll_hold_secs=2.0):
    h, w = frame.shape[:2]

    def txt(text, x, y, color=(255, 255, 255), scale=0.46):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

    def panel(x, y, pw, ph, alpha=0.55):
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x + pw, y + ph), (0, 0, 0), -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    def hold_bar(x, y, bar_w, progress, held_secs, total_secs, color):
        """
        progress   : 0.0 → 1.0 (how full the bar is)
        held_secs  : seconds elapsed (shown as text)
        total_secs : seconds required (shown as text)
        """
        bar_h  = 7
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), color, -1)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (100, 100, 100), 1)
        label = f"{held_secs:.1f}/{total_secs:.1f}s"
        cv2.putText(frame, label,
                    (x + bar_w + 6, y + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    rg = gesture_result.right
    lg = gesture_result.left

    # Decide panel height dynamically based on active bars 
    bars_to_show = []

    # Palm hold bar
    if rg.gesture == Gesture.PALM and rg.state in (GestureState.HOLDING, GestureState.ACTIVE):
        bars_to_show.append(('palm', rg.hold_duration / pause_secs,
                              rg.hold_duration, pause_secs,
                              "PALM HOLD", (0, 200, 255)))

    # Scroll up building bar
    if gesture_result.scroll_up_progress > 0.0 and gesture_result.scroll_dy == 0.0:
        held = gesture_result.scroll_up_progress * scroll_hold_secs
        bars_to_show.append(('scroll_up', gesture_result.scroll_up_progress,
                              held, scroll_hold_secs,
                              "SCROLL UP", (100, 255, 100)))

    # Scroll down building bar
    if gesture_result.scroll_down_progress > 0.0 and gesture_result.scroll_dy == 0.0:
        held = gesture_result.scroll_down_progress * scroll_hold_secs
        bars_to_show.append(('scroll_dn', gesture_result.scroll_down_progress,
                              held, scroll_hold_secs,
                              "SCROLL DN", (100, 100, 255)))

    base_h   = 82
    panel_h  = base_h + len(bars_to_show) * 22
    panel(0, 0, 265, panel_h)

    # Status lines 
    r_col  = (0, 255, 255) if two_hand.right_hand.hand_detected else (80, 80, 80)
    l_col  = (0, 165, 255) if two_hand.left_hand.hand_detected  else (80, 80, 80)
    r_name = rg.gesture.name if two_hand.right_hand.hand_detected else "—"
    l_name = lg.gesture.name if two_hand.left_hand.hand_detected  else "—"

    txt(f"FPS  {fps_ctr.fps:.0f}", 10, 22, (0, 255, 200))
    txt(f"R  {r_name}", 10, 46, r_col)
    txt(f"L  {l_name}", 10, 68, l_col)

    # Hold bars 
    bar_y = base_h - 4
    for _, progress, held, total, label, color in bars_to_show:
        txt(label, 10, bar_y + 6, color, scale=0.38)
        hold_bar(x=90, y=bar_y - 2, bar_w=130,
                 progress=progress, held_secs=held,
                 total_secs=total, color=color)
        bar_y += 22

    # Drag indicator 
    if gesture_result.is_dragging:
        txt("DRAGGING", 10, panel_h + 16, (0, 255, 50))

    # Scroll active indicator 
    if gesture_result.scroll_dy != 0.0:
        direction = "UP ↑" if gesture_result.scroll_dy < 0 else "DOWN ↓"
        s_col = (100, 255, 100) if gesture_result.scroll_dy < 0 else (100, 100, 255)
        txt(f"SCROLLING {direction}", 10, panel_h + 16, s_col)

    # Paused banner 
    if gesture_result.tracking_paused:
        bw, bh = 270, 32
        panel(w // 2 - bw // 2, 8, bw, bh, alpha=0.75)
        txt("PAUSED — open palm to resume",
            w // 2 - bw // 2 + 8, 28, (0, 120, 255), scale=0.42)

    # Bottom-right controls reference 
    lines = [
        ("CONTROLS",                   (0, 255, 200)),
        ("RIGHT HAND",                 (0, 255, 255)),
        (" index only   = cursor",     (200, 200, 200)),
        (" open palm 1s = pause",      (200, 200, 200)),
        ("LEFT HAND",                  (0, 165, 255)),
        (" thumb+index  = L-click",    (200, 200, 200)),
        (" thumb+mid    = R-click",    (200, 200, 200)),
        ("BOTH HANDS",                 (180, 180, 0)),
        (" both index   = scroll up",  (200, 200, 200)),
        (" both peace   = scroll dn",  (200, 200, 200)),
        (" R-point+L-hold = drag",     (200, 200, 200)),
    ]
    lh_line = 20
    pw      = 230
    ph      = len(lines) * lh_line + 10
    px      = w - pw - 8
    py      = h - ph - 8
    panel(px, py, pw, ph)
    for i, (line, col) in enumerate(lines):
        txt(line, px + 8, py + 16 + i * lh_line, col, scale=0.40)

    # Status dot
    cv2.circle(frame, (w - 16, 16), 7,
               (0, 255, 100) if (two_hand.right_hand.hand_detected or
                                 two_hand.left_hand.hand_detected) else (80, 80, 80), -1)
if __name__ == "__main__":
    main()