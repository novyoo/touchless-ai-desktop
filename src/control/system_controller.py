import logging
import time
from src.control.cursor_controller import CursorController
from src.control.click_handler     import ClickHandler
from src.control.scroll_handler    import ScrollHandler
from src.control.system_actions    import SystemActions
from src.vision.gesture_classifier import Gesture, GestureState, GestureResult

log = logging.getLogger(__name__)

class SystemController:

    def __init__(self, config: dict = None):
        self._config   = config or {}
        self._cursor   = CursorController(config)
        self._clicker  = ClickHandler(config)
        self._scroller = ScrollHandler(config)
        self._actions  = SystemActions()
        self._action_times: list = []
        self._max_per_sec = 30

    def start(self):
        self._cursor.start()
        log.info("SystemController ready")

    def update(self, two_hand_data, gesture_result: GestureResult):
        now = time.perf_counter()
        self._action_times = [t for t in self._action_times if now - t < 1.0]
        if len(self._action_times) >= self._max_per_sec:
            return
        self._action_times.append(now)

        if gesture_result.tracking_paused:
            return

        # Cursor: RIGHT hand, ONLY when gesture is POINTING 
        # If right hand is showing palm, peace, or any other pose - don't move cursor.
        rh = two_hand_data.right_hand
        if (rh.hand_detected and
                gesture_result.right.gesture == Gesture.POINTING and
                gesture_result.right.state   == GestureState.ACTIVE):
            norm_x, norm_y = rh.index_tip_norm
            cursor_pos     = self._cursor.update(norm_x, norm_y, is_frozen=False)
        else:
            cursor_pos = self._cursor.get_position()

        # Scroll 
        if gesture_result.scroll_dy != 0.0:
            self._scroller.update(0.0, gesture_result.scroll_dy)
        else:
            self._scroller.reset()

        # Drag 
        if gesture_result.is_dragging:
            self._clicker.start_drag_if_needed(cursor_pos)
        else:
            self._clicker.end_drag_if_needed(cursor_pos)
            self._clicker.update(gesture_result.left, cursor_pos)

    def execute_voice_action(self, action: str, payload: str = ""):
        action = action.lower().strip()
        if action == "screenshot":      self._actions.take_screenshot()
        elif action == "volume_up":     self._actions.volume_up()
        elif action == "volume_down":   self._actions.volume_down()
        elif action == "mute":          self._actions.mute()
        elif action == "open" and payload: self._actions.launch_app(payload)
        else:
            log.warning(f"Unknown voice action: '{action}'")

    def stop(self):
        self._clicker.stop()
        self._scroller.stop()
        self._cursor.stop()
        log.info("SystemController stopped")