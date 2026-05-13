import pyautogui
import time
import logging
from src.vision.gesture_classifier import Gesture, GestureState

log = logging.getLogger(__name__)
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

class ClickHandler:

    def __init__(self, config: dict = None):
        self.click_cooldown  = 0.4
        self._dragging       = False
        self._last_left:  float = 0.0
        self._last_right: float = 0.0

    def update(self, hand_gesture_result, cursor_pos: tuple):
        #Process left-hand gesture result for click actions.
        if self._dragging:
            return   # Don't fire clicks while dragging

        now     = time.perf_counter()
        gesture = hand_gesture_result.gesture

        if (gesture == Gesture.LEFT_CLICK and
                hand_gesture_result.just_triggered and
                now - self._last_left > self.click_cooldown):
            try:
                pyautogui.click(button='left')
                self._last_left = now
                log.debug(f"LEFT CLICK at {cursor_pos}")
            except Exception as e:
                log.warning(f"Left click failed: {e}")

        elif (gesture == Gesture.RIGHT_CLICK and
                hand_gesture_result.just_triggered and
                now - self._last_right > self.click_cooldown):
            try:
                pyautogui.click(button='right')
                self._last_right = now
                log.debug(f"RIGHT CLICK at {cursor_pos}")
            except Exception as e:
                log.warning(f"Right click failed: {e}")

    def start_drag_if_needed(self, cursor_pos: tuple):
        if not self._dragging:
            try:
                pyautogui.mouseDown(button='left')
                self._dragging = True
                log.debug(f"DRAG START at {cursor_pos}")
            except Exception as e:
                log.warning(f"Drag start failed: {e}")

    def end_drag_if_needed(self, cursor_pos: tuple):
        if self._dragging:
            try:
                pyautogui.mouseUp(button='left')
                self._dragging = False
                log.debug(f"DRAG END at {cursor_pos}")
            except Exception as e:
                log.warning(f"Drag end failed: {e}")

    def force_release(self):
        if self._dragging:
            try:
                pyautogui.mouseUp(button='left')
                self._dragging = False
            except Exception:
                pass

    def stop(self):
        self.force_release()