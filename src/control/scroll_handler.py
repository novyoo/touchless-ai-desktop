import pyautogui
import time
import logging

log = logging.getLogger(__name__)
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0


class ScrollHandler:
    """
    Smooth continuous scroll.
    Fires every frame while scroll gesture is active.
    Rate-limited to avoid OS scroll queue flooding.
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get('gestures', {})
        # How many scroll clicks to fire per second while gesture is held.
        # 8 = smooth and fast. 4 = slow. 12 = very fast.
        self._clicks_per_sec: float = 28.0
        self._last_scroll:    float = 0.0
        self._interval:       float = 1.0 / self._clicks_per_sec

    def update(self, scroll_dx: float, scroll_dy: float):
        """
        Call every frame while scroll gesture is active.
        Internally rate-limits to _clicks_per_sec.
        scroll_dy: negative = up, positive = down.
        """
        if abs(scroll_dx) < 0.0001 and abs(scroll_dy) < 0.0001:
            return

        now = time.perf_counter()
        if now - self._last_scroll < self._interval:
            return   # Not time yet — skip this frame

        self._last_scroll = now

        if abs(scroll_dy) > abs(scroll_dx):
            # Vertical scroll
            # pyautogui.scroll: positive = up, negative = down
            # Our scroll_dy: negative = up, positive = down → negate
            clicks = -10 if scroll_dy > 0 else 10
            try:
                pyautogui.scroll(clicks)
            except Exception as e:
                log.warning(f"Scroll failed: {e}")
        else:
            # Horizontal scroll
            clicks = 10 if scroll_dx > 0 else -10
            try:
                pyautogui.hscroll(clicks)
            except Exception as e:
                log.warning(f"HScroll failed: {e}")

    def reset(self):
        pass   # Nothing to clear - rate limiter resets naturally

    def stop(self):
        log.info("ScrollHandler stopped")