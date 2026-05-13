import pyautogui
import logging
import time
from src.utils.monitor_utils import get_primary_screen

log = logging.getLogger(__name__)
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

class CursorController:

    def __init__(self, config: dict = None):
        cfg = config or {}
        c   = cfg.get('cursor', {})

        self.smoothing    = c.get('smoothing_factor', 0.65)
        # dpi_scale: multiplies finger movement so less hand motion is needed.
        # 1.0 = hand must sweep full frame. 2.8 = ~1/3 of frame covers full screen.
        self.dpi_scale    = c.get('dpi_scale', 2.8)
        self.stab_radius  = c.get('stabilization_radius', 5)
        self.edge_accel   = c.get('edge_acceleration', 1.4)
        # margin: outer fraction of camera frame that maps to screen edges.
        # 0.04 = very small border, so almost all hand movement is usable.
        self.margin       = c.get('margin', 0.04)

        self._screen_w: float = 1920.0
        self._screen_h: float = 1080.0
        self._cur_x:    float = 960.0
        self._cur_y:    float = 540.0
        self._prev_norm_x: float = 0.5
        self._prev_norm_y: float = 0.5
        self._prev_time:   float = 0.0

    def start(self):
        screen = get_primary_screen()
        self._screen_w = float(screen['width'])
        self._screen_h = float(screen['height'])
        self._cur_x    = self._screen_w / 2.0
        self._cur_y    = self._screen_h / 2.0
        log.info(f"CursorController: {self._screen_w}x{self._screen_h} "
                 f"dpi_scale={self.dpi_scale} margin={self.margin}")

    def update(self, norm_x: float, norm_y: float,
               is_frozen: bool = False) -> tuple:

        if is_frozen:
            return (int(self._cur_x), int(self._cur_y))

        now = time.perf_counter()
        dt  = max(now - self._prev_time, 0.001)

        # Map with margin
        mx       = self.margin
        mapped_x = (norm_x - mx) / max(1.0 - 2.0 * mx, 0.01)
        mapped_y = (norm_y - mx) / max(1.0 - 2.0 * mx, 0.01)

        # Apply dpi_scale: amplify movement around center (0.5)
        # Formula: center + (position - center) * scale
        # This makes small movements near center cover more screen distance.
        scaled_x = 0.5 + (mapped_x - 0.5) * self.dpi_scale
        scaled_y = 0.5 + (mapped_y - 0.5) * self.dpi_scale

        # Clamp to [0,1]
        scaled_x = max(0.0, min(1.0, scaled_x))
        scaled_y = max(0.0, min(1.0, scaled_y))

        target_x = scaled_x * self._screen_w
        target_y = scaled_y * self._screen_h

        # Edge acceleration
        edge_zone = 0.08
        near_edge = (scaled_x < edge_zone or scaled_x > 1.0 - edge_zone or
                     scaled_y < edge_zone or scaled_y > 1.0 - edge_zone)
        alpha = min(0.92, self.smoothing * (self.edge_accel if near_edge else 1.0))

        new_x = alpha * target_x + (1.0 - alpha) * self._cur_x
        new_y = alpha * target_y + (1.0 - alpha) * self._cur_y

        # Stabilization deadzone
        dx   = new_x - self._cur_x
        dy   = new_y - self._cur_y
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < self.stab_radius:
            self._prev_time    = now
            self._prev_norm_x  = norm_x
            self._prev_norm_y  = norm_y
            return (int(self._cur_x), int(self._cur_y))

        self._cur_x = new_x
        self._cur_y = new_y

        px = int(max(0, min(self._screen_w - 1, self._cur_x)))
        py = int(max(0, min(self._screen_h - 1, self._cur_y)))

        try:
            pyautogui.moveTo(px, py, duration=0)
        except Exception as e:
            log.warning(f"moveTo failed: {e}")

        self._prev_time   = now
        self._prev_norm_x = norm_x
        self._prev_norm_y = norm_y
        return (px, py)

    def get_position(self) -> tuple:
        return (int(self._cur_x), int(self._cur_y))

    def stop(self):
        log.info("CursorController stopped")