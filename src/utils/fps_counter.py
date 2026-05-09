# FPSCounter, rolling-average FPS measurement.
# WHY rolling average (not instantaneous)?
# Instantaneous FPS (1 / last_frame_time) fluctuates wildly, you'd see
# "31fps... 28fps... 33fps..." jumping every frame.
# Rolling average smooths this over the last N frames, giving a stable reading.

import time
from collections import deque


class FPSCounter:
    """
    Measures actual frames-per-second using a rolling window average.
    
    Usage:
        fps_counter = FPSCounter(window=30)
        while running:
            fps_counter.tick()
            fps = fps_counter.fps
            # Display fps somewhere
    """

    def __init__(self, window: int = 30):
        """
        Args:
            window: How many recent frame times to average over.
                   30 = average over last 30 frames = ~1 second at 30fps.
                   Lower window = more reactive to slowdowns.
                   Higher window = more stable reading.
        """
        self.window = window
        
        # deque with maxlen automatically discards old entries when full
        # This is more efficient than a list that we'd have to slice
        self._timestamps: deque = deque(maxlen=window)
        
        self.fps: float = 0.0  # Current FPS (public, read this)

    def tick(self):
        """
        Record that a new frame was processed. Call once per frame.
        Updates self.fps automatically.
        """
        now = time.perf_counter()  # High-precision timer (better than time.time())
        self._timestamps.append(now)
        
        if len(self._timestamps) >= 2:
            # FPS = number of frames / time elapsed for those frames
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self.fps = (len(self._timestamps) - 1) / elapsed

    @property
    def fps_str(self) -> str:
        """Returns FPS as a formatted string like '29.8 fps'"""
        return f"{self.fps:.1f} fps"