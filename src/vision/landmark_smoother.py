# LandmarkSmoother — reduces jitter in landmark positions.
# TWO smoothing algorithms:
# 1. EMA (Exponential Moving Average) simple and fast
#    Formula: smoothed = alpha * raw + (1 - alpha) * previous_smoothed
#    Alpha = 0.3: very smooth, slight lag (good for scrolling)
#    Alpha = 0.7: more responsive, slight jitter (good for clicking)
# 2. Kalman Filter predictive, better for fast movements
#    Models the finger as a physical object with position and velocity.
#    Predicts where it WILL be, then corrects when it sees where it IS.
#    Handles sudden direction changes better than EMA.
# WHY BOTH? We use EMA for simplicity and Kalman optionally for precision mode.

import numpy as np
import logging
from typing import Optional
from collections import deque

log = logging.getLogger(__name__)


class EMALandmarkSmoother:
    """
    Exponential Moving Average smoother for hand landmarks.
    
    This is the primary smoother. It's fast (just a multiply and add per landmark)
    and adds negligible latency (~1-2ms).
    """

    def __init__(self, alpha: float = 0.5, num_landmarks: int = 21):
        """
        Args:
            alpha: Smoothing factor. 0.0 = frozen (no new data).
                   1.0 = no smoothing (raw data).
                   Recommended: 0.4–0.6 for cursor, 0.6–0.8 for gestures.
            num_landmarks: How many landmark points to track (21 for a hand).
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Alpha must be in (0.0, 1.0], got {alpha}")
        
        self.alpha = alpha
        self.num_landmarks = num_landmarks
        
        # _prev stores the last smoothed position.
        # None means "no previous data yet" — first frame uses raw data directly.
        self._prev: Optional[np.ndarray] = None

    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to a set of landmarks.
        
        Args:
            landmarks: Raw landmark array, shape (21, 3). X, Y, Z per point.
        
        Returns:
            Smoothed landmark array, same shape.
        """
        if self._prev is None:
            # First frame — nothing to blend with, just store and return raw
            self._prev = landmarks.copy()
            return landmarks.copy()
        
        # EMA formula: new_smooth = alpha * raw + (1 - alpha) * previous_smooth
        # NumPy applies this element-wise across all 21*3 = 63 values at once
        smoothed = self.alpha * landmarks + (1.0 - self.alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed

    def reset(self):
        """Reset smoother state. Call when tracking is lost and resumes."""
        self._prev = None
        log.debug("EMA smoother reset")


class OneEuroFilter:
    """
    One Euro Filter — a sophisticated adaptive filter that adjusts its smoothing
    strength based on how fast the finger is moving.
    
    WHY BETTER THAN PLAIN EMA?
    EMA has a fixed smoothing amount. The 1€ filter has two modes:
    - When moving SLOWLY (precise positioning): applies heavy smoothing to remove jitter
    - When moving FAST (crossing the screen): reduces smoothing to prevent lag
    
    This is exactly what you want for cursor control: stable when still, responsive
    when moving fast. The name comes from its bandwidth in Hertz (1 Hz ≈ 1 change/second).
    
    Parameters are tuned for hand tracking at 30fps:
        min_cutoff: Lower = smoother when still. 0.5–2.0 Hz.
        beta: Higher = faster response during fast movement. 0.001–0.1.
        d_cutoff: Derivative cutoff. Usually fixed at 1.0 Hz.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.008, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # State for each tracked value
        self._x_prev = None    # Previous filtered value
        self._dx_prev = 0.0    # Previous filtered derivative (velocity)
        self._t_prev = None    # Previous timestamp

    def _alpha(self, cutoff: float, dt: float) -> float:
        """Compute alpha from cutoff frequency and timestep."""
        # tau = time constant, alpha = EMA factor derived from it
        import math
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: float) -> float:
        """
        Filter a single scalar value.
        
        Args:
            x: The raw value (e.g., landmark x coordinate)
            timestamp: Current time in seconds
        
        Returns:
            Filtered value
        """
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = timestamp
            return x
        
        dt = max(timestamp - self._t_prev, 1e-6)  # Avoid division by zero
        
        # Compute derivative (how fast is the value changing?)
        dx = (x - self._x_prev) / dt
        
        # Filter the derivative
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        
        # Adaptive cutoff: faster movement → higher cutoff → less smoothing → more responsive
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter the value
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        
        # Store state for next frame
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp
        
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class IndexTipSmoother:
    """
    Specialized smoother for the index fingertip (the cursor control point).
    
    Uses the 1€ filter for adaptive smoothing — heavy when still (prevents jitter),
    light when moving fast (prevents lag).
    
    Also implements a STABILIZATION BUBBLE: if the finger moves less than
    `deadzone_px` pixels, the cursor doesn't move at all. This makes small
    tremors invisible.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.008, deadzone_norm: float = 0.003):
        """
        Args:
            min_cutoff: Smoothing strength when still. Lower = smoother but more lag.
            beta: Responsiveness when fast. Higher = more responsive.
            deadzone_norm: Normalized distance below which movement is ignored.
                          0.003 ≈ 3 pixels on a 1000px-wide screen. Prevents micro-jitter.
        """
        # One filter per axis
        self._filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.deadzone_norm = deadzone_norm
        self._last_output = None

    def smooth(self, x: float, y: float, timestamp: float) -> tuple:
        """
        Smooth the index fingertip position.
        
        Args:
            x, y: Normalized coordinates (0.0 to 1.0)
            timestamp: Current time.time() value
        
        Returns:
            (smoothed_x, smoothed_y) tuple
        """
        sx = self._filter_x.filter(x, timestamp)
        sy = self._filter_y.filter(y, timestamp)
        
        if self._last_output is None:
            self._last_output = (sx, sy)
            return (sx, sy)
        
        # Deadzone check: only update if we've moved beyond the threshold
        dx = sx - self._last_output[0]
        dy = sy - self._last_output[1]
        dist = (dx * dx + dy * dy) ** 0.5  # Euclidean distance
        
        if dist < self.deadzone_norm:
            # Movement too small — hold the cursor still
            return self._last_output
        
        self._last_output = (sx, sy)
        return (sx, sy)

    def reset(self):
        self._filter_x.reset()
        self._filter_y.reset()
        self._last_output = None