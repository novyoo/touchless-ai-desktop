# tests/test_landmark_smoother.py
import numpy as np
import time
import pytest
from src.vision.landmark_smoother import EMALandmarkSmoother, IndexTipSmoother


class TestEMALandmarkSmoother:
    
    def test_first_frame_returns_raw(self):
        """First frame has no previous data — should return raw input unchanged."""
        smoother = EMALandmarkSmoother(alpha=0.5)
        raw = np.ones((21, 3)) * 0.5
        result = smoother.smooth(raw)
        np.testing.assert_array_almost_equal(result, raw)
    
    def test_smoothing_moves_toward_new_value(self):
        """After first frame, output should be between old and new value."""
        smoother = EMALandmarkSmoother(alpha=0.5)
        # Frame 1: all zeros
        raw1 = np.zeros((21, 3))
        smoother.smooth(raw1)
        # Frame 2: all ones — smoothed should be 0.5 (midpoint)
        raw2 = np.ones((21, 3))
        result = smoother.smooth(raw2)
        np.testing.assert_array_almost_equal(result, np.ones((21, 3)) * 0.5)
    
    def test_high_alpha_is_more_responsive(self):
        """Higher alpha = closer to raw value = more responsive."""
        s_low  = EMALandmarkSmoother(alpha=0.2)
        s_high = EMALandmarkSmoother(alpha=0.8)
        raw1 = np.zeros((21, 3))
        raw2 = np.ones((21, 3))
        s_low.smooth(raw1)
        s_high.smooth(raw1)
        r_low  = s_low.smooth(raw2)
        r_high = s_high.smooth(raw2)
        # High alpha should be closer to 1.0 (the new raw value)
        assert r_high[0, 0] > r_low[0, 0]
    
    def test_reset_clears_state(self):
        """After reset, next frame behaves like first frame."""
        smoother = EMALandmarkSmoother(alpha=0.5)
        raw = np.ones((21, 3)) * 0.5
        smoother.smooth(raw)
        smoother.reset()
        raw2 = np.ones((21, 3))
        result = smoother.smooth(raw2)
        # After reset, result should equal raw2 (no previous state)
        np.testing.assert_array_almost_equal(result, raw2)
    
    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            EMALandmarkSmoother(alpha=0.0)
        with pytest.raises(ValueError):
            EMALandmarkSmoother(alpha=1.5)


class TestIndexTipSmoother:
    
    def test_deadzone_holds_cursor_still(self):
        """Tiny movements within deadzone should NOT change output."""
        smoother = IndexTipSmoother(deadzone_norm=0.05)
        t = time.perf_counter()
        # First position
        x1, y1 = smoother.smooth(0.5, 0.5, t)
        # Tiny movement (much less than 0.05 deadzone)
        x2, y2 = smoother.smooth(0.501, 0.501, t + 0.033)
        # Should NOT have moved
        assert x2 == x1
        assert y2 == y1
    
    def test_large_movement_updates_position(self):
        """Movement beyond deadzone SHOULD update the output — but the 1€ filter
        introduces intentional lag, so we feed several frames to let it converge."""
        smoother = IndexTipSmoother(deadzone_norm=0.01)
        t = time.perf_counter()
        
        # Feed 10 frames of position (0.9, 0.9) at 30fps intervals
        # The filter converges toward the target over multiple frames
        x_out, y_out = 0.1, 0.1
        for i in range(10):
            x_out, y_out = smoother.smooth(0.9, 0.9, t + i * 0.033)
        
        # After 10 frames the filter should have moved well past the midpoint
        assert x_out > 0.5, f"Expected x > 0.5 after convergence, got {x_out}"
        assert y_out > 0.5, f"Expected y > 0.5 after convergence, got {y_out}"