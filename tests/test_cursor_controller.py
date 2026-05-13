from unittest.mock import patch, MagicMock
import pytest
from src.control.cursor_controller import CursorController


@pytest.fixture
def ctrl():
    #CursorController with mocked screen (1920x1080) and mocked pyautogui.
    with patch('src.control.cursor_controller.get_primary_screen',
               return_value={'x':0,'y':0,'width':1920,'height':1080}), \
         patch('src.control.cursor_controller.pyautogui') as mock_pg:
        c = CursorController()
        c.start()
        c._mock_pg = mock_pg
        yield c


def test_frozen_cursor_does_not_move(ctrl):
    # When frozen, cursor position must not change.
    before = ctrl.get_position()
    ctrl.update(0.9, 0.9, is_frozen=True)
    after = ctrl.get_position()
    assert before == after


def test_cursor_moves_toward_target(ctrl):
    #After several frames pointing to top-left, cursor should move there.
    for _ in range(20):
        ctrl.update(0.0, 0.0)
    px, py = ctrl.get_position()
    # Should have moved significantly toward top-left
    assert px < 500
    assert py < 400


def test_precision_mode_moves_slower(ctrl):
    #Precision mode cursor should travel less distance than normal mode.
    # Reset both controllers to same starting position
    c_normal    = CursorController()
    c_precision = CursorController()

    with patch('src.control.cursor_controller.get_primary_screen',
               return_value={'x':0,'y':0,'width':1920,'height':1080}), \
         patch('src.control.cursor_controller.pyautogui'):
        c_normal.start()
        c_precision.start()
        # Both start at center; move toward (0.9, 0.9)
        for _ in range(5):
            c_normal.update(0.9, 0.9, is_precision=False)
            c_precision.update(0.9, 0.9, is_precision=True)

    nx, ny = c_normal.get_position()
    px, py = c_precision.get_position()
    # Normal should be further from center (960, 540) than precision
    import math
    dist_normal    = math.hypot(nx - 960, ny - 540)
    dist_precision = math.hypot(px - 960, py - 540)
    assert dist_normal > dist_precision


def test_cursor_clamped_to_screen(ctrl):
    # Cursor must never go outside screen bounds.
    ctrl.update(2.0, 2.0)   # Way outside normalized range
    px, py = ctrl.get_position()
    assert 0 <= px <= 1920
    assert 0 <= py <= 1080