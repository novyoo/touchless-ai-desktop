# Detects screen geometry for correct cursor coordinate mapping.
# We use screeninfo with a safe fallback to pyautogui if it fails.

import logging
log = logging.getLogger(__name__)


def get_primary_screen() -> dict:
    """
    Returns primary screen geometry as:
        {'x': 0, 'y': 0, 'width': 1920, 'height': 1080}

    Tries screeninfo first (supports multi-monitor).
    Falls back to pyautogui.size() if screeninfo fails.
    """
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        # Primary monitor is typically the one at (0, 0)
        for m in monitors:
            if m.x == 0 and m.y == 0:
                log.info(f"Primary screen: {m.width}x{m.height} at ({m.x},{m.y})")
                return {'x': m.x, 'y': m.y, 'width': m.width, 'height': m.height}
        # If no monitor at (0,0), just use the first one
        m = monitors[0]
        log.info(f"Using first monitor: {m.width}x{m.height}")
        return {'x': m.x, 'y': m.y, 'width': m.width, 'height': m.height}
    except Exception as e:
        log.warning(f"screeninfo failed ({e}), falling back to pyautogui")
        try:
            import pyautogui
            w, h = pyautogui.size()
            log.info(f"Screen via pyautogui: {w}x{h}")
            return {'x': 0, 'y': 0, 'width': w, 'height': h}
        except Exception as e2:
            log.error(f"pyautogui.size() also failed: {e2}")
            return {'x': 0, 'y': 0, 'width': 1920, 'height': 1080}


def get_all_screens() -> list:
    """
    Returns list of all monitors. Useful for future multi-monitor support.
    Each entry: {'x', 'y', 'width', 'height', 'name'}
    """
    try:
        from screeninfo import get_monitors
        return [
            {'x': m.x, 'y': m.y, 'width': m.width,
             'height': m.height, 'name': m.name or f"Monitor_{i}"}
            for i, m in enumerate(get_monitors())
        ]
    except Exception as e:
        log.warning(f"get_all_screens failed: {e}")
        primary = get_primary_screen()
        primary['name'] = 'Primary'
        return [primary]