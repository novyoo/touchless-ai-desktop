# High-level system actions triggered by voice commands or gesture macros.
# Volume, screenshot, app launch, etc.
# All actions go through a WHITELIST nothing arbitrary can execute.

import subprocess
import logging
import sys
import os
import time

log = logging.getLogger(__name__)


class SystemActions:
    """
    Executes safe, whitelisted system actions.

    WHY a whitelist?
    Voice commands and gesture macros could theoretically execute
    anything. The whitelist ensures only pre-approved, safe actions
    can be triggered — no arbitrary shell commands.
    """

    def __init__(self):
        self._last_screenshot: float = 0.0
        self._screenshot_cooldown    = 2.0  # seconds

    def take_screenshot(self) -> bool:
        # Save a screenshot to the user's desktop. 
        now = time.perf_counter()
        if now - self._last_screenshot < self._screenshot_cooldown:
            log.debug("Screenshot cooldown active")
            return False
        try:
            import pyautogui
            from datetime import datetime
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            desktop    = os.path.join(os.path.expanduser("~"), "Desktop")
            os.makedirs(desktop, exist_ok=True)
            filepath   = os.path.join(desktop, f"touchless_{timestamp}.png")
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            self._last_screenshot = now
            log.info(f"Screenshot saved: {filepath}")
            return True
        except Exception as e:
            log.error(f"Screenshot failed: {e}")
            return False

    def volume_up(self):
        # Increase system volume by one step.
        try:
            if sys.platform == "win32":
                import pyautogui
                pyautogui.press('volumeup')
            elif sys.platform == "darwin":
                subprocess.run(
                    ["osascript", "-e",
                     "set volume output volume (output volume of (get volume settings) + 10)"],
                    check=True
                )
            else:  # Linux
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
                               check=True)
            log.debug("Volume up")
        except Exception as e:
            log.warning(f"Volume up failed: {e}")

    def volume_down(self):
        # Decrease system volume by one step.
        try:
            if sys.platform == "win32":
                import pyautogui
                pyautogui.press('volumedown')
            elif sys.platform == "darwin":
                subprocess.run(
                    ["osascript", "-e",
                     "set volume output volume (output volume of (get volume settings) - 10)"],
                    check=True
                )
            else:
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"],
                               check=True)
            log.debug("Volume down")
        except Exception as e:
            log.warning(f"Volume down failed: {e}")

    def mute(self):
        # Toggle system mute
        try:
            if sys.platform == "win32":
                import pyautogui
                pyautogui.press('volumemute')
            elif sys.platform == "darwin":
                subprocess.run(
                    ["osascript", "-e",
                     "set volume output muted not (output muted of (get volume settings))"],
                    check=True
                )
            else:
                subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                               check=True)
            log.debug("Mute toggled")
        except Exception as e:
            log.warning(f"Mute failed: {e}")

    def launch_app(self, app_name: str) -> bool:
        """
        Launch a whitelisted application by name.
        app_name must be in SAFE_APPS — arbitrary names are rejected.
        """
        SAFE_APPS = {
            "notepad":    {"win32": "notepad.exe",       "darwin": "open -a TextEdit", "linux": "gedit"},
            "browser":    {"win32": "start chrome",      "darwin": "open -a 'Google Chrome'", "linux": "xdg-open https://"},
            "explorer":   {"win32": "explorer",          "darwin": "open ~", "linux": "nautilus ~"},
            "calculator": {"win32": "calc.exe",          "darwin": "open -a Calculator", "linux": "gnome-calculator"},
            "terminal":   {"win32": "cmd.exe",           "darwin": "open -a Terminal", "linux": "gnome-terminal"},
            "vscode":     {"win32": "code",              "darwin": "open -a 'Visual Studio Code'", "linux": "code"},
        }

        key = app_name.lower().strip()
        if key not in SAFE_APPS:
            log.warning(f"App '{app_name}' not in whitelist — rejected")
            return False

        platform = sys.platform
        platform_key = "darwin" if platform == "darwin" else \
                       "linux"  if platform.startswith("linux") else "win32"

        cmd = SAFE_APPS[key].get(platform_key)
        if not cmd:
            log.warning(f"No command for app '{key}' on {platform_key}")
            return False

        try:
            if platform_key == "win32":
                subprocess.Popen(cmd, shell=True)
            else:
                subprocess.Popen(cmd.split())
            log.info(f"Launched: {cmd}")
            return True
        except Exception as e:
            log.error(f"Launch failed: {e}")
            return False