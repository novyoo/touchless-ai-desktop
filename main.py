# Entry point for the Touchless AI Desktop Control System.
# Why keep this file minimal?
# Because main.py is the "door" to the application. Complex logic in the door
# makes testing harder and architecture messy. All real logic lives in src/.

import sys
import logging

# Set up basic logging FIRST, before importing anything else
# This ensures that even import errors get logged properly
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),     # Print to terminal
        logging.FileHandler("touchless.log"),  # Also save to a file
    ]
)

log = logging.getLogger("main")
log.info("Touchless AI Desktop starting...")

def main():
    """
    Main entry function.
    
    We wrap everything in a function (instead of running at module level)
    because PyInstaller handles packaged entry points better this way.
    It also makes the application testable — tests can import main.py
    without accidentally starting the app.
    """
    # Phase 0: Just verify the environment works
    # We'll replace this with the real startup sequence in Phase 1
    log.info("Phase 0 — environment check")
    
    try:
        import cv2
        log.info(f"OpenCV version: {cv2.__version__}")
    except ImportError as e:
        log.error(f"OpenCV not found: {e}")
        sys.exit(1)
    
    try:
        import mediapipe as mp
        log.info(f"MediaPipe version: {mp.__version__}")
    except ImportError as e:
        log.error(f"MediaPipe not found: {e}")
        sys.exit(1)
    
    try:
        from PyQt6.QtWidgets import QApplication
        log.info("PyQt6 loaded successfully")
    except ImportError as e:
        log.error(f"PyQt6 not found: {e}")
        sys.exit(1)
    
    log.info("All Phase 0 checks passed. Environment is healthy.")
    print("\n✓ Environment verified. Ready for Phase 1.\n")


# Why this pattern?
# if __name__ == "__main__": means "only run main() if this file was run
# directly (e.g. python main.py)". If another file imports main.py, this
# block is skipped. This is essential for safe testing.

if __name__ == "__main__":
    main()