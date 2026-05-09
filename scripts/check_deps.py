# scripts/check_deps.py
# Run this to verify all dependencies are correctly installed.
# Usage: python scripts/check_deps.py

import sys
import importlib

REQUIRED = [
    ("cv2",          "opencv-python"),
    ("mediapipe",    "mediapipe"),
    ("numpy",        "numpy"),
    ("pyautogui",    "pyautogui"),
    ("pynput",       "pynput"),
    ("speech_recognition", "SpeechRecognition"),
    ("vosk",         "vosk"),
    ("PyQt6",        "PyQt6"),
    ("yaml",         "PyYAML"),
    ("filterpy",     "filterpy"),
    ("mss",          "mss"),
    ("screeninfo",   "screeninfo"),
    ("pytest",       "pytest"),
]

print(f"\nPython version: {sys.version}\n")
print("Checking dependencies...\n")

all_ok = True
for module_name, package_name in REQUIRED:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✓ {package_name:<25} ({version})")
    except ImportError:
        print(f"  ✗ {package_name:<25} NOT FOUND — run: pip install {package_name}")
        all_ok = False

print()
if all_ok:
    print("✓ All dependencies OK. You're ready to build.\n")
else:
    print("✗ Some dependencies are missing. Install them and run this script again.\n")
    sys.exit(1)