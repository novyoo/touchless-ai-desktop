# WebcamCapture - opens the camera and provides a clean interface to get frames.

# WHY a separate class?
# The rest of the system doesn't care HOW we get frames (webcam, file, IP stream).
# It just wants frames. Separation of concerns: if we switch to an IP camera,
# only this file changes.


import cv2          # OpenCV: the standard library for working with cameras and images
import logging      # Python's built-in logging system

log = logging.getLogger(__name__)
# __name__ is a special Python variable that equals this file's module path
# e.g. "src.vision.capture" — so log messages are labelled clearly


class WebcamCapture:
    """
    Manages the connection to a webcam and provides frames on demand.
    
    Usage:
        cap = WebcamCapture(device_id=0, width=1280, height=720)
        cap.start()
        frame = cap.read()   # Returns a numpy array (the image)
        cap.release()
    """

    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Args:
            device_id: Which camera to use. 0 = default/built-in. 1, 2 = external USB cameras.
            width:     Requested capture width in pixels. Camera may not honor this exactly.
            height:    Requested capture height in pixels.
            fps:       Requested frames per second. Camera may cap this.
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        
        # _cap will hold the cv2.VideoCapture object after start() is called.
        # We use None to mean "not yet started" — helps with error checking.
        self._cap = None
        
        # Track whether the camera was successfully opened
        self.is_running = False

    def start(self) -> bool:
        """
        Opens the camera device and configures it.
        
        Returns:
            True if the camera opened successfully, False otherwise.
        """
        log.info(f"Opening camera device {self.device_id} ({self.width}x{self.height} @ {self.fps}fps)")
        
        # cv2.VideoCapture(0) opens your default webcam.
        # The number corresponds to /dev/video0 on Linux, or device index on Windows/Mac.
        self._cap = cv2.VideoCapture(self.device_id)
        
        if not self._cap.isOpened():
            # This happens if the camera is in use by another app, or doesn't exist
            log.error(f"Failed to open camera device {self.device_id}")
            return False
        
        # CAP_PROP_FRAME_WIDTH / HEIGHT / FPS are OpenCV constants that tell the
        # driver what settings we want. The camera may not support these exactly
        # it will use the closest supported resolution.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Read back what the camera actually set (it may differ from what we asked)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        log.info(f"Camera opened: actual resolution {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
        
        self.is_running = True
        return True

    def read(self):
        """
        Reads the next frame from the camera.
        
        Returns:
            numpy.ndarray: The frame as a BGR image array (height x width x 3 channels).
            Returns None if the frame couldn't be read.
                          
        WHY BGR and not RGB?
            OpenCV historically stores images in Blue-Green-Red order instead of
            the more common Red-Green-Blue order. This is a legacy quirk.
            MediaPipe expects RGB, so we'll convert in the next step.
        """
        if not self.is_running or self._cap is None:
            log.warning("Tried to read from camera before calling start()")
            return None
        
        # cap.read() returns two values:
        # ret (bool): True if the frame was read successfully
        # frame (numpy array): The actual image data, or None if ret is False
        ret, frame = self._cap.read()
        
        if not ret:
            log.warning("Failed to read frame from camera (camera may have disconnected)")
            return None
        
        return frame

    def release(self):
        """
        Cleanly closes the camera and releases system resources.
        
        IMPORTANT: Always call this when you're done! If you don't, the camera
        stays locked and other applications can't use it until you restart Python.
        """
        if self._cap is not None:
            self._cap.release()
            log.info("Camera released")
        self.is_running = False

    def __enter__(self):
        """Allows using this class as a context manager: with WebcamCapture() as cap:"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called automatically when exiting a 'with' block — ensures release() is called."""
        self.release()