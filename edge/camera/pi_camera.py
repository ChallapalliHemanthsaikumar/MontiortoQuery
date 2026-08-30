"""Live capture from Raspberry Pi Camera using picamera2."""

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False


class PiCamera:
    """Capture frames from the Raspberry Pi Camera Module."""

    def __init__(self, resolution=(640, 480), fps=15):
        if not HAS_PICAMERA:
            raise RuntimeError("picamera2 not available — are you on a Raspberry Pi?")
        self.resolution = resolution
        self.fps = fps
        self.camera = None

    def open(self):
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
        return self

    def read(self):
        if self.camera is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        return self.camera.capture_array()

    def release(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()
