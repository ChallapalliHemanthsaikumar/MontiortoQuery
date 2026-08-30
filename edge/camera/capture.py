"""Abstracted frame capture — file-based on Windows, Pi Camera on Raspberry Pi."""

import cv2


class VideoCapture:
    """Capture frames from a video file or camera index."""

    def __init__(self, source):
        """
        Args:
            source: path to video file, or camera index (0 for default camera).
        """
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        return self

    def read(self):
        if self.cap is None:
            raise RuntimeError("Capture not opened. Call open() first.")
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()

    @property
    def fps(self):
        if self.cap is None:
            return 0
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self):
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()
