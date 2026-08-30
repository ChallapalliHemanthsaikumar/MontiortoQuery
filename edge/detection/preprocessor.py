"""Image preprocessing for ant detection."""

import cv2
import numpy as np


class Preprocessor:
    """Convert raw frames into clean binary masks for contour detection."""

    def __init__(self, blur_kernel=7, block_size=25, constant=12):
        self.blur_kernel = blur_kernel
        self.block_size = block_size
        self.constant = constant

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.block_size, self.constant
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned
