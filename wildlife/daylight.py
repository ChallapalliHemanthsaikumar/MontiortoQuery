"""Detect whether there's enough daylight for useful captures."""

import cv2
import numpy as np
from datetime import datetime


class DaylightDetector:
    """Skip captures when it's too dark for the Pi Camera (no IR)."""

    def __init__(self, brightness_threshold=30, dark_streak_limit=5):
        self.brightness_threshold = brightness_threshold
        self.dark_streak_limit = dark_streak_limit
        self.dark_streak = 0

    def is_daylight(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < self.brightness_threshold:
            self.dark_streak += 1
        else:
            self.dark_streak = 0
        return self.dark_streak < self.dark_streak_limit

    def get_brightness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return round(float(np.mean(gray)), 1)
