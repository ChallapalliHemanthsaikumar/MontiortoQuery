"""Motion detection using frame differencing."""

import cv2
import numpy as np


class MotionDetector:
    """Detect significant motion between consecutive frames."""

    def __init__(self, threshold=25, min_area=500, cooldown_frames=15):
        self.threshold = threshold
        self.min_area = min_area
        self.cooldown_frames = cooldown_frames
        self.prev_gray = None
        self.frames_since_trigger = 0

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        delta = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > self.min_area)

        self.prev_gray = gray
        self.frames_since_trigger += 1

        motion_detected = motion_area > 0 and self.frames_since_trigger >= self.cooldown_frames
        if motion_detected:
            self.frames_since_trigger = 0

        motion_pct = motion_area / (frame.shape[0] * frame.shape[1]) * 100
        return motion_detected, round(motion_pct, 2)
