"""Ant detection using contour analysis."""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: float
    contour: np.ndarray


class AntDetector:
    """Detect ant-sized blobs from a binary mask."""

    def __init__(self, min_area=80, max_area=1500):
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append(Detection(
                    x=x, y=y, w=w, h=h, area=area, contour=contour
                ))
        return detections

    @staticmethod
    def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
        annotated = frame.copy()
        for det in detections:
            cv2.rectangle(annotated, (det.x, det.y),
                          (det.x + det.w, det.y + det.h), color, thickness)
            cx, cy = det.x + det.w // 2, det.y + det.h // 2
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(annotated, f"Count: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return annotated
