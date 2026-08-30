"""Food zone definition and entry/exit detection."""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ZoneEvent(Enum):
    ENTER = "food_zone_enter"
    EXIT = "food_zone_exit"


@dataclass
class ZoneTransition:
    ant_id: int
    event: ZoneEvent
    frame: int
    x: int
    y: int


def detect_food(frame, padding=40):
    """Auto-detect the food source as a small bright/white blob in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)

    # White food (sugar, etc.): very high brightness + low saturation
    white_mask = cv2.inRange(hsv, (0, 0, 220), (180, 50, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30 or area > 3000:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.3:
            continue
        score = circularity
        if score > best_score:
            best = c
            best_score = score

    if best is None:
        return None

    x, y, w, h = cv2.boundingRect(best)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + padding * 2
    h = h + padding * 2
    return FoodZone(x, y, w, h)


class FoodZone:
    """Rectangular food zone that detects ant entry and exit."""

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.inside = set()

    def contains(self, cx, cy):
        return self.x <= cx <= self.x + self.w and self.y <= cy <= self.y + self.h

    def update(self, tracked_objects, frame_num):
        transitions = []
        current_inside = set()

        for ant_id, centroid in tracked_objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            if self.contains(cx, cy):
                current_inside.add(ant_id)
                if ant_id not in self.inside:
                    transitions.append(ZoneTransition(
                        ant_id=ant_id, event=ZoneEvent.ENTER,
                        frame=frame_num, x=cx, y=cy
                    ))

        for ant_id in self.inside - current_inside:
            transitions.append(ZoneTransition(
                ant_id=ant_id, event=ZoneEvent.EXIT,
                frame=frame_num, x=0, y=0
            ))

        self.inside = current_inside
        return transitions

    def draw(self, frame, color=(0, 200, 255), thickness=2):
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), color, thickness)
        cv2.putText(frame, "FOOD", (self.x + 5, self.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        count = len(self.inside)
        cv2.putText(frame, f"In zone: {count}", (self.x + 5, self.y + self.h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
