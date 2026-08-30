"""Food zone definition and entry/exit detection."""

import cv2
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
