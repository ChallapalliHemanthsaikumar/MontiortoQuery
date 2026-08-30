"""Emit structured behavioral events from the edge pipeline."""

import json
import os
from datetime import datetime, timezone


class EventEmitter:
    """Convert pipeline observations into structured events and queue them."""

    def __init__(self, device_id="antvision-pi01", experiment_id="exp001"):
        self.device_id = device_id
        self.experiment_id = experiment_id
        self.events = []

    def _base_event(self, event_type, frame=None):
        event = {
            "device_id": self.device_id,
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
        }
        if frame is not None:
            event["frame"] = frame
        return event

    def emit_zone_enter(self, ant_id, frame, x, y, speed=0.0):
        event = self._base_event("food_zone_enter", frame)
        event.update({"ant_id": ant_id, "x": x, "y": y,
                       "speed": speed, "zone": "food"})
        self.events.append(event)
        return event

    def emit_zone_exit(self, ant_id, frame, x, y):
        event = self._base_event("food_zone_exit", frame)
        event.update({"ant_id": ant_id, "x": x, "y": y, "zone": "outside"})
        self.events.append(event)
        return event

    def emit_metrics_snapshot(self, frame, ant_count, zone_occupancy, avg_speed):
        event = self._base_event("metrics_snapshot", frame)
        event["metrics"] = {
            "ant_count": ant_count,
            "zone_occupancy": zone_occupancy,
            "avg_speed": round(avg_speed, 2),
        }
        self.events.append(event)
        return event

    def emit_experiment_start(self):
        self.events.append(self._base_event("experiment_start"))

    def emit_experiment_end(self, frame):
        self.events.append(self._base_event("experiment_end", frame))

    def save_events(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2)
        return len(self.events)
