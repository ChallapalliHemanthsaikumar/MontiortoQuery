"""Log every capture event to CSV and JSON for later analysis."""

import csv
import json
import os
from datetime import datetime, timezone


class DataLogger:
    """Append-only event log — survives crashes, easy to load in pandas."""

    def __init__(self, data_dir, experiment_id):
        self.data_dir = data_dir
        self.experiment_id = experiment_id
        os.makedirs(data_dir, exist_ok=True)

        self.csv_path = os.path.join(data_dir, f"{experiment_id}_events.csv")
        self.json_path = os.path.join(data_dir, f"{experiment_id}_events.json")
        self.summary_path = os.path.join(data_dir, f"{experiment_id}_summary.json")

        self.events = []
        self.stats = {
            "experiment_id": experiment_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "total_frames": 0,
            "motion_triggers": 0,
            "heartbeat_captures": 0,
            "dark_skips": 0,
            "storage_skips": 0,
            "images_saved": 0,
        }

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "frame_num", "trigger", "brightness",
                    "motion_pct", "num_detections", "boxes", "image_path",
                    "free_gb"
                ])

    def log_event(self, frame_num, trigger, brightness, motion_pct,
                  num_detections, boxes, image_path, free_gb):
        ts = datetime.now(timezone.utc).isoformat()
        event = {
            "timestamp": ts,
            "frame_num": frame_num,
            "trigger": trigger,
            "brightness": brightness,
            "motion_pct": motion_pct,
            "num_detections": num_detections,
            "boxes": boxes,
            "image_path": image_path,
            "free_gb": free_gb,
        }
        self.events.append(event)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts, frame_num, trigger, brightness, motion_pct,
                num_detections, json.dumps(boxes), image_path, free_gb
            ])

        self.stats["images_saved"] += 1

    def log_skip(self, reason):
        if reason == "dark":
            self.stats["dark_skips"] += 1
        elif reason == "storage":
            self.stats["storage_skips"] += 1

    def log_frame(self):
        self.stats["total_frames"] += 1

    def log_motion(self):
        self.stats["motion_triggers"] += 1

    def log_heartbeat(self):
        self.stats["heartbeat_captures"] += 1

    def save_summary(self):
        self.stats["end_time"] = datetime.now(timezone.utc).isoformat()
        self.stats["images_saved"] = len(self.events)
        with open(self.summary_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        with open(self.json_path, "w") as f:
            json.dump(self.events, f, indent=2)
        return self.summary_path
