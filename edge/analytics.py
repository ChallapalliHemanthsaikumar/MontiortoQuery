"""Behavioral analytics — aggregate metrics from detection, tracking, and zone events."""

import json
from dataclasses import dataclass, asdict
from edge.detection.food_zone import ZoneEvent


@dataclass
class ExperimentMetrics:
    total_frames: int = 0
    fps: float = 15.0
    total_unique_ants: int = 0
    first_discovery_frame: int = -1
    first_discovery_time_sec: float = -1.0
    total_zone_entries: int = 0
    total_zone_exits: int = 0
    peak_zone_occupancy: int = 0
    avg_speed_px_per_sec: float = 0.0
    avg_trajectory_length_px: float = 0.0

    def to_dict(self):
        return asdict(self)

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)


class BehaviorAnalyzer:
    """Compute experiment-level behavioral metrics."""

    def __init__(self, fps=15.0):
        self.fps = fps
        self.zone_transitions = []
        self.peak_occupancy = 0
        self.occupancy_over_time = []
        self.ant_counts_over_time = []

    def record_transitions(self, transitions):
        self.zone_transitions.extend(transitions)

    def record_frame(self, num_detected, num_in_zone):
        self.ant_counts_over_time.append(num_detected)
        self.occupancy_over_time.append(num_in_zone)
        if num_in_zone > self.peak_occupancy:
            self.peak_occupancy = num_in_zone

    def compute_metrics(self, tracker, trajectory_recorder):
        metrics = ExperimentMetrics()
        metrics.fps = self.fps
        metrics.total_unique_ants = tracker.next_id
        metrics.total_frames = len(self.ant_counts_over_time)
        metrics.peak_zone_occupancy = self.peak_occupancy

        entries = [t for t in self.zone_transitions if t.event == ZoneEvent.ENTER]
        exits = [t for t in self.zone_transitions if t.event == ZoneEvent.EXIT]
        metrics.total_zone_entries = len(entries)
        metrics.total_zone_exits = len(exits)

        if entries:
            first = min(entries, key=lambda t: t.frame)
            metrics.first_discovery_frame = first.frame
            metrics.first_discovery_time_sec = round(first.frame / self.fps, 2)

        speeds = trajectory_recorder.get_all_speeds()
        if speeds:
            metrics.avg_speed_px_per_sec = round(
                sum(speeds.values()) / len(speeds), 2
            )

        lengths = [trajectory_recorder.get_trajectory_length(aid)
                    for aid in trajectory_recorder.paths]
        if lengths:
            metrics.avg_trajectory_length_px = round(
                sum(lengths) / len(lengths), 2
            )

        return metrics
