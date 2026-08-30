"""Trajectory recording and speed calculation for tracked ants."""

import numpy as np
from collections import defaultdict


class TrajectoryRecorder:
    """Record positions over time and compute movement metrics."""

    def __init__(self, fps=15.0):
        self.fps = fps
        self.paths = defaultdict(list)

    def update(self, tracked_objects, frame_num):
        for ant_id, centroid in tracked_objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            self.paths[ant_id].append((frame_num, cx, cy))

    def get_speed(self, ant_id):
        """Average speed in pixels per second."""
        path = self.paths.get(ant_id, [])
        if len(path) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(1, len(path)):
            dx = path[i][1] - path[i - 1][1]
            dy = path[i][2] - path[i - 1][2]
            total_dist += np.sqrt(dx * dx + dy * dy)

        total_frames = path[-1][0] - path[0][0]
        if total_frames == 0:
            return 0.0
        return total_dist / (total_frames / self.fps)

    def get_trajectory_length(self, ant_id):
        """Total distance traveled in pixels."""
        path = self.paths.get(ant_id, [])
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][1] - path[i - 1][1]
            dy = path[i][2] - path[i - 1][2]
            total += np.sqrt(dx * dx + dy * dy)
        return total

    def get_all_speeds(self):
        return {ant_id: self.get_speed(ant_id) for ant_id in self.paths}

    def draw_trails(self, frame, trail_length=30, color=(255, 255, 0)):
        """Draw recent movement trails on frame."""
        for ant_id, path in self.paths.items():
            recent = path[-trail_length:]
            if len(recent) < 2:
                continue
            points = [(p[1], p[2]) for p in recent]
            for i in range(1, len(points)):
                alpha = int(255 * i / len(points))
                cv2.line(frame, points[i - 1], points[i],
                         (color[0], color[1], min(alpha, 255)), 1)
        return frame


import cv2
