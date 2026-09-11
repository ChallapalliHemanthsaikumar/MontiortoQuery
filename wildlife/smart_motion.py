"""Wildlife-grade motion detection — filters out leaves, wind, and sun shifts."""

import cv2
import numpy as np


class WildlifeMotionDetector:
    """
    Uses MOG2 background subtraction with solidity filtering and
    multi-frame temporal confirmation to distinguish real wildlife
    from leaves, wind, and shadows.
    """

    def __init__(self, min_contour_area=3000, min_motion_pct=0.5,
                 max_motion_pct=40.0, cooldown_seconds=10, fps=0.2,
                 min_solidity=0.3, confirm_frames=2, match_distance=60):
        self.min_contour_area = min_contour_area
        self.min_motion_pct = min_motion_pct
        self.max_motion_pct = max_motion_pct
        self.cooldown_frames = max(1, int(cooldown_seconds * fps))
        self.frames_since_trigger = self.cooldown_frames
        self.min_solidity = min_solidity
        self.confirm_frames = confirm_frames
        self.match_distance = match_distance

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=True
        )
        self.frame_count = 0
        self.motion_history = []

    def _check_temporal_consistency(self, centroids):
        """Return centroids that have a nearby match in previous frames."""
        if len(self.motion_history) < self.confirm_frames - 1:
            return []

        confirmed = []
        for cx, cy in centroids:
            matches = 0
            for prev_centroids in self.motion_history[-(self.confirm_frames - 1):]:
                for px, py in prev_centroids:
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if dist < self.match_distance:
                        matches += 1
                        break
            if matches >= self.confirm_frames - 1:
                confirmed.append((cx, cy))
        return confirmed

    def detect(self, frame):
        self.frame_count += 1
        self.frames_since_trigger += 1

        small = cv2.resize(frame, (320, 240))
        mask = self.bg_subtractor.apply(small)

        # Remove shadows (MOG2 marks shadows as 127)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter: only keep large, compact contours (animals, not scattered leaves)
        significant = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if not (0.2 < aspect < 5.0):
                continue

            # Solidity: animals are compact blobs, leaves are jagged/hollow
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.min_solidity:
                    continue

            significant.append((area, (x, y, w, h)))

        total_pixels = 320 * 240
        motion_area = sum(a for a, _ in significant)
        motion_pct = (motion_area / total_pixels) * 100

        # Too much motion = wind shaking everything or sudden light change
        if motion_pct > self.max_motion_pct:
            self.motion_history.append([])
            return False, round(motion_pct, 2), []

        # Multi-frame temporal consistency check
        centroids = [(x + w // 2, y + h // 2) for _, (x, y, w, h) in significant]
        confirmed = self._check_temporal_consistency(centroids)

        self.motion_history.append(centroids)
        if len(self.motion_history) > self.confirm_frames + 2:
            self.motion_history.pop(0)

        if self.confirm_frames > 1 and len(significant) > 0:
            if len(confirmed) > 0:
                # Keep only temporally confirmed detections
                confirmed_set = set(confirmed)
                significant = [
                    (a, (x, y, w, h)) for a, (x, y, w, h) in significant
                    if (x + w // 2, y + h // 2) in confirmed_set
                ]
                motion_area = sum(a for a, _ in significant)
                motion_pct = (motion_area / total_pixels) * 100
            else:
                # Motion found but not confirmed across frames — likely leaves
                return False, round(motion_pct, 2), []

        triggered = (motion_pct >= self.min_motion_pct and
                     self.frames_since_trigger >= self.cooldown_frames and
                     len(significant) > 0)

        if triggered:
            self.frames_since_trigger = 0

        # Scale bounding boxes back to original resolution
        h_scale = frame.shape[0] / 240
        w_scale = frame.shape[1] / 320
        boxes = []
        for _, (x, y, w, h) in significant:
            boxes.append((
                int(x * w_scale), int(y * h_scale),
                int(w * w_scale), int(h * h_scale)
            ))

        return triggered, round(motion_pct, 2), boxes

    def warmup_needed(self):
        return self.frame_count < 10
