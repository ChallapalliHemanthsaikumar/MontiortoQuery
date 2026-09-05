"""Monitor disk space and enforce capture budgets."""

import os
import shutil
import glob


class StorageManager:
    """Prevent filling up the SD card with captures."""

    def __init__(self, data_dir, min_free_gb=5.0, max_captures_per_hour=120,
                 max_local_images=2000):
        self.data_dir = data_dir
        self.min_free_bytes = int(min_free_gb * 1024 * 1024 * 1024)
        self.max_captures_per_hour = max_captures_per_hour
        self.max_local_images = max_local_images
        self.captures_this_hour = 0
        self.current_hour = -1

    def can_capture(self):
        hour = int(os.popen("date +%H").read().strip() or "0")
        if hour != self.current_hour:
            self.current_hour = hour
            self.captures_this_hour = 0

        if self.captures_this_hour >= self.max_captures_per_hour:
            return False, "hourly capture limit reached"

        free = shutil.disk_usage(self.data_dir).free
        if free < self.min_free_bytes:
            return False, f"disk low: {free / (1024**3):.1f}GB free"

        return True, "ok"

    def record_capture(self):
        self.captures_this_hour += 1

    def cleanup_old_frames(self, keep_recent=500):
        """Delete oldest local frames if we exceed max_local_images."""
        pattern = os.path.join(self.data_dir, "**", "*.jpg")
        files = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
        if len(files) > self.max_local_images:
            to_delete = files[:len(files) - keep_recent]
            for f in to_delete:
                os.remove(f)
            return len(to_delete)
        return 0

    def get_stats(self):
        free = shutil.disk_usage(self.data_dir).free
        pattern = os.path.join(self.data_dir, "**", "*.jpg")
        count = len(glob.glob(pattern, recursive=True))
        return {
            "free_gb": round(free / (1024**3), 1),
            "local_images": count,
            "captures_this_hour": self.captures_this_hour,
        }
