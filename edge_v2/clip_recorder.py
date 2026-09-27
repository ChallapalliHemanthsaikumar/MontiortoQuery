"""Dynamic clip recorder — records as long as the trigger is active.

OLD approach: fixed 15-second clips. A 3-min conversation = 12 clips.
NEW approach: start when trigger fires, stop when trigger ends + post-roll.
    One event = one clip, however long it takes.

How it works:
    1. Event fires → start FFmpeg subprocess (piped stdin)
    2. Drain ring buffer for pre-roll (what happened BEFORE trigger)
    3. Main loop keeps calling feed_frame() / feed_audio()
    4. Trigger stops → start post-roll countdown
    5. Post-roll expires → send SIGINT to FFmpeg → clean MP4

Why SIGINT and not kill?
    MP4 files have a "moov atom" — an index at the end that tells players
    where each frame lives. If you kill -9 FFmpeg, it never writes this
    index and the file is unplayable. SIGINT lets FFmpeg finalize cleanly.

Two recording modes:
    STREAMING: pipe raw frames/audio to FFmpeg in real-time (for live camera)
    BATCH: collect all frames, encode at the end (for post-processing)

We use BATCH for now because:
    - Simpler (no pipe management, no SIGINT timing issues on Windows)
    - Works on both Pi and Windows
    - The main loop collects frames in a list, then background-encodes

The streaming approach is better for very long events (10+ minutes)
where holding all frames in memory is too expensive. We can upgrade
later if needed.
"""

import os
import time
import subprocess
import threading
import cv2
import numpy as np
from datetime import datetime


class DynamicClipRecorder:
    """Records variable-length A/V clips based on trigger duration.

    State machine:
        IDLE → trigger fires → RECORDING (collecting frames)
        RECORDING → trigger active → keep collecting
        RECORDING → trigger stops → POST_ROLL (countdown)
        POST_ROLL → countdown expires → ENCODING (background FFmpeg)
        ENCODING → done → IDLE

    Usage:
        recorder = DynamicClipRecorder(output_dir="data/clips")

        # When trigger fires:
        recorder.start_recording(pre_roll_frames, pre_roll_audio, event_type)

        # Every frame while trigger is active:
        recorder.feed_frame(frame)
        recorder.feed_audio(pcm_chunk)

        # When trigger stops:
        recorder.trigger_ended()

        # Keep calling feed_frame even during post-roll!
        # The recorder handles the countdown internally.
    """

    def __init__(self, output_dir="data/clips", ffmpeg_path="ffmpeg",
                 post_roll_seconds=5.0, fps=15, sample_rate=48000,
                 max_clip_seconds=300):
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path
        self.post_roll_seconds = post_roll_seconds
        self.fps = fps
        self.sample_rate = sample_rate
        self.max_clip_seconds = max_clip_seconds

        self._state = "idle"
        self._video_frames = []
        self._audio_chunks = []
        self._event_type = ""
        self._start_time = 0.0
        self._trigger_end_time = 0.0
        self._current_clip_path = None
        self._best_frame = None
        self._best_frame_score = -1.0
        self._yolo_detections = []
        self._encode_thread = None

        os.makedirs(output_dir, exist_ok=True)

    @property
    def is_recording(self):
        return self._state in ("recording", "post_roll")

    @property
    def state(self):
        return self._state

    @property
    def duration(self):
        if not self._video_frames:
            return 0.0
        return len(self._video_frames) / max(self.fps, 1)

    def start_recording(self, pre_roll_video, pre_roll_audio,
                        event_type="event"):
        """Begin a new dynamic clip.

        Args:
            pre_roll_video: list of (ts, frame) from video ring buffer drain
            pre_roll_audio: list of (ts, pcm) from audio ring buffer drain
            event_type: tag for the filename
        """
        if self._state != "idle":
            return

        self._state = "recording"
        self._event_type = event_type
        self._start_time = time.time()
        self._trigger_end_time = 0.0
        self._best_frame = None
        self._best_frame_score = -1.0
        self._yolo_detections = []

        self._video_frames = [(ts, frame.copy()) for ts, frame in pre_roll_video]
        self._audio_chunks = list(pre_roll_audio)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"event_{timestamp}_{event_type}.mp4"
        self._current_clip_path = os.path.join(self.output_dir, filename)

    def feed_frame(self, frame, yolo_detections=None):
        """Add a frame to the current recording.

        Call this every frame while recording (including post-roll).
        Also tracks the best frame for VLM analysis.
        """
        if not self.is_recording:
            return

        self._video_frames.append((time.time(), frame.copy()))

        self._track_best_frame(frame, yolo_detections)
        if yolo_detections:
            self._yolo_detections = yolo_detections

        if self.duration >= self.max_clip_seconds:
            self._finalize()

        if self._state == "post_roll":
            elapsed_since_trigger_end = time.time() - self._trigger_end_time
            if elapsed_since_trigger_end >= self.post_roll_seconds:
                self._finalize()

    def feed_audio(self, pcm_chunk):
        """Add an audio chunk to the current recording."""
        if not self.is_recording:
            return
        self._audio_chunks.append((time.time(), pcm_chunk))

    def trigger_ended(self):
        """Signal that the trigger condition is no longer active.

        Starts the post-roll countdown. The recorder keeps collecting
        frames for post_roll_seconds more, then finalizes.
        """
        if self._state == "recording":
            self._state = "post_roll"
            self._trigger_end_time = time.time()

    def trigger_renewed(self):
        """The trigger re-activated during post-roll. Cancel the countdown.

        This handles the case where someone walks out of frame for 2 seconds
        then walks back. Instead of creating two clips, we extend the first.
        """
        if self._state == "post_roll":
            self._state = "recording"
            self._trigger_end_time = 0.0

    def force_stop(self):
        """Force-finalize the current clip (e.g., on shutdown)."""
        if self.is_recording:
            self._finalize()

    def wait(self, timeout=30):
        """Wait for background encoding to finish. Call before exit."""
        if self._encode_thread and self._encode_thread.is_alive():
            print("  [clip] Waiting for FFmpeg to finish encoding...")
            self._encode_thread.join(timeout=timeout)

    def _track_best_frame(self, frame, yolo_detections):
        """Track the sharpest or highest-confidence frame for VLM."""
        score = 0.0
        if yolo_detections:
            score = max(d["confidence"] for d in yolo_detections) * 1000
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if score > self._best_frame_score:
            self._best_frame_score = score
            self._best_frame = frame.copy()

    def _finalize(self):
        """Stop recording and encode the clip in a background thread."""
        self._state = "encoding"

        video_frames = self._video_frames
        audio_chunks = self._audio_chunks
        output_path = self._current_clip_path
        fps = self.fps
        sample_rate = self.sample_rate

        self._video_frames = []
        self._audio_chunks = []

        self._encode_thread = threading.Thread(
            target=self._encode,
            args=(video_frames, audio_chunks, output_path, fps, sample_rate),
        )
        self._encode_thread.start()

        self._state = "idle"

    def get_best_frame(self):
        """Return the best frame from the last/current recording."""
        return self._best_frame

    def get_clip_path(self):
        return self._current_clip_path

    def get_audio_pcm(self):
        """Return concatenated audio PCM from the current/last recording."""
        if self._audio_chunks:
            chunks = [c for _, c in self._audio_chunks]
            if chunks:
                return np.concatenate(chunks)
        return np.array([], dtype=np.float32)

    def _encode(self, video_frames, audio_chunks, output_path,
                fps, sample_rate):
        """Background-encode collected frames into MP4 via FFmpeg."""
        if not video_frames:
            return

        first_frame = video_frames[0][1]
        height, width = first_frame.shape[:2]
        has_audio = len(audio_chunks) > 0

        video_duration = len(video_frames) / fps
        print(f"  [clip] Encoding: {len(video_frames)} frames ({video_duration:.1f}s), "
              f"{len(audio_chunks)} audio chunks, has_audio={has_audio}")

        if has_audio:
            total_samples = sum(len(chunk) for _, chunk in audio_chunks)
            audio_duration = total_samples / sample_rate
            print(f"  [clip] Audio: {total_samples} samples = {audio_duration:.1f}s "
                  f"@ {sample_rate}Hz")

        video_tmp = output_path + ".raw_video"
        audio_tmp = output_path + ".raw_audio"

        try:
            with open(video_tmp, "wb") as f:
                for ts, frame in video_frames:
                    f.write(frame.tobytes())

            if has_audio:
                pcm = np.concatenate([chunk for _, chunk in audio_chunks])
                peak = np.max(np.abs(pcm))
                if peak > 0:
                    target_peak = 0.9
                    gain = min(target_peak / peak, 30.0)
                    pcm = pcm * gain
                    print(f"  [clip] Audio normalized: peak {peak:.4f} → "
                          f"{peak*gain:.2f} (gain {gain:.1f}x)")
                pcm_16 = (pcm * 32767).astype(np.int16)
                with open(audio_tmp, "wb") as f:
                    f.write(pcm_16.tobytes())
                audio_file_mb = os.path.getsize(audio_tmp) / (1024 * 1024)
                print(f"  [clip] Raw audio file: {audio_file_mb:.2f}MB")

            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", str(fps),
                "-i", video_tmp,
            ]

            if has_audio:
                cmd.extend([
                    "-f", "s16le",
                    "-ar", str(sample_rate),
                    "-ac", "1",
                    "-i", audio_tmp,
                ])

            cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "fast", "-crf", "23"])

            if has_audio:
                cmd.extend(["-c:a", "aac", "-b:a", "128k"])

            cmd.extend(["-movflags", "+faststart", output_path])

            print(f"  [clip] FFmpeg cmd: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, timeout=300)

            stderr = result.stderr.decode("utf-8", errors="replace")
            if result.returncode == 0:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  [clip] {os.path.basename(output_path)} "
                      f"({video_duration:.1f}s, {size_mb:.1f}MB)")
                for line in stderr.split("\n"):
                    if "Stream" in line or "Audio:" in line or "Video:" in line:
                        print(f"  [clip] {line.strip()}")
            else:
                print(f"  [clip] FFmpeg error: {stderr[:500]}")

        except subprocess.TimeoutExpired:
            print(f"  [clip] FFmpeg timed out")
        except Exception as e:
            print(f"  [clip] Encoding failed: {e}")
        finally:
            for tmp in [video_tmp, audio_tmp]:
                if os.path.exists(tmp):
                    os.remove(tmp)


class RollingStorage:
    """Rolling disk storage — keeps clips within a size budget.

    When total clip storage exceeds max_gb, the oldest clips are deleted
    automatically. This creates a 24-hour rolling window on a 3GB budget.

    Why 3GB?
        - Pi 4 has 64GB SD card, 3GB is ~5% of the card
        - H.264 at 640x480 CRF 23 ≈ 500Kbps-1.5Mbps
        - 3GB holds ~4-6 hours of continuous event footage
        - On a typical day with 1-2 hours of events: easily 24+ hours

    The storage manager runs a cleanup check after every new clip.
    It sorts by modification time and deletes oldest-first until
    total size is under the budget.
    """

    def __init__(self, clip_dir="data/clips", max_gb=3.0,
                 min_free_sd_gb=2.0):
        self.clip_dir = clip_dir
        self.max_bytes = int(max_gb * 1024 * 1024 * 1024)
        self.min_free_bytes = int(min_free_sd_gb * 1024 * 1024 * 1024)
        os.makedirs(clip_dir, exist_ok=True)

    def cleanup(self):
        """Delete oldest clips until total size is within budget.

        Returns number of clips deleted.
        """
        clips = self._get_clips_sorted_oldest_first()
        total_size = sum(size for _, size in clips)
        deleted = 0

        while clips and (total_size > self.max_bytes or
                         not self._has_free_space()):
            path, size = clips.pop(0)
            try:
                os.remove(path)
                total_size -= size
                deleted += 1
            except OSError:
                pass

        return deleted

    def get_stats(self):
        """Return storage statistics."""
        clips = self._get_clips_sorted_oldest_first()
        total_size = sum(size for _, size in clips)
        clip_count = len(clips)

        oldest_age_hours = 0.0
        if clips:
            oldest_mtime = os.path.getmtime(clips[0][0])
            oldest_age_hours = (time.time() - oldest_mtime) / 3600

        free_gb = self._get_free_space_gb()

        return {
            "clip_count": clip_count,
            "total_size_mb": round(total_size / (1024 * 1024), 1),
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
            "budget_gb": round(self.max_bytes / (1024 * 1024 * 1024), 1),
            "oldest_hours": round(oldest_age_hours, 1),
            "free_disk_gb": round(free_gb, 1),
        }

    def can_record(self):
        """Check if there's room to record another clip."""
        return self._has_free_space()

    def _get_clips_sorted_oldest_first(self):
        """List all MP4 clips sorted by modification time (oldest first)."""
        clips = []
        for f in os.listdir(self.clip_dir):
            if f.endswith(".mp4"):
                path = os.path.join(self.clip_dir, f)
                size = os.path.getsize(path)
                clips.append((path, size))
        clips.sort(key=lambda x: os.path.getmtime(x[0]))
        return clips

    def _has_free_space(self):
        """Check if the SD card has enough free space."""
        try:
            import shutil
            usage = shutil.disk_usage(self.clip_dir)
            return usage.free >= self.min_free_bytes
        except Exception:
            return True

    def _get_free_space_gb(self):
        try:
            import shutil
            usage = shutil.disk_usage(self.clip_dir)
            return usage.free / (1024 * 1024 * 1024)
        except Exception:
            return -1.0
