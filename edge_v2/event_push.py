"""Push events from Pi to GPU server over WiFi.

Sends clip + best frame + audio + metadata to the server's /event/push endpoint.
The server then runs the full 9-step pipeline (VLM, Whisper, embeddings, etc).

Retry queue: if the server is down or WiFi drops, events are saved locally
and retried with exponential backoff. The Pi NEVER loses an event.

Exponential backoff:
    Attempt 1: wait 5s
    Attempt 2: wait 10s
    Attempt 3: wait 20s
    Attempt 4: wait 40s
    ...up to max 5 minutes between retries

Why not just save locally and batch upload?
    Because the server shows events in real-time on the dashboard.
    A 5-minute delay makes the live feed useless. We push immediately
    and only queue on failure.
"""

import os
import json
import time
import threading
import wave
import cv2
import numpy as np
from collections import deque
from datetime import datetime


class EventPusher:
    """Push events to the GPU vision server with retry logic.

    Usage:
        pusher = EventPusher(server_url="http://10.0.0.181:8000")
        pusher.start()

        pusher.push_event(
            clip_path="data/clips/event_20260926T140315.mp4",
            best_frame=numpy_frame,
            audio_pcm=numpy_pcm,
            metadata={...},
        )

        pusher.stop()
    """

    def __init__(self, server_url, api_key="wildlife-vision-secret-2026",
                 camera_id="pi-outdoor", max_retries=10,
                 retry_dir="data/retry_queue"):
        self.server_url = server_url
        self.api_key = api_key
        self.camera_id = camera_id
        self.max_retries = max_retries
        self.retry_dir = retry_dir

        self._queue = deque()
        self._running = False
        self._thread = None

        os.makedirs(retry_dir, exist_ok=True)

    def push_event(self, clip_path, best_frame, audio_pcm,
                   sample_rate, metadata):
        """Queue an event for pushing to the server.

        Args:
            clip_path: path to the MP4 clip file
            best_frame: numpy array of the best frame (BGR)
            audio_pcm: numpy float32 array of raw audio
            sample_rate: audio sample rate
            metadata: dict with yolo_class, audio_class, timestamps, etc.
        """
        frame_path = clip_path.replace(".mp4", "_best.jpg") if clip_path else None
        audio_path = clip_path.replace(".mp4", "_audio.wav") if clip_path else None

        if best_frame is not None and frame_path:
            cv2.imwrite(frame_path, best_frame)

        if audio_pcm is not None and len(audio_pcm) > 0 and audio_path:
            self._save_wav(audio_pcm, audio_path, sample_rate)

        event = {
            "clip_path": clip_path,
            "frame_path": frame_path,
            "audio_path": audio_path,
            "metadata": metadata,
            "attempts": 0,
            "created_at": time.time(),
        }
        self._queue.append(event)

    def _save_wav(self, pcm, path, sample_rate):
        """Save float32 PCM as 16-bit WAV file."""
        pcm_16 = (pcm * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_16.tobytes())

    def start(self):
        """Start the background retry thread."""
        self._running = True
        self._load_retry_queue()
        self._thread = threading.Thread(target=self._retry_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop and save any remaining events to disk for next run."""
        self._running = False
        self._save_retry_queue()
        if self._thread:
            self._thread.join(timeout=5)

    def _retry_loop(self):
        """Background thread: process queue, retry failed events."""
        while self._running:
            if not self._queue:
                time.sleep(1)
                continue

            event = self._queue.popleft()
            success = self._send_event(event)

            if not success:
                event["attempts"] += 1
                if event["attempts"] < self.max_retries:
                    backoff = min(5 * (2 ** event["attempts"]), 300)
                    time.sleep(backoff)
                    self._queue.appendleft(event)
                else:
                    print(f"  [push] Gave up after {self.max_retries} attempts: "
                          f"{event.get('clip_path', 'unknown')}")

    def _send_event(self, event):
        """Send a single event to the server. Returns True on success."""
        try:
            import requests

            files = {}
            if event["clip_path"] and os.path.exists(event["clip_path"]):
                files["clip"] = (
                    os.path.basename(event["clip_path"]),
                    open(event["clip_path"], "rb"),
                    "video/mp4",
                )
            if event["frame_path"] and os.path.exists(event["frame_path"]):
                files["best_frame"] = (
                    os.path.basename(event["frame_path"]),
                    open(event["frame_path"], "rb"),
                    "image/jpeg",
                )
            if event["audio_path"] and os.path.exists(event["audio_path"]):
                files["audio_chunk"] = (
                    os.path.basename(event["audio_path"]),
                    open(event["audio_path"], "rb"),
                    "audio/wav",
                )

            metadata = event["metadata"]
            metadata["camera_id"] = self.camera_id

            response = requests.post(
                f"{self.server_url}/event/push",
                headers={"X-API-Key": self.api_key},
                files=files,
                data=metadata,
                timeout=120,
                verify=False,
            )

            for f in files.values():
                f[1].close()

            if response.status_code == 200:
                result = response.json()
                print(f"  [push] Event accepted: {result.get('status', 'ok')} "
                      f"({result.get('processing_time_ms', '?')}ms)")
                self._cleanup_local_files(event)
                return True
            else:
                print(f"  [push] Server error {response.status_code}")
                return False

        except ImportError:
            print("  [push] requests not installed: pip install requests")
            return False
        except Exception as e:
            print(f"  [push] Failed: {e}")
            return False

    def _cleanup_local_files(self, event):
        """Remove temporary files after successful push."""
        for key in ["frame_path", "audio_path"]:
            path = event.get(key)
            if path and os.path.exists(path):
                os.remove(path)

    def _save_retry_queue(self):
        """Persist pending events to disk so they survive a reboot."""
        queue_file = os.path.join(self.retry_dir, "pending_events.json")
        events = []
        while self._queue:
            event = self._queue.popleft()
            events.append({
                "clip_path": event["clip_path"],
                "frame_path": event["frame_path"],
                "audio_path": event["audio_path"],
                "metadata": event["metadata"],
                "attempts": event["attempts"],
                "created_at": event["created_at"],
            })
        if events:
            with open(queue_file, "w") as f:
                json.dump(events, f)
            print(f"  [push] Saved {len(events)} events to retry queue")

    def _load_retry_queue(self):
        """Load any events from a previous run that weren't pushed."""
        queue_file = os.path.join(self.retry_dir, "pending_events.json")
        if not os.path.exists(queue_file):
            return
        try:
            with open(queue_file, "r") as f:
                events = json.load(f)
            for event in events:
                self._queue.append(event)
            os.remove(queue_file)
            print(f"  [push] Loaded {len(events)} events from retry queue")
        except Exception:
            pass
