"""Ring buffers for video frames and audio PCM.

A ring buffer keeps the last N items in memory. When full, new items
push out the oldest. This gives you "pre-roll" — the seconds BEFORE
a trigger event happened.

Video buffer: stores (timestamp, frame) tuples
Audio buffer: stores (timestamp, pcm_chunk) tuples

Both use collections.deque(maxlen=N) which is:
  - O(1) append (add to right)
  - O(1) popleft (remove from left)
  - Thread-safe for single-producer single-consumer
  - Automatic eviction when maxlen is reached
"""

import time
from collections import deque
from threading import Lock


class VideoRingBuffer:
    """Keeps the last N seconds of video frames in memory.

    At 15fps with 10s buffer = 150 frames × ~300KB each ≈ 45MB RAM.
    At 30fps with 10s buffer = 300 frames × ~300KB each ≈ 90MB RAM.
    Pi 4 has 4GB — this is fine.
    """

    def __init__(self, duration_seconds=10, fps=15):
        self.max_frames = int(duration_seconds * fps)
        self.fps = fps
        self._buffer = deque(maxlen=self.max_frames)
        self._lock = Lock()

    def push(self, frame):
        ts = time.time()
        with self._lock:
            self._buffer.append((ts, frame))

    def drain(self):
        """Return all buffered frames and clear. Used when recording a clip."""
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
        return frames

    def peek_latest(self):
        """Get the most recent frame without removing it."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    def __len__(self):
        return len(self._buffer)


class AudioRingBuffer:
    """Keeps the last N seconds of raw PCM audio in memory.

    Audio comes in as numpy arrays (chunks of samples).
    At 48kHz mono float32, 10 seconds = 480,000 × 4 bytes = ~1.9MB.
    Negligible on Pi.
    """

    def __init__(self, duration_seconds=10, sample_rate=48000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        max_chunks = int((duration_seconds * sample_rate) / chunk_size)
        self._buffer = deque(maxlen=max_chunks)
        self._lock = Lock()

    def push(self, pcm_chunk):
        ts = time.time()
        with self._lock:
            self._buffer.append((ts, pcm_chunk))

    def drain(self):
        """Return all buffered audio chunks and clear."""
        with self._lock:
            chunks = list(self._buffer)
            self._buffer.clear()
        return chunks

    def get_last_n_seconds(self, seconds):
        """Get the last N seconds of audio without draining."""
        n_chunks = int((seconds * self.sample_rate) / self.chunk_size)
        with self._lock:
            items = list(self._buffer)
        return items[-n_chunks:] if len(items) >= n_chunks else items

    def get_chunks_after(self, timestamp):
        """Get all chunks newer than the given timestamp."""
        with self._lock:
            items = list(self._buffer)
        return [(ts, chunk) for ts, chunk in items if ts > timestamp]

    def __len__(self):
        return len(self._buffer)
