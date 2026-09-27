"""Smart deduplication — don't save the same scene twice.

Problem: A person stands at the gate for 5 minutes. Motion detector fires
every few seconds. Without dedup: 50 nearly identical events. Useless.

Solution: Two-layer dedup on the Pi (fast, before network):
    1. pHash visual check: is this frame visually the same as the last saved?
    2. Adaptive cooldown: different event types get different wait times

pHash (Perceptual Hash):
    Shrink → grayscale → DCT → threshold → 64-bit fingerprint.
    Compare two hashes with Hamming distance (count differing bits).
    Same scene = distance < 10. Different scene = distance > 15.

Adaptive cooldowns:
    speech  → 5s   (people talk fast, each sentence matters)
    person  → 15s  (walking speed, scene changes slowly)
    car     → 30s  (car takes time to pass through frame)
    ambient → 60s  (nothing urgent, save disk space)

The server runs a SECOND dedup pass using CLIP embeddings (more accurate
but needs GPU). The Pi dedup is a fast filter to avoid sending duplicates
over WiFi.
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple


class PerceptualHasher:
    """Compute and compare perceptual hashes using DCT."""

    def __init__(self, hash_size=8):
        self.hash_size = hash_size
        self.resize_to = hash_size * 4

    def compute(self, frame) -> int:
        """Compute 64-bit perceptual hash of a frame.

        Steps:
            1. Resize to 32×32 (removes fine detail)
            2. Grayscale
            3. DCT (Discrete Cosine Transform)
            4. Keep 8×8 low-frequency block
            5. Threshold at median → binary hash
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.resize_to, self.resize_to),
                           interpolation=cv2.INTER_AREA)

        dct = cv2.dct(np.float32(small))
        low_freq = dct[:self.hash_size, :self.hash_size]

        median = np.median(low_freq)
        bits = (low_freq > median).flatten()

        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    def distance(self, hash1: int, hash2: int) -> int:
        """Hamming distance — count bits that differ between two hashes.

        XOR gives 1 where bits differ, then count the 1s.
        """
        xor = hash1 ^ hash2
        return bin(xor).count("1")


COOLDOWNS = {
    "person_speech": 5.0,
    "person_silent": 15.0,
    "animal": 15.0,
    "vehicle": 30.0,
    "audio_only": 10.0,
    "unclassified_motion": 20.0,
    "heartbeat": 0.0,
}


class SmartDedup:
    """Two-layer deduplication: pHash + adaptive cooldown.

    Usage:
        dedup = SmartDedup()

        # On each fusion gate SAVE decision:
        should_save, reason = dedup.check(frame, event_type)
        if should_save:
            record_clip()
            dedup.record_save(frame, event_type)
    """

    def __init__(self, phash_threshold=10, default_cooldown=15.0):
        self.hasher = PerceptualHasher()
        self.phash_threshold = phash_threshold
        self.default_cooldown = default_cooldown

        self._last_hash: Optional[int] = None
        self._last_save_time: float = 0.0
        self._last_event_type: Optional[str] = None
        self._last_audio_class: Optional[str] = None

    def check(self, frame, event_type: str, audio_class: str = ""
              ) -> Tuple[bool, str]:
        """Check if this event should be saved or is a duplicate.

        Returns:
            (should_save, reason) tuple
        """
        now = time.time()

        # Heartbeats always pass
        if event_type == "heartbeat":
            return True, "heartbeat"

        # Cooldown check
        cooldown = COOLDOWNS.get(event_type, self.default_cooldown)
        elapsed = now - self._last_save_time
        if elapsed < cooldown:
            return False, f"cooldown: {elapsed:.1f}s < {cooldown:.1f}s"

        # New speech content overrides visual dedup
        if event_type == "person_speech" and self._last_event_type != "person_speech":
            return True, "new speech event (overrides visual dedup)"

        if (event_type == "person_speech" and
                audio_class and audio_class != self._last_audio_class):
            return True, "audio class changed"

        # pHash visual dedup
        current_hash = self.hasher.compute(frame)
        if self._last_hash is not None:
            dist = self.hasher.distance(current_hash, self._last_hash)
            if dist < self.phash_threshold:
                return False, f"pHash similar: distance={dist} < {self.phash_threshold}"

        return True, "new scene"

    def record_save(self, frame, event_type: str, audio_class: str = ""):
        """Record that we saved this event (update dedup state)."""
        self._last_hash = self.hasher.compute(frame)
        self._last_save_time = time.time()
        self._last_event_type = event_type
        self._last_audio_class = audio_class
