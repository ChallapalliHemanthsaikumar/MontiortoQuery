"""Threaded audio capture from USB microphone.

Uses sounddevice (pip install sounddevice) which wraps PortAudio.
Works on Pi (ALSA), Windows (WASAPI), and Mac (CoreAudio).

Audio capture runs in its own thread via sounddevice's callback API.
Each time the sound card fills a buffer (1024 samples ≈ 21ms at 48kHz),
the callback fires and we push that chunk into the ring buffer.

The main thread never touches the microphone directly — it reads
from the ring buffer whenever it needs audio data.

Hardware setup on Pi:
    USB mic plugged in → shows as "plughw:3,0" or similar
    Find yours with: python -m sounddevice (lists all devices)

High-pass filter at 200Hz removes:
    - Low-frequency wind rumble
    - AC hum (50/60Hz)
    - Vibration through the mount
    These would confuse YAMNet into false positives.
"""

import numpy as np
import threading


class AudioCapture:
    """Continuous audio capture into a ring buffer.

    Usage:
        audio = AudioCapture(ring_buffer, device="plughw:3,0")
        audio.start()
        ...
        audio.stop()
    """

    def __init__(self, ring_buffer, sample_rate=48000, channels=1,
                 chunk_size=1024, device=None, highpass_hz=200):
        self.ring_buffer = ring_buffer
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self.highpass_hz = highpass_hz
        self._stream = None
        self._running = False

        self._prev_sample = 0.0
        self._alpha = self._compute_highpass_alpha(highpass_hz, sample_rate)

    def _compute_highpass_alpha(self, cutoff_hz, sample_rate):
        """First-order IIR high-pass filter coefficient.

        A high-pass filter lets high frequencies through and blocks low ones.
        Alpha close to 1.0 = aggressive filtering (blocks more low freq).
        Alpha close to 0.5 = gentle filtering.

        For 200Hz cutoff at 48kHz: alpha ≈ 0.974
        This removes wind/hum while keeping speech (300Hz+) intact.
        """
        if cutoff_hz <= 0:
            return 0.0
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        dt = 1.0 / sample_rate
        return rc / (rc + dt)

    def _highpass_filter(self, chunk):
        """Apply first-order IIR high-pass filter in-place.

        IIR = Infinite Impulse Response. "Infinite" because each output
        depends on the previous output (feedback loop), so a single input
        pulse affects the output forever (though it decays quickly).

        The math: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        - x[n] = current input sample
        - y[n] = current output (filtered) sample
        - alpha = how aggressively to filter (0.974 for 200Hz @ 48kHz)

        This is the simplest possible high-pass filter. One multiply,
        two additions per sample. Runs in <1ms for 1024 samples on Pi.
        """
        if self._alpha <= 0:
            return chunk

        filtered = np.empty_like(chunk)
        prev = self._prev_sample
        for i in range(len(chunk)):
            prev = self._alpha * (prev + chunk[i] - (chunk[i - 1] if i > 0 else self._prev_sample))
            filtered[i] = prev
        self._prev_sample = prev
        return filtered

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice on each audio chunk (runs on audio thread).

        indata: numpy array shape (chunk_size, channels), dtype float32
        frames: number of samples in this chunk
        status: error flags (overflow, underflow)

        This callback MUST be fast — if it takes too long, audio drops.
        That's why we just filter + push to buffer, nothing else.
        """
        if status:
            pass

        chunk = indata[:, 0].copy() if self.channels == 1 else indata.copy()
        chunk = self._highpass_filter(chunk)
        self.ring_buffer.push(chunk)

    def start(self):
        """Start capturing audio in background thread."""
        import sounddevice as sd

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            device=self.device,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        """Stop capturing."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def is_alive(self):
        return self._stream is not None and self._stream.active

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
