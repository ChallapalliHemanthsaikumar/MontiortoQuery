"""Audio classifier — two modes, zero heavy dependencies.

Mode 1: SIMPLE (default, numpy only)
    Uses spectral analysis to classify audio into 6 categories:
    silence, speech, music, noise, impact, animal

    How it works (all numpy, no ML model):
        Raw PCM → FFT → spectral features → threshold rules → class

    Features extracted:
        - RMS energy: how loud is it? silence < 0.01
        - Spectral centroid: where is the frequency center?
            speech lives at 300-3000 Hz
            music is broader, 100-8000 Hz
            wind/noise is very spread out
        - Zero-crossing rate: how often does the wave cross zero?
            speech: moderate (50-200 crossings/sec)
            noise: very high (>300)
            music: low to moderate
        - Spectral flatness: how "flat" is the spectrum?
            white noise = 1.0 (all frequencies equal)
            pure tone = 0.0 (one frequency dominates)
            speech ≈ 0.1-0.3
        - Spectral rolloff: below what frequency is 85% of energy?
            speech: 1000-4000 Hz
            noise: spread high
            bass/engine: low

    Accuracy: ~75% for the 6 categories (good enough for fusion gate)
    Speed: <1ms per 1-second chunk (just numpy FFT)

Mode 2: YAMNET ONNX (when yamnet.onnx is available)
    Full 521-class YAMNet model via onnxruntime.
    pip install onnxruntime (15MB, works everywhere)
    Accuracy: ~90%+ (Google-trained on AudioSet)

    The mel spectrogram preprocessing runs in numpy:
        PCM → pre-emphasis → windowing → FFT → mel filterbank → log
    Then onnxruntime runs the neural network on those features.
"""

import os
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

HIGH_PRIORITY_SOUNDS = {
    "speech", "conversation", "narration",
    "shout", "scream", "cry",
    "glass_break", "shatter",
    "siren",
    "gunshot",
    "dog", "bark", "growl",
    "alarm", "smoke_alarm",
    "doorbell", "knock",
}

MEDIUM_PRIORITY_SOUNDS = {
    "vehicle", "car", "engine",
    "footsteps",
    "door", "slam",
    "music", "singing",
    "horn",
    "motorcycle", "truck",
}

IGNORE_SOUNDS = {
    "silence",
    "wind", "noise",
    "rain",
    "ambient",
}


class SimpleAudioClassifier:
    """Spectral-feature audio classifier. Pure numpy, no model file.

    Classifies 1-second audio chunks into:
        silence, speech, music, noise, impact, animal
    """

    def __init__(self, sample_rate=16000, min_confidence=0.3):
        self.sample_rate = sample_rate
        self.min_confidence = min_confidence

    def classify(self, pcm_audio, source_rate=48000):
        if len(pcm_audio) == 0:
            return [("silence", 0.95)]

        if source_rate != self.sample_rate:
            ratio = int(source_rate / self.sample_rate)
            if ratio > 1:
                pcm_audio = pcm_audio[::ratio]

        audio = pcm_audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        features = self._extract_features(audio)
        return self._classify_features(features)

    def _extract_features(self, audio):
        n = len(audio)

        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))

        # Zero-crossing rate
        signs = np.sign(audio)
        zcr = np.sum(np.abs(np.diff(signs)) > 0) / n

        # FFT magnitude spectrum
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)

        # Avoid division by zero
        mag_sum = np.sum(magnitude) + 1e-10

        # Spectral centroid (frequency center of mass)
        centroid = np.sum(freqs * magnitude) / mag_sum

        # Spectral bandwidth (spread around centroid)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / mag_sum)

        # Spectral rolloff (frequency below which 85% of energy lives)
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

        # Spectral flatness (geometric mean / arithmetic mean)
        # 1.0 = white noise, 0.0 = pure tone
        log_mag = np.log(magnitude + 1e-10)
        geo_mean = np.exp(np.mean(log_mag))
        arith_mean = np.mean(magnitude) + 1e-10
        flatness = geo_mean / arith_mean

        # Energy in speech band (300-3000 Hz)
        speech_mask = (freqs >= 300) & (freqs <= 3000)
        speech_energy = np.sum(magnitude[speech_mask]) / mag_sum

        # Energy in low band (50-300 Hz) — engines, bass
        low_mask = (freqs >= 50) & (freqs <= 300)
        low_energy = np.sum(magnitude[low_mask]) / mag_sum

        return {
            "rms": rms,
            "zcr": zcr,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "flatness": flatness,
            "speech_energy": speech_energy,
            "low_energy": low_energy,
        }

    def _classify_features(self, f):
        """Rule-based classification from spectral features."""
        results = []

        # Silence: very low energy
        if f["rms"] < 0.005:
            return [("silence", 0.95)]

        # Impact: very high energy, high ZCR (glass break, slam, bark)
        if f["rms"] > 0.15 and f["zcr"] > 0.15:
            results.append(("impact", min(0.95, f["rms"] * 3)))

        # Speech: energy concentrated in 300-3000Hz, moderate ZCR
        speech_score = 0.0
        if f["speech_energy"] > 0.4 and 0.03 < f["zcr"] < 0.2:
            speech_score = f["speech_energy"] * 0.7 + (1.0 - f["flatness"]) * 0.3
            if 500 < f["centroid"] < 3000:
                speech_score += 0.15
            speech_score = min(0.95, speech_score)
            if speech_score > 0.3:
                results.append(("speech", round(speech_score, 3)))

        # Engine/vehicle: energy in low band, low ZCR
        if f["low_energy"] > 0.3 and f["zcr"] < 0.1 and f["centroid"] < 500:
            engine_score = f["low_energy"] * 0.6 + (1 - f["zcr"]) * 0.2
            results.append(("engine", round(min(0.9, engine_score), 3)))

        # Music: wider bandwidth, moderate energy, low flatness
        if (f["bandwidth"] > 1500 and f["flatness"] < 0.2 and
                f["rms"] > 0.02 and speech_score < 0.4):
            music_score = (1 - f["flatness"]) * 0.5 + min(f["bandwidth"] / 4000, 0.5)
            results.append(("music", round(min(0.85, music_score), 3)))

        # Wind/noise: high flatness, high ZCR
        if f["flatness"] > 0.4 and f["zcr"] > 0.15:
            noise_score = f["flatness"] * 0.6 + f["zcr"] * 0.4
            results.append(("noise", round(min(0.9, noise_score), 3)))

        # Animal: high-pitched, short bursts (rough heuristic)
        if (f["centroid"] > 2000 and f["rms"] > 0.05 and
                f["zcr"] > 0.1 and speech_score < 0.3):
            animal_score = 0.4
            results.append(("animal", animal_score))

        if not results:
            results.append(("ambient", 0.5))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r for r in results if r[1] >= self.min_confidence]

    def get_priority(self, results):
        if not results:
            return "ignore"
        for name, conf in results:
            if name in HIGH_PRIORITY_SOUNDS:
                return "high"
        for name, conf in results:
            if name in MEDIUM_PRIORITY_SOUNDS:
                return "medium"
        for name, conf in results:
            if name in IGNORE_SOUNDS:
                return "ignore"
        return "low"

    def is_speech(self, results):
        return any(name == "speech" for name, _ in results)

    def get_dominant_class(self, results):
        if not results:
            return "silence", 0.0
        return results[0]


class YAMNetClassifier:
    """Full YAMNet via ONNX Runtime (521 classes, ~90% accuracy).

    Requires:
        pip install onnxruntime   (15MB, works on Pi + Windows + Mac)
        models/yamnet.onnx        (run: python models/convert_yamnet.py)

    The converted model takes raw 16kHz float32 audio as input —
    mel spectrogram computation happens inside the model, same as
    the original TFLite version. Just feed it audio samples.
    """

    def __init__(self, model_path=None, class_map_path=None,
                 sample_rate=16000, top_k=3, min_confidence=0.3):
        self.sample_rate = sample_rate
        self.top_k = top_k
        self.min_confidence = min_confidence

        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "yamnet.onnx")
        if class_map_path is None:
            class_map_path = os.path.join(MODELS_DIR, "yamnet_classes.csv")

        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.class_names = self._load_class_map(class_map_path)

    def _load_class_map(self, path):
        import csv
        names = {}
        if not os.path.exists(path):
            return names
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            name_col = header.index("display_name") if "display_name" in header else 2
            for i, row in enumerate(reader):
                if len(row) > name_col:
                    names[i] = row[name_col].strip()
        return names

    def classify(self, pcm_audio, source_rate=48000):
        if len(pcm_audio) == 0:
            return [("silence", 0.95)]

        # Resample to 16kHz
        if source_rate == 48000:
            audio = pcm_audio[::3]
        elif source_rate != 16000:
            ratio = source_rate / 16000
            indices = np.arange(0, len(pcm_audio), ratio).astype(int)
            audio = pcm_audio[indices]
        else:
            audio = pcm_audio

        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        outputs = self.session.run(None, {self.input_name: audio})
        scores = outputs[0]

        if len(scores.shape) == 2:
            scores = np.mean(scores, axis=0)

        top_indices = np.argsort(scores)[::-1][:self.top_k]
        results = []
        for idx in top_indices:
            name = self.class_names.get(idx, f"class_{idx}")
            conf = float(scores[idx])
            if conf >= self.min_confidence:
                results.append((name, round(conf, 3)))
        return results

    def get_priority(self, results):
        if not results:
            return "ignore"
        for name, conf in results:
            if name in HIGH_PRIORITY_SOUNDS:
                return "high"
        for name, conf in results:
            if name in MEDIUM_PRIORITY_SOUNDS:
                return "medium"
        for name, conf in results:
            if name in IGNORE_SOUNDS:
                return "ignore"
        return "low"

    def is_speech(self, results):
        speech_classes = {"Speech", "Conversation", "Narration, monologue"}
        return any(name in speech_classes for name, _ in results)

    def get_dominant_class(self, results):
        if not results:
            return "silence", 0.0
        return results[0]


class YAMNetTFLiteClassifier:
    """YAMNet via tflite-runtime (521 classes).

    The lightest path to full YAMNet:
        pip install tflite-runtime   (5MB, works on Pi + Windows + Mac)
        models/yamnet.tflite         (3MB, already downloaded)

    The TFLite model takes raw 16kHz float32 audio as input.
    It does the mel spectrogram conversion internally — no preprocessing
    needed on our side. Just feed it audio samples.
    """

    def __init__(self, model_path=None, class_map_path=None,
                 top_k=3, min_confidence=0.3):
        self.top_k = top_k
        self.min_confidence = min_confidence

        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "yamnet.tflite")
        if class_map_path is None:
            class_map_path = os.path.join(MODELS_DIR, "yamnet_classes.csv")

        self.class_names = self._load_class_map(class_map_path)
        self.interpreter = self._load_model(model_path)

    def _load_class_map(self, path):
        import csv
        names = {}
        if not os.path.exists(path):
            return names
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            name_col = header.index("display_name") if "display_name" in header else 2
            for i, row in enumerate(reader):
                if len(row) > name_col:
                    names[i] = row[name_col].strip()
        return names

    def _load_model(self, path):
        # Try lightweight runtimes first, fall back to full tensorflow
        # Pi: tflite-runtime works (ARM wheels exist)
        # Windows: ai-edge-litert or tensorflow-cpu
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=path)
        except ImportError:
            try:
                from ai_edge_litert import interpreter as litert
                interpreter = litert.Interpreter(model_path=path)
            except ImportError:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        return interpreter

    def classify(self, pcm_audio, source_rate=48000):
        if len(pcm_audio) == 0:
            return [("silence", 0.95)]

        # Resample to 16kHz (YAMNet expects this)
        if source_rate == 48000:
            audio = pcm_audio[::3]
        elif source_rate != 16000:
            ratio = source_rate / 16000
            indices = np.arange(0, len(pcm_audio), ratio).astype(int)
            audio = pcm_audio[indices]
        else:
            audio = pcm_audio

        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.resize_tensor_input(
            input_details[0]["index"], audio.shape
        )
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(input_details[0]["index"], audio)
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(output_details[0]["index"])
        if len(scores.shape) == 2:
            scores = np.mean(scores, axis=0)

        top_indices = np.argsort(scores)[::-1][:self.top_k]
        results = []
        for idx in top_indices:
            name = self.class_names.get(idx, f"class_{idx}")
            conf = float(scores[idx])
            if conf >= self.min_confidence:
                results.append((name, round(conf, 3)))
        return results

    def get_priority(self, results):
        if not results:
            return "ignore"
        for name, conf in results:
            if name in HIGH_PRIORITY_SOUNDS:
                return "high"
        for name, conf in results:
            if name in MEDIUM_PRIORITY_SOUNDS:
                return "medium"
        for name, conf in results:
            if name in IGNORE_SOUNDS:
                return "ignore"
        return "low"

    def is_speech(self, results):
        speech_classes = {"Speech", "Conversation", "Narration, monologue"}
        return any(name in speech_classes for name, _ in results)

    def get_dominant_class(self, results):
        if not results:
            return "silence", 0.0
        return results[0]


def AudioClassifier(**kwargs):
    """Factory: returns best available classifier.

    Priority order:
        1. YAMNet TFLite (tflite-runtime) — 521 classes, lightest install
        2. YAMNet ONNX (onnxruntime) — 521 classes
        3. Simple spectral (numpy) — 6 categories, no install needed
    """
    # Try 1: YAMNet TFLite (pip install tflite-runtime)
    tflite_path = os.path.join(MODELS_DIR, "yamnet.tflite")
    if os.path.exists(tflite_path):
        try:
            clf = YAMNetTFLiteClassifier(model_path=tflite_path)
            print("  Audio: YAMNet TFLite (521 classes)")
            return clf
        except ImportError:
            pass
        except Exception as e:
            print(f"  Audio: YAMNet TFLite failed ({e})")

    # Try 2: YAMNet ONNX (pip install onnxruntime)
    onnx_path = os.path.join(MODELS_DIR, "yamnet.onnx")
    if os.path.exists(onnx_path):
        try:
            clf = YAMNetClassifier(model_path=onnx_path)
            print("  Audio: YAMNet ONNX (521 classes)")
            return clf
        except ImportError:
            pass
        except Exception as e:
            print(f"  Audio: YAMNet ONNX failed ({e})")

    # Fallback: simple spectral classifier (numpy only)
    print("  Audio: simple spectral classifier (numpy, 6 categories)")
    print("  To upgrade → pip install ai-edge-litert  (Windows)")
    print("             → pip install tflite-runtime   (Pi)")
    return SimpleAudioClassifier()
