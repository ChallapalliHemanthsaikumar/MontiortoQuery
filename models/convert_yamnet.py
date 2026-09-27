"""Convert yamnet.tflite → yamnet.onnx

One-time conversion. Run this once, push yamnet.onnx to git.
Pi pulls and runs with onnxruntime — same pattern as yolov8n.onnx.

Usage:
    pip install tf2onnx onnxruntime
    python models/convert_yamnet.py
"""

import os
import sys
import subprocess

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(MODELS_DIR, "yamnet.tflite")
ONNX_PATH = os.path.join(MODELS_DIR, "yamnet.onnx")


def install_if_missing(package):
    try:
        __import__(package.replace("-", "_"))
        return True
    except ImportError:
        print(f"  Installing {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "-q"]
        )
        return True


def convert():
    print("=" * 50)
    print("  YAMNET TFLite → ONNX Converter")
    print("=" * 50)

    if not os.path.exists(TFLITE_PATH):
        print(f"  ERROR: {TFLITE_PATH} not found")
        print(f"  Run: python models/download_models.py first")
        sys.exit(1)

    if os.path.exists(ONNX_PATH):
        size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        print(f"  yamnet.onnx already exists ({size_mb:.1f}MB)")
        print(f"  Delete it first if you want to reconvert.")
        return

    install_if_missing("onnxruntime")
    install_if_missing("tf2onnx")

    print(f"\n  Converting {TFLITE_PATH} → {ONNX_PATH}")
    print(f"  This takes 10-30 seconds...\n")

    result = subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--tflite", TFLITE_PATH,
            "--output", ONNX_PATH,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and os.path.exists(ONNX_PATH):
        size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        print(f"  OK: yamnet.onnx created ({size_mb:.1f}MB)")
        verify()
    else:
        print(f"  Conversion failed:")
        print(f"  {result.stderr[:500]}")
        if "tensorflow" in result.stderr.lower():
            print(f"\n  tf2onnx needs tensorflow for this model.")
            print(f"  Run: pip install tensorflow")
            print(f"  Then: python models/convert_yamnet.py")
            print(f"  Then: pip uninstall tensorflow  (optional, free up space)")


def verify():
    """Quick sanity check — load the ONNX model and inspect shape."""
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(ONNX_PATH)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        print(f"\n  Model verified:")
        print(f"    Input:  {input_info.name} shape={input_info.shape} "
              f"dtype={input_info.type}")
        print(f"    Output: {output_info.name} shape={output_info.shape} "
              f"dtype={output_info.type}")

        # Test inference with 1 second of silence at 16kHz
        test_audio = np.zeros(16000, dtype=np.float32)
        outputs = session.run(None, {input_info.name: test_audio})
        scores = outputs[0]
        print(f"    Test inference: output shape={scores.shape}")
        if len(scores.shape) == 2:
            print(f"    Classes: {scores.shape[-1]}")
        print(f"\n  Ready to use. Run:")
        print(f"    python -m edge_v2.main --live --show")

    except Exception as e:
        print(f"\n  Verification failed: {e}")
        print(f"  The ONNX file may still work — try running the pipeline.")

    print(f"\n  Push to git:")
    print(f"    git add models/yamnet.onnx")
    print(f"    git push")


if __name__ == "__main__":
    convert()
