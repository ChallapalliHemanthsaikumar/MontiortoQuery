"""Download + prepare all models for edge_v2 pipeline.

One command does everything:
    python models/download_models.py

What it does:
    1. Copies yolov8n.onnx from repo root into models/
    2. Downloads yamnet.tflite (3MB) from TF Hub
    3. Downloads yamnet_classes.csv (class names)
    4. Converts yamnet.tflite → yamnet.onnx (auto-installs tf2onnx)

After this, push models/ to git. Pi pulls and runs — no installs on device
except onnxruntime.
"""

import os
import sys
import shutil
import subprocess
import urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url, dest_path, description=""):
    filename = os.path.basename(dest_path)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  [SKIP] {filename} already exists ({size_mb:.1f}MB)")
        return True

    print(f"  [DOWNLOAD] {filename} — {description}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  [OK] {filename} ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {filename}: {e}")
        return False


def copy_yolo_from_root():
    root_onnx = os.path.join(MODELS_DIR, "..", "yolov8n.onnx")
    dest_onnx = os.path.join(MODELS_DIR, "yolov8n.onnx")

    if os.path.exists(dest_onnx):
        size_mb = os.path.getsize(dest_onnx) / (1024 * 1024)
        print(f"  [SKIP] yolov8n.onnx already exists ({size_mb:.1f}MB)")
        return True

    if os.path.exists(root_onnx):
        shutil.copy2(root_onnx, dest_onnx)
        size_mb = os.path.getsize(dest_onnx) / (1024 * 1024)
        print(f"  [COPY] yolov8n.onnx from repo root ({size_mb:.1f}MB)")
        return True

    print("  [WARN] yolov8n.onnx not found in repo root")
    return False


def pip_install(package):
    """Install a pip package if not already installed."""
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        print(f"  [INSTALL] {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            stdout=subprocess.DEVNULL,
        )


def convert_yamnet_to_onnx():
    """Convert yamnet.tflite → yamnet.onnx using tf2onnx."""
    tflite_path = os.path.join(MODELS_DIR, "yamnet.tflite")
    onnx_path = os.path.join(MODELS_DIR, "yamnet.onnx")

    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  [SKIP] yamnet.onnx already exists ({size_mb:.1f}MB)")
        return True

    if not os.path.exists(tflite_path):
        print(f"  [SKIP] yamnet.tflite not found, cannot convert")
        return False

    print(f"  [CONVERT] yamnet.tflite → yamnet.onnx")

    # Install conversion tools
    pip_install("onnxruntime")
    pip_install("tf2onnx")

    print(f"  Converting (10-30 seconds)...")
    result = subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--tflite", tflite_path,
            "--output", onnx_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  [OK] yamnet.onnx ({size_mb:.1f}MB)")
        return True

    # tf2onnx might need tensorflow for some models
    if "tensorflow" in result.stderr.lower() or "No module" in result.stderr:
        print(f"  [INFO] tf2onnx needs tensorflow for this conversion.")
        print(f"  Installing tensorflow (one-time, can remove after)...")
        try:
            pip_install("tensorflow")
            result = subprocess.run(
                [
                    sys.executable, "-m", "tf2onnx.convert",
                    "--tflite", tflite_path,
                    "--output", onnx_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and os.path.exists(onnx_path):
                size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"  [OK] yamnet.onnx ({size_mb:.1f}MB)")
                print(f"  TIP: run 'pip uninstall tensorflow -y' to free ~500MB")
                return True
        except Exception as e:
            print(f"  [FAIL] tensorflow install failed: {e}")

    print(f"  [FAIL] Conversion failed: {result.stderr[:300]}")
    return False


def verify_yamnet_onnx():
    """Quick test — load model and run silent audio through it."""
    onnx_path = os.path.join(MODELS_DIR, "yamnet.onnx")
    if not os.path.exists(onnx_path):
        return

    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        print(f"  [VERIFY] Input: {inp.name} {inp.shape} | Output: {out.name} {out.shape}")

        test_audio = np.zeros(16000, dtype=np.float32)
        scores = session.run(None, {inp.name: test_audio})[0]
        print(f"  [VERIFY] Inference OK — output shape {scores.shape}")
    except Exception as e:
        print(f"  [VERIFY] Warning: {e}")


def main():
    print("=" * 55)
    print("  MODEL SETUP — edge_v2 pipeline (one-time)")
    print("=" * 55)

    # Step 1: YOLO
    print("\n--- YOLO (object detection) ---")
    copy_yolo_from_root()

    # Step 2: YAMNet downloads
    print("\n--- YAMNet (audio classification) ---")
    tflite_path = os.path.join(MODELS_DIR, "yamnet.tflite")
    classes_path = os.path.join(MODELS_DIR, "yamnet_classes.csv")

    download_file(
        "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite",
        tflite_path,
        "Audio classifier — 521 classes",
    )
    download_file(
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
        classes_path,
        "Class name mapping",
    )

    # Step 3: Convert to ONNX
    print("\n--- Convert to ONNX ---")
    convert_yamnet_to_onnx()
    verify_yamnet_onnx()

    # Summary
    print("\n" + "=" * 55)
    print("  MODELS READY")
    print("=" * 55)
    for f in ["yolov8n.onnx", "yamnet.onnx", "yamnet_classes.csv"]:
        path = os.path.join(MODELS_DIR, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK  {f:25s} {size_mb:.1f}MB")
        else:
            print(f"  --  {f:25s} missing")

    print(f"\n  Push to git:")
    print(f"    git add models/")
    print(f"    git push")
    print("=" * 55)


if __name__ == "__main__":
    main()
