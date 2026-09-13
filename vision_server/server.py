"""Vision Server — receives frames from Pi, runs Qwen VLM, returns descriptions.

Usage:
    cd vision_server
    python server.py

    # Or with HTTPS:
    USE_HTTPS=true python server.py
"""

import os
import sys
import time
import ssl
import subprocess
import json
import csv
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import uvicorn

from config import SERVER_HOST, SERVER_PORT, API_KEY, USE_HTTPS, CERT_DIR
from vlm import describe_frame
from embeddings import get_embedding

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(title="Wildlife Vision Server")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def init_csv():
    """Create events CSV if it doesn't exist."""
    csv_path = os.path.join(DATA_DIR, "events.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "camera_id", "trigger_type", "yolo_class",
                "yolo_confidence", "brightness", "motion_pct",
                "vlm_description", "image_path", "processing_time_ms",
            ])
    return csv_path


def save_event(event, image_bytes):
    """Save event to CSV and image to disk."""
    # Save image
    ts = event["timestamp"].replace(":", "").replace("-", "")[:15] or \
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{ts}_{event.get('yolo_class', 'unknown')}.jpg"
    image_path = os.path.join(IMAGES_DIR, filename)
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Append to CSV
    csv_path = init_csv()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            event["timestamp"], event.get("camera_id", ""),
            event.get("trigger_type", ""), event.get("yolo_class", ""),
            event.get("yolo_confidence", 0), event.get("brightness", 0),
            event.get("motion_pct", 0), event["description"],
            image_path, event["processing_time_ms"],
        ])

    # Save JSON event (for Neo4j migration later)
    json_path = os.path.join(DATA_DIR, "events.json")
    events = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            events = json.load(f)
    event["image_path"] = image_path
    events.append(event)
    with open(json_path, "w") as f:
        json.dump(events, f, indent=2)

    return image_path


async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


@app.get("/health")
async def health():
    """Health check — no auth required."""
    return {"status": "ok", "model": "qwen2.5vl:3b"}


@app.post("/analyze")
async def analyze(
    frame: UploadFile = File(...),
    timestamp: str = Form(""),
    yolo_class: str = Form(""),
    yolo_confidence: float = Form(0.0),
    brightness: float = Form(0.0),
    motion_pct: float = Form(0.0),
    frame_num: int = Form(0),
    camera_id: str = Form("pi-default"),
    trigger_type: str = Form(""),
    api_key: str = Depends(verify_api_key),
):
    """Analyze a frame with Qwen2.5-VL and return description + embedding."""
    start = time.time()

    image_bytes = await frame.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image")

    # Run VLM
    description = describe_frame(
        image_bytes,
        yolo_class=yolo_class,
        yolo_confidence=yolo_confidence,
    )

    # Generate embedding for semantic search
    embedding = get_embedding(description)

    elapsed = time.time() - start

    event = {
        "description": description,
        "embedding": embedding,
        "yolo_class": yolo_class,
        "yolo_confidence": yolo_confidence,
        "timestamp": timestamp,
        "brightness": brightness,
        "motion_pct": motion_pct,
        "camera_id": camera_id,
        "trigger_type": trigger_type,
        "frame_num": frame_num,
        "processing_time_ms": int(elapsed * 1000),
    }

    # Store locally (will forward to Neo4j in Phase 3)
    image_path = save_event(event, image_bytes)

    print(f"[{timestamp or 'no-ts'}] {trigger_type} | "
          f"{yolo_class or 'unknown'} | {elapsed:.1f}s | "
          f"{description[:80]}...")
    print(f"  -> saved: {image_path}")

    return {"status": "stored", "processing_time_ms": int(elapsed * 1000)}


def generate_self_signed_cert():
    """Generate self-signed TLS cert for HTTPS."""
    os.makedirs(CERT_DIR, exist_ok=True)
    cert_file = os.path.join(CERT_DIR, "server.crt")
    key_file = os.path.join(CERT_DIR, "server.key")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file

    print("Generating self-signed TLS certificate...")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", key_file, "-out", cert_file,
        "-days", "365", "-nodes",
        "-subj", "/CN=wildlife-vision-server",
    ], check=True)
    print(f"Certificate saved to {CERT_DIR}/")
    return cert_file, key_file


if __name__ == "__main__":
    print("=" * 50)
    print("  WILDLIFE VISION SERVER")
    print("=" * 50)
    print(f"  Host:     {SERVER_HOST}:{SERVER_PORT}")
    print(f"  HTTPS:    {USE_HTTPS}")
    print(f"  API Key:  {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"  Model:    qwen2.5vl:3b")
    print("=" * 50)
    print()
    print("  Endpoints:")
    print(f"    GET  /health          — health check (no auth)")
    print(f"    POST /analyze         — send image, get description")
    print()
    print("  Pi command:")
    proto = "https" if USE_HTTPS else "http"
    print(f"    curl -X POST {proto}://<this-ip>:{SERVER_PORT}/analyze \\")
    print(f"      -H 'X-API-Key: {API_KEY}' \\")
    print(f"      -F 'frame=@test.jpg' -F 'yolo_class=person'")
    print("=" * 50)

    ssl_kwargs = {}
    if USE_HTTPS:
        cert_file, key_file = generate_self_signed_cert()
        ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        **ssl_kwargs,
    )
