"""Test client — simulates what the Pi will send to the vision server.

Usage:
    # Start server first in another terminal:
    #   cd vision_server && python server.py

    # Then run this:
    python test_client.py <image_path>
    python test_client.py ../20260911T162915_f62.jpg
"""

import requests
import sys
import time
import os

SERVER_URL = os.getenv("VISION_SERVER", "http://localhost:8000")
API_KEY = os.getenv("VISION_API_KEY", "wildlife-vision-secret-2026")


def analyze_image(image_path):
    """Send an image to the vision server and print the result."""
    print(f"Sending: {image_path}")
    print(f"Server:  {SERVER_URL}/analyze")
    print("-" * 50)

    start = time.time()

    with open(image_path, "rb") as f:
        response = requests.post(
            f"{SERVER_URL}/analyze",
            headers={"X-API-Key": API_KEY},
            files={"frame": (os.path.basename(image_path), f, "image/jpeg")},
            data={
                "timestamp": "2026-09-12T10:00:00Z",
                "yolo_class": "person",
                "yolo_confidence": 0.94,
                "brightness": 124.0,
                "motion_pct": 7.3,
                "frame_num": 1,
                "camera_id": "pi-room",
                "trigger_type": "wildlife_person",
            },
            verify=False,
        )

    elapsed = time.time() - start

    if response.status_code == 200:
        result = response.json()
        print(f"Description: {result['description']}")
        print(f"Embedding:   {len(result['embedding'])} dimensions")
        print(f"Server time: {result['processing_time_ms']}ms")
        print(f"Total time:  {elapsed:.1f}s (includes network)")
    elif response.status_code == 403:
        print("ERROR: Invalid API key!")
    else:
        print(f"ERROR {response.status_code}: {response.text}")

    print("-" * 50)


def test_no_auth():
    """Test that requests without API key are rejected."""
    print("Testing without API key (should fail)...")
    try:
        response = requests.post(
            f"{SERVER_URL}/analyze",
            files={"frame": ("test.jpg", b"fake", "image/jpeg")},
            verify=False,
        )
        if response.status_code == 403:
            print("PASS: Request correctly rejected (403)")
        else:
            print(f"FAIL: Got {response.status_code} instead of 403")
    except requests.ConnectionError:
        print("ERROR: Cannot connect to server. Is it running?")
    print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <image_path>")
        print("Example: python test_client.py ../20260911T162915_f62.jpg")
        sys.exit(1)

    # Test auth
    test_no_auth()

    # Test analysis
    analyze_image(sys.argv[1])
