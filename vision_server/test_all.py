"""Test the full pipeline: VLM + entity extraction + Neo4j + semantic search.

Usage:
    python test_all.py                          # test with first available image
    python test_all.py data/images/some.jpg     # test with specific image
"""

import os
import sys
import time
import requests
import json

SERVER = "http://localhost:8000"
API_KEY = "wildlife-vision-secret-2026"
HEADERS = {"X-API-Key": API_KEY}
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "images")


def find_test_image():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if os.path.exists(IMAGES_DIR):
        images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])
        if images:
            return os.path.join(IMAGES_DIR, images[-1])
    print("No test image found. Pass an image path as argument.")
    sys.exit(1)


def test_health():
    print("=" * 60)
    print("  TEST 1: Health Check")
    print("=" * 60)
    r = requests.get(f"{SERVER}/health")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.json()}")
    print()


def test_analyze(image_path):
    print("=" * 60)
    print(f"  TEST 2: Analyze Image")
    print(f"  Image: {image_path}")
    print("=" * 60)
    start = time.time()
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{SERVER}/analyze",
            headers=HEADERS,
            files={"frame": ("test.jpg", f, "image/jpeg")},
            data={
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "yolo_class": "person",
                "yolo_confidence": 0.85,
                "brightness": 120.0,
                "motion_pct": 5.0,
                "frame_num": 999,
                "camera_id": "test-local",
                "trigger_type": "test",
            },
            verify=False,
        )
    elapsed = time.time() - start
    result = r.json()
    print(f"  Status:     {r.status_code}")
    print(f"  Neo4j:      {result.get('neo4j')}")
    print(f"  Entities:   {result.get('entities')}")
    print(f"  Server ms:  {result.get('processing_time_ms')}")
    print(f"  Total time: {elapsed:.1f}s")
    print()
    return result


def test_search(query):
    print("=" * 60)
    print(f"  TEST 3: Semantic Search — \"{query}\"")
    print("=" * 60)
    r = requests.get(f"{SERVER}/search", headers=HEADERS, params={"q": query, "limit": 5})
    result = r.json()
    print(f"  Query: {result.get('query')}")
    for i, obs in enumerate(result.get("results", []), 1):
        print(f"\n  --- Result {i} (score: {obs.get('score', 0):.3f}) ---")
        print(f"  Time:        {obs.get('timestamp')}")
        print(f"  Species:     {obs.get('species')}")
        print(f"  Trigger:     {obs.get('trigger')}")
        print(f"  Description: {obs.get('description', '')[:120]}...")
        print(f"  Image:       {obs.get('image_path')}")
    print()


def test_stats():
    print("=" * 60)
    print("  TEST 4: Graph Stats")
    print("=" * 60)
    r = requests.get(f"{SERVER}/stats", headers=HEADERS)
    result = r.json()
    print(f"  Neo4j connected: {result.get('neo4j')}")
    print(f"  Observations:    {result.get('total_obs')}")
    print(f"  Species:         {result.get('species')}")
    print(f"  Cameras:         {result.get('cameras')}")
    print()


def test_entities():
    print("=" * 60)
    print("  TEST 5: Entities in Graph")
    print("=" * 60)
    try:
        from neo4j_store import get_driver
        driver = get_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (obs:Observation)-[:CONTAINS]->(e:Entity)
                OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
                RETURN obs.id AS obs_id,
                       e.name AS entity,
                       e.type AS type,
                       e.attributes AS attributes,
                       d.text AS description
                ORDER BY obs.timestamp DESC
                LIMIT 20
            """)
            records = [dict(r) for r in result]
            if not records:
                print("  No entities found yet.")
            for r in records:
                print(f"  [{r['type']}] {r['entity']}")
                if r['attributes']:
                    print(f"    Attributes: {r['attributes']}")
                print(f"    From: {r['obs_id']}")
                print()
    except Exception as e:
        print(f"  Could not query Neo4j directly: {e}")
        print("  Use the /search endpoint instead.")
    print()


TEST_IMAGES = [
    {
        "path": os.path.join(IMAGES_DIR, "20260913T013040_wildlife_motion.jpg"),
        "yolo_class": "wildlife_motion",
        "trigger_type": "wildlife_motion",
    },
    {
        "path": os.path.join(IMAGES_DIR, "20260913T022848_person.jpg"),
        "yolo_class": "person",
        "trigger_type": "wildlife_person",
    },
    {
        "path": os.path.join(IMAGES_DIR, "20260913T011213_heartbeat.jpg"),
        "yolo_class": "heartbeat",
        "trigger_type": "heartbeat",
    },
]


def test_batch():
    print("=" * 60)
    print("  BATCH TEST: 3 images through full pipeline")
    print("=" * 60)
    for i, img in enumerate(TEST_IMAGES, 1):
        print(f"\n  --- Image {i}/3: {os.path.basename(img['path'])} ---")
        start = time.time()
        with open(img["path"], "rb") as f:
            r = requests.post(
                f"{SERVER}/analyze",
                headers=HEADERS,
                files={"frame": (os.path.basename(img["path"]), f, "image/jpeg")},
                data={
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "yolo_class": img["yolo_class"],
                    "yolo_confidence": 0.85,
                    "brightness": 120.0,
                    "motion_pct": 5.0,
                    "frame_num": 1000 + i,
                    "camera_id": "test-batch",
                    "trigger_type": img["trigger_type"],
                },
                verify=False,
            )
        elapsed = time.time() - start
        result = r.json()
        print(f"  Status:     {r.status_code}")
        print(f"  Neo4j:      {result.get('neo4j')}")
        print(f"  Entities:   {result.get('entities')}")
        print(f"  Time:       {elapsed:.1f}s")
        if result.get("description"):
            print(f"  VLM says:   {result['description']}")
        if result.get("extracted"):
            for e in result["extracted"].get("entities", []):
                print(f"    -> [{e['type']}] {e['name']} | {e.get('attributes', [])}")
            for rel in result["extracted"].get("relationships", []):
                print(f"    -> {rel['from']} --{rel['relation']}--> {rel['to']}")
    print()


if __name__ == "__main__":
    test_health()
    test_stats()

    print("\n>>> Running 3 images through VLM + Entity Extraction + Neo4j...\n")
    test_batch()

    print("Waiting 2s for indexing...\n")
    time.sleep(2)

    test_search("person walking")
    test_search("building outdoor")
    test_search("animal or bird")
    test_entities()

    print("=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)
