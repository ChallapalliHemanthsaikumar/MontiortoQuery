# ActivityVision

**A Privacy-First Edge AI Activity Monitor with Searchable Timeline**

ActivityVision is an advanced computer vision system built on a Raspberry Pi that detects human activity, automatically blurs faces for privacy, logs rich structured events, and provides a searchable timeline dashboard. Ask it "what happened Wednesday at 9:20 AM?" and get the answer instantly.

Built on top of the AntVision infrastructure — reusing the full AWS IoT pipeline for a real-world use case.

---

## What It Does

- Detects people in the camera view using YOLOv8-nano (optimized for Pi)
- Automatically blurs all faces before saving — privacy by design
- Tracks individuals across frames (entry, exit, direction, dwell time)
- Logs every event with rich metadata to DynamoDB
- Uploads blurred snapshots to S3, organized by date and time
- Dashboard with searchable timeline, activity graphs, and heatmaps
- Query by date, time range, zone, or activity type

---

## Example Queries

> "What happened on Wednesday between 9:00 and 10:00 AM?"
> "How many people passed through today?"
> "When was the busiest hour this week?"
> "Was there any movement after midnight?"
> "Show me all activity near the entrance"

---

## Architecture

```
                    +------------------+
                    |   Pi Camera      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Raspberry Pi    |
                    |                  |
                    |  YOLOv8-nano     |  <-- Person detection
                    |  Face Blurring   |  <-- Privacy layer
                    |  Activity Zones  |  <-- Zone monitoring
                    |  Tracking        |  <-- Multi-person tracking
                    +--------+---------+
                             |
                       Rich Activity
                          Events
                             |
                    +--------v---------+
                    |   AWS IoT Core   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |     Lambda       |
                    +--------+---------+
                             |
                    +--------+----------+
                    v                   v
              +-----------+       +-----------+
              | DynamoDB  |       |     S3    |
              | Activity  |       |  Blurred  |
              |   Events  |       |  Images   |
              +-----------+       +-----------+
                    |
                    v
              +----------------+
              |   Dashboard    |
              | Timeline Search|
              | Activity Graphs|
              |   Heatmaps    |
              +----------------+
```

---

## Privacy Design

Privacy is not an afterthought — it is the core design principle.

| Layer | Protection |
|-------|-----------|
| **Edge (Pi)** | Faces detected and blurred before any image leaves the device |
| **Storage (S3)** | Only blurred images are stored — original frames are never saved |
| **Events (DynamoDB)** | No identity data — only anonymous person IDs, positions, and actions |
| **Dashboard** | Shows activity patterns, not people — counts, graphs, timelines |

No facial recognition. No identity tracking. No re-identification across sessions.

---

## Event Schema

Every detected activity produces a rich structured event:

```json
{
  "device_id": "activityvision-pi01",
  "timestamp": "2026-08-30T09:20:15Z",
  "event_type": "activity_snapshot",
  "person_count": 2,
  "persons": [
    {
      "id": 1,
      "bbox": [120, 80, 200, 400],
      "zone": "entrance",
      "action": "walking",
      "direction": "east",
      "speed_px_s": 45.2,
      "dwell_time_s": 12.5
    },
    {
      "id": 2,
      "bbox": [350, 100, 180, 380],
      "zone": "hallway",
      "action": "standing",
      "direction": "stationary",
      "speed_px_s": 0.0,
      "dwell_time_s": 45.0
    }
  ],
  "scene_summary": "2 people: 1 walking east near entrance, 1 standing in hallway",
  "motion_level": 0.73,
  "frame_s3_key": "activity/2026-08-30/09-20-15_f1234.jpg"
}
```

---

## Development Roadmap

### Phase 1 — Person Detection + Privacy (Current)

- [ ] YOLOv8-nano person detection on Pi
- [ ] Face detection using Haar cascades / YOLO-face
- [ ] Gaussian blur on all detected faces
- [ ] Adapt centroid tracker for person-sized objects
- [ ] Rich event logging with person count, positions, zones
- [ ] S3 upload of blurred frames only
- [ ] Timeline-based dashboard with date/time search

### Phase 2 — Activity Zones + Analytics

- [ ] Define activity zones (entrance, hallway, room areas)
- [ ] Zone entry/exit tracking with dwell time
- [ ] Direction detection (which way are people moving)
- [ ] Activity graphs: people per hour, busiest times
- [ ] Daily/weekly activity summaries
- [ ] Heatmap overlay showing high-traffic areas

### Phase 3 — Action Recognition

- [ ] Pose estimation (MoveNet / MediaPipe on Pi)
- [ ] Action classification: walking, sitting, standing, running
- [ ] Unusual activity detection (anomaly detection)
- [ ] Alert system for configurable triggers (motion at 3 AM, etc.)
- [ ] SNS notifications to phone

### Phase 4 — LLM-Powered Intelligence

- [ ] Natural language query interface over activity data
- [ ] LLM-generated daily activity summaries
- [ ] Conversational dashboard: "What was unusual this week?"
- [ ] Pattern learning: "Tuesdays are typically quiet after 8 PM"
- [ ] Predictive analytics: expected vs actual activity

### Phase 5 — Edge AI Optimization

- [ ] ONNX / TFLite model optimization for Pi
- [ ] Quantization (INT8) for faster inference
- [ ] Multi-camera support with central aggregation
- [ ] On-device model fine-tuning
- [ ] Federated learning across multiple deployments

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Detection** | YOLOv8-nano, OpenCV DNN, Ultralytics |
| **Privacy** | OpenCV Haar Cascades, Gaussian Blur |
| **Tracking** | Centroid Tracker, Deep SORT (Phase 3) |
| **Pose/Action** | MoveNet, MediaPipe (Phase 3) |
| **Edge Runtime** | ONNX Runtime, TFLite (Phase 5) |
| **Edge Hardware** | Raspberry Pi 4/5, Pi Camera Module |
| **Cloud** | AWS IoT Core, Lambda, DynamoDB, S3, API Gateway |
| **Infrastructure** | Terraform |
| **Dashboard** | HTML/JS, Chart.js, Timeline.js |
| **LLM Integration** | Claude API (Phase 4) |

---

## Repository Structure

```
activityvision/
├── README.md                    # AntVision (original project)
├── README-ActivityMonitor.md    # This file
│
├── edge/
│   ├── main.py                  # Pipeline entry point (supports both modes)
│   ├── camera/                  # Camera capture (shared)
│   ├── detection/
│   │   ├── detector.py          # Ant detection (original)
│   │   ├── person_detector.py   # YOLO person detection (new)
│   │   ├── face_blurrer.py      # Face detection + blur (new)
│   │   ├── food_zone.py         # Food zone (original)
│   │   ├── activity_zone.py     # Activity zones (new)
│   │   └── motion.py            # Motion detection (shared)
│   ├── tracking/
│   │   ├── tracker.py           # Centroid tracker (shared)
│   │   └── trajectory.py        # Trajectory recording (shared)
│   ├── iot_publisher.py         # MQTT publishing (shared)
│   ├── image_uploader.py        # S3 upload (shared)
│   └── event_emitter.py         # Event creation (extended)
│
├── cloud/
│   ├── terraform/               # Infrastructure (extended)
│   ├── lambda/                  # Event processing (extended)
│   └── schemas/                 # Event schemas (extended)
│
├── dashboard/                   # Timeline UI (redesigned)
│
├── models/                      # YOLO weights, face models (new)
│
└── tests/                       # Test suite
```

---

## Quick Start

```bash
# On Raspberry Pi
# 1. Install dependencies
pip install ultralytics opencv-python numpy

# 2. Download YOLOv8-nano model (smallest, fastest)
yolo export model=yolov8n.pt format=onnx

# 3. Run activity monitor
python edge/main.py --live --mode activity \
    --iot-endpoint YOUR_ENDPOINT \
    --s3-bucket antvision-data-dev

# 4. View dashboard
open dashboard/index.html
```

---

## Comparison with AntVision

| Feature | AntVision | ActivityVision |
|---------|-----------|----------------|
| **Target** | Ants (tiny, unreliable) | People (always present) |
| **Detection** | Contour analysis | YOLOv8 deep learning |
| **Privacy** | N/A | Face blurring, no identity |
| **Events** | Zone enter/exit | Rich activity logs |
| **Query** | By experiment | By date, time, zone, action |
| **Dashboard** | Real-time metrics | Searchable timeline + graphs |
| **AI Level** | Classical CV | Deep learning → LLM |

---

## License

Experimental and educational project. Privacy-respecting by design.
