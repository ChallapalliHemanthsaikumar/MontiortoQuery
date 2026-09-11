# HallGuard

**Privacy-First Person Activity Monitor powered by Edge AI**

HallGuard is a Raspberry Pi-based activity monitoring system for indoor spaces. It uses YOLOv8 for real-time person detection, automatically blurs all faces on-device before any data is saved or uploaded, tracks movement across user-defined activity zones, and streams structured events to AWS for dashboarding and analysis.

Built on top of the AntVision infrastructure — same AWS pipeline, completely new CV brain.

---

## What It Does

- Detects people in real-time using YOLOv8-nano on a Raspberry Pi
- Blurs all faces ON-DEVICE before any frame is saved, uploaded, or transmitted
- Tracks individuals with persistent IDs across frames
- Monitors user-defined activity zones (entrance, living room, kitchen entry, etc.)
- Logs zone enter/exit events with timestamps and dwell times
- Uploads blurred snapshots to S3 on detection events
- Streams structured events via MQTT to AWS IoT Core
- Stores metrics in DynamoDB for querying
- Displays everything on a real-time web dashboard

---

## System Architecture

```
                         EDGE (Raspberry Pi 4)
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  Pi Camera ──> Motion Filter ──> YOLOv8-nano         │
  │                  (skip YOLO      (person detect)     │
  │                   if static)         │               │
  │                                      v               │
  │                              ┌──────────────┐        │
  │                              │ FACE BLURRER  │        │
  │                              │ (MediaPipe)   │        │
  │                              │ on-device     │        │
  │                              └──────┬───────┘        │
  │                                     │                │
  │                         PRIVACY GATE                 │
  │                  (faces destroyed before this point)  │
  │                                     │                │
  │                    ┌────────────────┼────────────┐   │
  │                    v                v            v    │
  │              Centroid          Activity      Trajectory│
  │              Tracker           Zones        Recorder  │
  │              (person IDs)    (enter/exit)  (speed/path)│
  │                    │                │            │    │
  │                    v                v            v    │
  │              ┌─────────────────────────────┐         │
  │              │    Event Emitter + Analytics │         │
  │              └─────────┬───────────────────┘         │
  │                        │                             │
  │            ┌───────────┼────────────┐                │
  │            v                        v                │
  │     S3 Upload               MQTT Publish             │
  │   (blurred frames)        (JSON events)              │
  └────────┬───────────────────────┬─────────────────────┘
           │                       │
           v                       v
  ┌──────────────────────────────────────────────────────┐
  │                    AWS CLOUD                          │
  │                                                      │
  │  S3 Bucket ◄──────────────  IoT Core                 │
  │  (blurred captures         (MQTT endpoint)           │
  │   + session summaries)          │                    │
  │       │                         v                    │
  │       │              Lambda (event processor)        │
  │       │                         │                    │
  │       │                         v                    │
  │       │                    DynamoDB                   │
  │       │                   (events + metrics)         │
  │       │                         │                    │
  │       └────────┐    ┌───────────┘                    │
  │                v    v                                │
  │           Lambda (API handler)                       │
  │                  │                                   │
  │                  v                                   │
  │           API Gateway (HTTP)                         │
  └──────────────────┬───────────────────────────────────┘
                     │
                     v
  ┌──────────────────────────────────────────────────────┐
  │                 DASHBOARD (Browser)                   │
  │                                                      │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
  │  │ People   │ │ Zone     │ │ Zone     │ │ Avg     │ │
  │  │ Tracked  │ │ Activity │ │ Entries  │ │ Speed   │ │
  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │
  │                                                      │
  │  ┌─────────────────────┐  ┌────────────────────────┐ │
  │  │ Activity Event Log  │  │ Latest Blurred Capture │ │
  │  │ (color-coded)       │  │                        │ │
  │  └─────────────────────┘  └────────────────────────┘ │
  │                                                      │
  │  ┌──────────────────────────────────────────────────┐ │
  │  │         Blurred Capture Gallery                  │ │
  │  └──────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────┘
```

---

## Privacy Architecture

This is the core design principle. **No raw face data ever leaves the Raspberry Pi.**

```
  Camera captures raw frame (faces visible)
       │
       ▼
  YOLOv8-nano detects person bounding boxes
       │
       ▼
  MediaPipe detects face regions within each person box
       │
       ▼
  Gaussian blur (99x99 kernel) destroys face pixels in-place
       │
       ▼
  ╔══════════════════════════════════════════════╗
  ║           P R I V A C Y   G A T E           ║
  ║  Everything below only sees blurred frames   ║
  ╚══════════════════════════════════════════════╝
       │
       ├──> S3 Upload (blurred frames only)
       ├──> Local disk save (blurred only)
       ├──> Dashboard live view (blurred only)
       └──> Output video recording (blurred only)
```

**Guarantees:**
- Face blurring runs on-device before any frame is persisted or transmitted
- Raw face pixels exist only in a local variable that is immediately overwritten
- Person IDs are numeric track IDs (1, 2, 3...), not personal identities
- An automated test verifies: face detection on all output frames finds zero faces

---

## Edge Pipeline — How It Works

### Step-by-step data flow:

```
1. Pi Camera captures frame at 640x480 @ 15 FPS
                    │
2. Motion Detector (frame differencing)
   ├── No motion ──> skip to next frame (saves CPU)
   └── Motion detected ──> continue
                    │
3. YOLOv8-nano inference (~200-300ms on Pi 4)
   ├── No persons ──> skip (but may upload heartbeat frame)
   └── Persons found ──> list of bounding boxes + confidence
                    │
4. Face Blurrer (MediaPipe, ~10ms per face)
   └── Detects faces within person boxes, applies Gaussian blur
   └── Frame is now SAFE — all faces destroyed
                    │
5. Centroid Tracker
   └── Matches person boxes across frames using distance
   └── Assigns persistent IDs (Person #1, #2, ...)
   └── Handles disappearance/reappearance
                    │
6. Activity Zone Manager
   ├── Checks if each person centroid is inside a defined zone
   ├── Emits zone_enter event (with timestamp)
   ├── Emits zone_exit event (with dwell time)
   └── Tracks per-zone occupancy counts
                    │
7. Trajectory Recorder
   └── Records position history per person
   └── Computes speed (px/sec) and path length
   └── Draws movement trails on frame
                    │
8. Smart Capture Triggers
   ├── person_detected ──> upload blurred frame to S3
   ├── zone_event ──> upload blurred frame to S3
   ├── motion burst ──> upload blurred frame to S3
   └── hourly heartbeat ──> upload scene snapshot to S3
                    │
9. Event Publishing (MQTT to AWS IoT Core)
   ├── zone_enter / zone_exit events
   ├── metrics_snapshot (every N frames)
   └── session_start / session_end events
```

---

## Activity Zones

Users define rectangular zones in their hall by editing a JSON config file:

```json
// config/zones.json
{
  "zones": [
    {
      "name": "entrance",
      "x": 0, "y": 0, "w": 200, "h": 480,
      "color": [0, 255, 0]
    },
    {
      "name": "living_room",
      "x": 200, "y": 0, "w": 280, "h": 480,
      "color": [255, 165, 0]
    },
    {
      "name": "kitchen_entry",
      "x": 480, "y": 0, "w": 160, "h": 480,
      "color": [0, 0, 255]
    }
  ]
}
```

The system tracks:
- Who entered which zone and when
- How long they stayed (dwell time)
- How many people are in each zone at any moment
- Zone transition patterns over time

---

## Event Schema

Every event published to AWS follows this structure:

```json
{
  "device_id": "hallguard-pi01",
  "session_id": "s20260902_1430",
  "timestamp": "2026-09-02T14:30:15Z",
  "event_type": "zone_enter",
  "frame": 450,
  "person_id": 3,
  "zone": "living_room",
  "x": 320,
  "y": 240,
  "confidence": 0.85,
  "speed": 12.5
}
```

### Event Types

| Event | Description |
|-------|-------------|
| `zone_enter` | Person entered a defined activity zone |
| `zone_exit` | Person left a zone (includes `dwell_time_s`) |
| `person_detected` | New person first detected in frame |
| `metrics_snapshot` | Periodic summary (person count, zone occupancy, avg speed) |
| `session_start` | Monitoring session began |
| `session_end` | Monitoring session ended (includes full session summary) |

---

## Cloud Infrastructure (AWS)

All infrastructure is managed with Terraform:

| Service | Resource | Purpose |
|---------|----------|---------|
| **IoT Core** | Thing + MQTT topic | Receives events from Pi via mutual TLS |
| **IoT Credentials Provider** | Role alias | Lets Pi upload to S3 using its IoT certificate (no stored AWS keys) |
| **Lambda** | Event processor | Routes IoT events to DynamoDB/S3 |
| **Lambda** | API handler | Serves dashboard data via API Gateway |
| **DynamoDB** | Events table | Stores all events and metrics (partition: session_id, sort: timestamp) |
| **S3** | Data bucket | Stores blurred captures and session summaries |
| **API Gateway** | HTTP API | Public endpoints for dashboard |

---

## Dashboard

A single-page dark-themed web dashboard showing:

- **Metric Cards**: People Tracked, Zone Activity, Zone Entries, Avg Speed
- **Activity Event Log**: Color-coded events (green = zone enter, red = zone exit, blue = metrics)
- **Latest Capture**: Most recent blurred frame from the camera
- **Capture Gallery**: Grid of all blurred snapshots with trigger-type badges
- **Auto-refresh**: Polls API every 5 seconds for live updates

---

## Repository Structure

```
hallguard/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Dev dependencies (no Pi-specific)
├── .env.example                 # Environment variable template
├── run.sh                       # Pi launch script
│
├── config/
│   ├── zones.json               # Activity zone definitions
│   └── hallguard.json           # App configuration
│
├── edge/
│   ├── main.py                  # Entry point (routes to activity monitor)
│   ├── activity_monitor.py      # Main pipeline orchestrator
│   ├── analytics.py             # Session metrics & aggregation
│   ├── event_emitter.py         # Structured event creation
│   ├── image_uploader.py        # S3 upload with IoT credentials
│   ├── iot_publisher.py         # MQTT event publishing
│   │
│   ├── camera/
│   │   ├── capture.py           # OpenCV video capture wrapper
│   │   └── pi_camera.py         # Raspberry Pi camera wrapper
│   │
│   ├── detection/
│   │   ├── person_detector.py   # YOLOv8-nano person detection
│   │   ├── face_blurrer.py      # MediaPipe face detection + blur
│   │   ├── activity_zone.py     # Multi-zone enter/exit/dwell tracking
│   │   └── motion.py            # Frame-differencing motion detection
│   │
│   └── tracking/
│       ├── tracker.py           # Centroid-based multi-person tracker
│       └── trajectory.py        # Position history, speed, trail drawing
│
├── cloud/
│   ├── terraform/               # AWS infrastructure (IoT, Lambda, DynamoDB, S3, API GW)
│   ├── lambda/                  # Lambda function code
│   └── schemas/                 # Event JSON schemas
│
├── dashboard/
│   ├── index.html               # Single-page monitoring dashboard
│   └── server.py                # Local development server
│
├── tests/                       # Test suite
├── data/                        # Local captures and test data
└── docs/                        # Documentation
```

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Edge Hardware** | Raspberry Pi 4, Pi Camera Module |
| **Person Detection** | YOLOv8-nano (ultralytics) |
| **Face Blurring** | MediaPipe Face Detection |
| **Computer Vision** | OpenCV, NumPy |
| **Tracking** | Centroid-based multi-object tracker |
| **Cloud** | AWS IoT Core, Lambda, DynamoDB, S3, API Gateway |
| **Infrastructure** | Terraform |
| **Dashboard** | HTML/CSS/JS (single-page, dark theme) |
| **Communication** | MQTT (TLS) for events, HTTPS for image upload |

---

## Getting Started

### Prerequisites
- Raspberry Pi 4 with Pi Camera Module
- Python 3.9+
- AWS account with IoT Core, Lambda, DynamoDB, S3

### Install dependencies

```bash
# On Raspberry Pi
pip install -r requirements.txt

# On dev machine (Windows/Mac)
pip install -r requirements-dev.txt
```

### Configure zones

Edit `config/zones.json` to match your hall layout. Each zone is a named rectangle in pixel coordinates.

### Run locally (with test video)

```bash
python edge/main.py --mode person --input test_video.mp4 --show
```

### Run on Raspberry Pi (live camera)

```bash
# Set up environment variables
cp .env.example .env
# Edit .env with your AWS endpoints

# Run
bash run.sh
```

### Deploy AWS infrastructure

```bash
cd cloud/terraform
terraform init
terraform plan
terraform apply
```

### Open dashboard

Navigate to the API Gateway URL output by Terraform, or run the local dev server:

```bash
python dashboard/server.py
# Open http://localhost:5050
```

---

## Build Phases

| Phase | What | Status |
|-------|------|--------|
| 1 | Person Detection (YOLOv8-nano) | Pending |
| 2 | Face Blurring (MediaPipe) | Pending |
| 3 | Activity Zone Tracking | Pending |
| 4 | Pipeline Integration | Pending |
| 5 | Event System + Analytics | Pending |
| 6 | Cloud Infrastructure Updates | Pending |
| 7 | Dashboard Redesign | Pending |
| 8 | Testing & Validation | Pending |

---

## Privacy & Ethics

HallGuard is designed for monitoring your own private spaces (your hall, your home). It is **not** designed for surveillance of public spaces or non-consenting individuals.

- All faces are blurred before data leaves the device
- No facial recognition or identification is performed
- Person tracking uses anonymous numeric IDs only
- Users control what zones are monitored and what data is stored
- All data can be deleted from AWS at any time

---

## License

This project is intended for experimental and educational purposes.
