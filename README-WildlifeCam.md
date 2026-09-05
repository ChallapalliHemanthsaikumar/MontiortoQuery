# WildlifeCam — Smart Backyard Wildlife Monitor

A Raspberry Pi-powered wildlife camera that captures and logs animal activity in your backyard. Uses intelligent motion filtering to distinguish real wildlife from wind, leaves, and sun changes — so you don't fill your SD card with false triggers.

## What It Does

- Captures frames from a Pi Camera at configurable intervals
- Detects wildlife motion using MOG2 background subtraction (adapts to gradual light/shadow changes)
- Filters out wind noise, scattered leaf motion, and sudden light shifts
- Auto-pauses at night (Pi Camera has no IR — dark frames are useless)
- Logs every event to CSV + JSON for data analysis
- Protects your SD card with disk space guards and hourly capture limits
- Uploads to AWS S3 when cloud is configured (reuses existing IoT infrastructure)
- Generates annotated frames with bounding boxes, brightness, and motion stats

## Architecture

```
Pi Camera (5s interval)
    |
    v
Daylight Check ──(too dark)──> Sleep 60s, retry
    |
    v (bright enough)
MOG2 Background Subtraction
    |
    v
Smart Motion Filter
  - Min contour area: 3000px (ignores small leaf flicker)
  - Aspect ratio filter: 0.2–5.0 (ignores thin shadow lines)
  - Max motion cap: 40% (ignores "everything moved" = wind gust)
  - Cooldown: 15s between triggers
    |
    v
Storage Guard ──(disk < 5GB or hourly limit hit)──> Skip
    |
    v
Save annotated frame (local + optional S3)
Log to CSV + JSON
```

## Hardware

| Component | Required? | Notes |
|-----------|-----------|-------|
| Raspberry Pi (3B+/4/5) | Yes | Any Pi with camera port |
| Pi Camera Module v2/v3 | Yes | Standard module — daytime only |
| Pi NoIR Camera + IR LEDs | Optional | For night vision (not required) |
| SD Card (32GB+) | Yes | Pipeline protects disk space |
| Power supply | Yes | 5V, needs to run 48+ hours |
| Weatherproof case | Recommended | If camera is outside |

### How to Check Your Pi Camera Type

```bash
# List detected cameras
libcamera-hello --list-cameras

# Physical check:
# - Standard camera lens = greenish/blueish (has IR filter, NO night vision)
# - NoIR camera lens = reddish/pinkish (no IR filter, HAS night vision)
```

## Quick Start

### 1. Setup on Pi

```bash
# SSH into your Pi
ssh pi@<your-pi-ip>

# Clone the project (or copy files)
cd ~/ANT\ project

# Install dependencies
pip install opencv-python-headless numpy

# Optional: for S3 uploads
pip install boto3 requests
```

### 2. Test Camera

```bash
# Verify camera works
libcamera-hello -t 5000

# Take a test shot
libcamera-jpeg -o test.jpg
```

### 3. Run WildlifeCam

```bash
# Basic run — live Pi Camera, local storage
python -m wildlife.main --live --experiment wildlife001

# With custom settings
python -m wildlife.main --live \
    --experiment wildlife001 \
    --capture-interval 5 \
    --heartbeat-minutes 30 \
    --min-area 3000 \
    --min-free-gb 5 \
    --max-per-hour 120

# With S3 upload (uses existing IoT certs)
python -m wildlife.main --live \
    --experiment wildlife001 \
    --s3-bucket antvision-data-dev \
    --credentials-endpoint <your-endpoint> \
    --role-alias antvision-device-alias

# Run in background (survives SSH disconnect)
nohup python -m wildlife.main --live --experiment wildlife001 > wildlife.log 2>&1 &
```

### 4. Run for 2 Days

```bash
# Start in a tmux or screen session so it survives SSH disconnect
tmux new -s wildlife
python -m wildlife.main --live --experiment sep2026_48hr
# Press Ctrl+B then D to detach

# Re-attach later to check progress
tmux attach -t wildlife
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--live` | off | Use Pi Camera (vs video file) |
| `--capture-interval` | 5s | Seconds between frame grabs |
| `--heartbeat-minutes` | 30 | Guaranteed capture interval even without motion |
| `--cooldown` | 15s | Min seconds between motion triggers |
| `--min-area` | 3000 | Min pixel area to count as motion (raise if too many false triggers) |
| `--min-motion-pct` | 0.5% | Min % of frame that must move |
| `--dark-threshold` | 30 | Brightness below this = night mode |
| `--min-free-gb` | 5.0 | Stop saving if disk free drops below this |
| `--max-per-hour` | 120 | Cap on saves per hour |
| `--max-local-images` | 2000 | Auto-delete oldest when exceeded |

## Storage Estimation

At 640x480, JPEG quality 85:

| Scenario | Images/Day | Storage/Day | 2-Day Total |
|----------|-----------|-------------|-------------|
| Quiet yard (heartbeats only) | ~48 | ~5 MB | ~10 MB |
| Moderate activity | ~200 | ~20 MB | ~40 MB |
| Busy (squirrels, birds) | ~500 | ~50 MB | ~100 MB |
| Worst case (max 120/hr, 14hr daylight) | ~1680 | ~170 MB | ~340 MB |

With 42GB free, you have room for **weeks** even at worst case.

## Data Output

### Folder Structure

```
data/wildlife/wildlife001/
├── wildlife001_events.csv       # Every capture event (pandas-ready)
├── wildlife001_events.json      # Same data in JSON
├── wildlife001_summary.json     # Run statistics
├── wildlife_motion/             # Motion-triggered captures
│   ├── 20260904T143022_f120.jpg
│   └── ...
└── heartbeat/                   # Periodic heartbeat captures
    ├── 20260904T140000_f1.jpg
    └── ...
```

### CSV Fields

| Field | Description |
|-------|-------------|
| `timestamp` | UTC ISO timestamp |
| `frame_num` | Sequential frame number |
| `trigger` | `wildlife_motion` or `heartbeat` |
| `brightness` | Mean pixel brightness (0-255) |
| `motion_pct` | Percent of frame with motion |
| `num_detections` | Number of motion regions |
| `boxes` | Bounding box coordinates JSON |
| `image_path` | Path to saved image |
| `free_gb` | Disk free space at capture time |

### Summary JSON

```json
{
  "experiment_id": "wildlife001",
  "start_time": "2026-09-04T14:00:00+00:00",
  "end_time": "2026-09-06T14:00:00+00:00",
  "total_frames": 24192,
  "images_saved": 347,
  "motion_triggers": 299,
  "heartbeat_captures": 48,
  "dark_skips": 8640,
  "storage_skips": 0
}
```

## How False Trigger Filtering Works

| Problem | How We Handle It |
|---------|-----------------|
| Leaves blowing | Small scattered contours — filtered by `min_contour_area` (3000px) |
| Wind shaking everything | Motion > 40% of frame — rejected as non-wildlife |
| Sun moving / shadows | MOG2 adapts background model over time |
| Sudden cloud shadow | MOG2 shadow detection (marks shadows separately) |
| Night darkness | Brightness check — auto-pauses when too dark |
| Rain on lens | Large uniform motion — filtered by max_motion_pct |

## Monitoring a Running Session

```bash
# Check the log
tail -f wildlife.log

# Check disk space
df -h

# Check how many images captured
ls data/wildlife/wildlife001/wildlife_motion/ | wc -l

# View latest summary
cat data/wildlife/wildlife001/wildlife001_summary.json | python -m json.tool

# Load events in Python
import pandas as pd
df = pd.read_csv("data/wildlife/wildlife001/wildlife001_events.csv")
print(df.describe())
print(df.groupby("trigger").count())
```

## Reused Infrastructure

This project builds on the existing AntVision edge pipeline:

| Component | Source | What It Does |
|-----------|--------|-------------|
| `edge/camera/pi_camera.py` | AntVision | Pi Camera capture via picamera2 |
| `edge/camera/capture.py` | AntVision | Video file capture for testing |
| `edge/image_uploader.py` | AntVision | S3 upload with IoT credential refresh |
| `wildlife/smart_motion.py` | New | MOG2-based wildlife motion detection |
| `wildlife/daylight.py` | New | Brightness-based daylight detection |
| `wildlife/storage_manager.py` | New | Disk space guard + capture budgets |
| `wildlife/data_logger.py` | New | CSV + JSON event logging |
| `wildlife/main.py` | New | Wildlife camera pipeline |

## Next Steps (After 2-Day Run)

- **Classify species**: Add a lightweight MobileNet/YOLOv8-nano classifier to label squirrel vs bird vs rabbit
- **Activity dashboard**: Plot motion events by hour, most active times, species frequency
- **Night vision**: Add Pi NoIR camera + IR LED board for 24/7 monitoring
- **Alert system**: Push notification when large animal (deer) detected
- **Time-lapse**: Stitch heartbeat images into a 48-hour timelapse video
