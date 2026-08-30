# AntVision

**An Edge Computer Vision System for Observing and Measuring Ant Behavior**

AntVision is an experimental computer-vision project built around a Raspberry Pi and camera. The goal is to observe ants around a controlled food source, detect and track their movement, convert visual observations into structured behavioral data, and eventually stream the results to AWS for real-time monitoring and long-term analysis.

---

## Project Goal

> **Place a controlled food source in an observation area and use computer vision to understand how ant activity changes over time.**

Instead of simply recording a video, AntVision turns video into measurable data.

The system will eventually answer questions such as:

- How long does it take for the first ant to discover the food?
- How many ants arrive over time?
- How does activity change after food is discovered?
- How long do ants remain near the food?
- What paths do ants follow?
- How does movement speed change?
- When does activity reach its peak?

---

## Architecture

```
                    ┌──────────────────┐
                    │   Pi Camera      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Raspberry Pi    │
                    │                  │
                    │ Computer Vision  │
                    │ Detection        │
                    │ Tracking         │
                    └────────┬─────────┘
                             │
                       Behavioral
                          Events
                             │
                             ▼
                    ┌──────────────────┐
                    │   AWS IoT Core   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Lambda       │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    ▼                  ▼
              ┌───────────┐      ┌───────────┐
              │ DynamoDB  │      │     S3    │
              │  Metrics  │      │  Images/  │
              │           │      │   Video   │
              └───────────┘      └───────────┘
                    │
                    ▼
              ┌──────────────┐
              │  Dashboard   │
              │ Live Metrics │
              └──────────────┘
```

> **Note:** The AWS portion is planned and will be implemented after the local computer-vision pipeline is working.

---

## Development Roadmap

### Phase 1 — Environment Setup

- [ ] Create GitHub repository
- [ ] Set up Python development environment
- [ ] Configure Raspberry Pi
- [ ] Verify Pi Camera
- [ ] Establish Raspberry Pi ↔ Wi-Fi connectivity
- [ ] Set up Git workflow

### Phase 2 — Camera Pipeline

- [ ] Capture frames from Pi Camera
- [ ] Save sample images
- [ ] Record test videos
- [ ] Determine optimal camera angle
- [ ] Determine suitable lighting
- [ ] Create a controlled observation area

### Phase 3 — Computer Vision

- [ ] Implement image preprocessing
- [ ] Implement motion detection
- [ ] Detect ants
- [ ] Determine ant coordinates
- [ ] Track individual ants
- [ ] Calculate trajectories
- [ ] Calculate movement speed
- [ ] Define food zone
- [ ] Detect food-zone entry/exit

### Phase 4 — Behavioral Analytics

Generate measurements such as:

- First discovery time
- Ant count
- Arrival rate
- Departure rate
- Food-zone activity
- Average speed
- Trajectory length
- Activity over time

### Phase 5 — AWS

- [ ] Define event schema
- [ ] Set up Terraform
- [ ] Create AWS IoT Core infrastructure
- [ ] Configure authentication
- [ ] Create DynamoDB table
- [ ] Create S3 storage
- [ ] Create Lambda processing
- [ ] Send Raspberry Pi events to AWS
- [ ] Store experiment data

### Phase 6 — Dashboard

- [ ] Live Pi status
- [ ] Current ant count
- [ ] Food-zone activity
- [ ] Arrival/departure rate
- [ ] Movement statistics
- [ ] Historical experiment data
- [ ] Trajectory visualization

### Phase 7 — Experiments

#### Experiment 001 — Food Discovery

**Question:** How does ant activity change after a food source is discovered?

Measurements:

- Time to first discovery
- Number of ants detected
- Arrival rate
- Food-zone activity
- Average movement speed
- Activity over time

Future experiments may investigate different food sources, quantities, distances, times of day, and activity decay after food removal.

---

## Repository Structure

```
antvision/
├── README.md
├── requirements.txt
├── .gitignore
│
├── edge/
│   ├── camera/         # Camera capture and configuration
│   ├── detection/      # Ant detection algorithms
│   ├── tracking/       # Multi-object tracking
│   └── main.py         # Edge pipeline entry point
│
├── cloud/
│   ├── terraform/      # Infrastructure as Code
│   ├── lambda/         # AWS Lambda functions
│   └── schemas/        # Event schemas
│
├── dashboard/          # Real-time monitoring UI
│
├── experiments/        # Experiment configurations and results
│
├── data/               # Local data (not committed)
│
├── tests/              # Test suite
│
└── docs/
    └── architecture/   # Architecture documentation
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Edge** | Raspberry Pi, Pi Camera, Python, OpenCV |
| **Computer Vision** | Image processing, motion detection, object detection, multi-object tracking |
| **Cloud** | AWS IoT Core, AWS Lambda, DynamoDB, S3 |
| **Infrastructure** | Terraform, AWS CLI |
| **Development** | Git, GitHub, Python venv, VS Code |

---

## Example Event

A detected behavioral event:

```json
{
  "device_id": "antvision-pi01",
  "experiment_id": "exp001",
  "timestamp": "2026-08-29T20:15:31Z",
  "ant_id": 17,
  "event_type": "food_zone_enter",
  "x": 421,
  "y": 238,
  "speed": 1.2,
  "zone": "food"
}
```

The exact schema will evolve as the computer-vision pipeline is developed.

---

## Experimental Philosophy

AntVision is designed around **repeatable experiments**. Each experiment records:

- Experiment ID
- Date and time
- Food type and quantity
- Observation duration
- Camera configuration
- Environmental conditions
- Computer vision version
- Results

The objective is to make observations reproducible and allow results from different experiments to be compared.

---

## Privacy & Data

The system is designed for a controlled observation environment. No people or personally identifiable information are intended to be captured. Video and image data remain limited to the experiment area.

---

## Current Status

**Early Development**

```
Development Environment → Raspberry Pi Camera → Camera Capture → Computer Vision Prototype
```

AWS infrastructure will be introduced after the local computer-vision pipeline is validated.

---

## License

This project is currently intended as an experimental and educational project.
