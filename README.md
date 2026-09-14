# AntVision — Multi-Device AI Activity Monitor

A distributed AI system that captures, understands, and queries real-world activity using computer vision, vision-language models, and a knowledge graph — running locally across 3 devices.

Started as an ant behavior observation system, evolved into a full-stack AI monitoring pipeline.

**[Live Showcase](docs/showcase.html)** | **[Demo Video](docs/chat_database.mp4)** | **[Project Deep Dive](docs/PROJECT_SHOWCASE.md)**

---

## System Architecture

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#1a1a2e', 'primaryTextColor': '#fff', 'primaryBorderColor': '#3498db', 'lineColor': '#e74c3c', 'secondaryColor': '#16213e', 'tertiaryColor': '#0f3460', 'fontSize': '14px'}}}%%

flowchart LR
    subgraph PI["RASPBERRY PI 4 — EDGE DEVICE"]
        direction TB
        CAM["Pi Camera Module 3\n640x480 @ 3s interval"]
        MOT["Motion Detection\nMOG2 + Solidity Filter"]
        YOLO["YOLOv8-nano ONNX\nPerson | Animal | Vehicle"]
        CAM --> MOT --> YOLO
    end

    subgraph WIN["WINDOWS RTX 3060 — GPU SERVER"]
        direction TB
        API["FastAPI Server\n/analyze  /search  /ask"]
        VLM["Qwen2.5-VL 7B\nVision-Language Model"]
        EMB["Sentence Embeddings\nall-MiniLM-L6-v2  384-dim"]
        NEO[("Neo4j Graph DB\n421 nodes  2242 rels")]
        DASH["Streamlit Dashboard\nChat UI + Images"]
        API --> VLM --> EMB --> NEO
        NEO --> DASH
    end

    subgraph MAC["MACBOOK M1 PRO — REASONING SERVER"]
        direction TB
        LLM["Qwen2.5 32B\nLarge Reasoning Model"]
        ENT["Entity Extraction\nperson — NEAR — shed"]
        CYP["Cypher Generation\nNatural Language to Query"]
        FIX["Self-Correction\nError — Fix — Retry 2x"]
        LLM --> ENT
        LLM --> CYP --> FIX
    end

    YOLO ==>|"WiFi HTTP POST\nframe + metadata"| API
    VLM -.->|"Ollama API\nWiFi"| LLM
    LLM -.->|"entities +\nrelationships"| NEO
    FIX -.->|"Cypher\nqueries"| NEO

    style PI fill:#0d2818,stroke:#2ecc71,stroke-width:3px,color:#2ecc71
    style WIN fill:#0d1b2a,stroke:#3498db,stroke-width:3px,color:#3498db
    style MAC fill:#1a0d2e,stroke:#9b59b6,stroke-width:3px,color:#9b59b6

    style CAM fill:#145214,stroke:#2ecc71,color:#fff
    style MOT fill:#145214,stroke:#2ecc71,color:#fff
    style YOLO fill:#145214,stroke:#2ecc71,color:#fff

    style API fill:#0a2a4a,stroke:#3498db,color:#fff
    style VLM fill:#0a2a4a,stroke:#3498db,color:#fff
    style EMB fill:#0a2a4a,stroke:#3498db,color:#fff
    style NEO fill:#0a2a4a,stroke:#3498db,color:#fff
    style DASH fill:#0a2a4a,stroke:#3498db,color:#fff

    style LLM fill:#2a0a3a,stroke:#9b59b6,color:#fff
    style ENT fill:#2a0a3a,stroke:#9b59b6,color:#fff
    style CYP fill:#2a0a3a,stroke:#9b59b6,color:#fff
    style FIX fill:#2a0a3a,stroke:#9b59b6,color:#fff
```

```
  Raspberry Pi 4              Windows RTX 3060             MacBook M1 Pro
  (Edge Device)               (GPU Server)                 (Reasoning Server)

  Pi Camera Module 3          FastAPI Server               Ollama
        |                          |                       qwen2.5:32b
   Motion Detection           Qwen2.5-VL 7B                    |
   (MOG2 + Solidity)         Scene Description            Entity Extraction
        |                          |                      Cypher Generation
   YOLOv8-nano ONNX          Sentence Embeddings          Query Answering
   Person/Animal Detection    (all-MiniLM-L6-v2)               |
        |                          |                            |
   WiFi HTTP POST -------->  /analyze endpoint                  |
                                   |                            |
                              Neo4j Graph DB  <-----------------+
                              (Vector Index)
                                   |
                            Streamlit Dashboard
                            (Chat UI + Images)
```

---

## Sample Captures

### Heartbeat — Scheduled Capture (11:04 AM)
<img src="vision_server/data/images/20260913T110412_heartbeat.jpg" width="700">

No activity detected. Background scene captured every 5 minutes as a baseline.

### Person Detected (4:34 PM)
<img src="vision_server/data/images/20260913T163413_person.jpg" width="700">

YOLOv8-nano detected a person near the building. Frame sent to VLM for scene description.

### Motion Triggered — Vehicle (9:24 AM)
<img src="vision_server/data/images/20260913T092450_unclassified_motion.jpg" width="700">

Motion detection triggered by vehicle movement. Solidity filter confirmed real motion.

### Person Detected (6:44 PM)
<img src="vision_server/data/images/20260913T184448_person.jpg" width="700">

Person detected near the carport. Entity extraction identified: person, building, greenery.

---

## What It Does

1. **Captures** — Pi Camera grabs frames every 3 seconds, runs motion detection with solidity filtering to ignore leaves and wind
2. **Classifies** — YOLOv8-nano (ONNX) runs on-device to identify people vs. animals vs. unclassified motion
3. **Describes** — Frames are sent to a GPU server running Qwen2.5-VL 7B, which generates rich natural language scene descriptions
4. **Extracts** — A 32B reasoning model extracts structured entities (person, vehicle, clothing, objects) and relationships
5. **Stores** — Everything goes into a Neo4j knowledge graph with vector embeddings for semantic search
6. **Queries** — Natural language query agent generates Cypher, self-corrects on errors, and explains results
7. **Displays** — Streamlit dashboard shows chat interface with inline images and graph stats

---

## Technical Highlights

### Edge AI on Raspberry Pi
- Background subtraction (MOG2) with solidity filtering to reject jagged leaf contours
- Multi-frame temporal confirmation for robust detection
- YOLOv8-nano exported to ONNX format for ARM compatibility (no PyTorch on Pi)
- Non-blocking frame upload via background threads — detection pipeline never waits for VLM

### Vision-Language Model Pipeline
- Qwen2.5-VL 7B on RTX 3060 GPU for scene understanding (~2-5s per frame)
- Rich descriptions: clothing colors, actions, spatial relationships, object counts
- 384-dimensional sentence embeddings (all-MiniLM-L6-v2) for semantic search

### Knowledge Graph (Neo4j)

![Neo4j Graph Visualization](docs/visualisation.svg)

- 6 node types: Camera, Observation, Species, TimeWindow, Description, Entity
- 7+ relationship types: CAPTURED, CLASSIFIED_AS, OCCURRED_DURING, DESCRIBED_BY, CONTAINS, NEAR, WEARING
- Vector index on Description embeddings for cosine similarity search
- Entity extraction creates queryable attributes (clothing, actions, positions)

### Agentic Query System
- 32B reasoning model generates Cypher queries from natural language
- Self-correction: if a query fails, the error is fed back to the model to fix
- Combines graph traversal with vector search for hybrid retrieval
- Streamlit chat UI with inline image display

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Camera | Raspberry Pi 4 + Pi Camera Module 3 |
| Motion Detection | OpenCV MOG2 + contour analysis |
| Object Detection | YOLOv8-nano (ONNX Runtime) |
| Vision-Language Model | Qwen2.5-VL 7B (Ollama) |
| Reasoning Model | Qwen2.5 32B (Ollama) |
| Text Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Graph Database | Neo4j with vector index |
| API Server | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Communication | HTTP REST + API key auth |

---

## Project Structure

```
antvision/
├── wildlife/                  # Pi edge pipeline
│   ├── main.py               # Main capture loop with YOLO + vision server
│   ├── smart_motion.py       # MOG2 + solidity + temporal confirmation
│   ├── classify.py           # YOLOv8 ONNX classifier
│   ├── daylight.py           # Brightness-based day/night detection
│   ├── storage_manager.py    # Disk management + cleanup
│   └── data_logger.py        # CSV event logging
├── vision_server/             # GPU server
│   ├── server.py             # FastAPI (VLM + Neo4j + search + query agent)
│   ├── vlm.py                # Qwen2.5-VL wrapper via Ollama
│   ├── embeddings.py         # Sentence-transformer embeddings
│   ├── entity_extractor.py   # LLM entity/relationship extraction
│   ├── neo4j_store.py        # Graph database operations + vector search
│   ├── query_agent.py        # Natural language -> Cypher -> results
│   └── config.py             # Multi-host Ollama configuration
├── dashboard/                 # Web UI
│   └── app.py                # Streamlit chat dashboard with images
├── edge/                      # Camera drivers + image upload
├── cloud/                     # AWS infrastructure (Terraform)
├── wildlife_vision.sh         # Pi start/stop/status script
└── docs/
    ├── WILDLIFE_VISION_SYSTEM.md
    └── ADVANCED_AI_SYSTEMS_GUIDE.md
```

---

## Running the System

### Pi (Edge)
```bash
./wildlife_vision.sh start     # run in background
./wildlife_vision.sh logs      # watch detections live
./wildlife_vision.sh stop      # stop
```

### Windows GPU Server
```bash
cd vision_server
python server.py
```

### Dashboard
```bash
streamlit run dashboard/app.py
```

### Example Queries
Ask in the Streamlit chat:
- "What happened at 11am today?"
- "Show me all people detected"
- "Yellow shirt person details"
- "Any animals spotted?"
- "What was near the building this morning?"

---

## Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Edge AI Inference | YOLOv8-nano ONNX on Raspberry Pi ARM |
| Vision-Language Models | Qwen2.5-VL 7B for scene description |
| Distributed AI Systems | 3 devices with role-based compute split |
| Knowledge Graphs | Neo4j with entities, relationships, vector index |
| RAG Pipeline | Embeddings + semantic search + LLM explanation |
| Agentic AI | LLM generates Cypher, self-corrects on failure |
| Real-time Processing | Camera -> detection -> VLM -> graph pipeline |
| Model Serving | Ollama as inference server across network |
| Quantization Tradeoffs | 3B vs 7B vs 32B, VRAM-aware deployment |
| Full-stack AI | Streamlit + FastAPI + Neo4j + edge devices |

---

## Evolution

```
Phase 1: Ant observation with Pi Camera + OpenCV
Phase 2: Wildlife monitoring with smart motion detection
Phase 3: Person detection with YOLOv8 on Pi
Phase 4: VLM scene descriptions on GPU server
Phase 5: Neo4j knowledge graph with vector search
Phase 6: Entity extraction and relationship mapping
Phase 7: Natural language query agent with self-correction
Phase 8: Streamlit dashboard with chat UI
```

---

## License

This project is intended as an experimental and educational project.
