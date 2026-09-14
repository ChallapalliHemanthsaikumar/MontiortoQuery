# From Dumb Camera to Intelligent Knowledge System

**What if your security camera could answer questions?**

Security cameras record 24/7. Gigabytes of video. Nobody watches most of it. The data is wasted.

I turned a $75 Raspberry Pi camera into an AI system that understands what it sees, stores knowledge in a graph database, and answers natural language questions — all running locally across 3 devices. No cloud APIs, no subscriptions.

---

## The Transformation Pipeline

A single camera frame goes through 6 stages of AI processing, transforming raw pixels into queryable knowledge.

### Step 1 — Is anything happening?

**Motion Detection** on the Raspberry Pi using OpenCV background subtraction (MOG2).

Not every pixel change matters. The system uses **solidity filtering** to measure how compact a moving region is — a person is a solid blob, a leaf fluttering in the wind is a jagged shape. This single filter eliminates most false positives.

A **temporal confirmation** check requires motion in the same area across consecutive frames. Random noise triggers once. Real movement persists.

```
Raw Frame → Background Subtraction → Contour Detection → Solidity Filter → Temporal Check
                                                                              ↓
                                                                    "Yes, something real moved"
```

### Step 2 — What is it?

**YOLOv8-nano** runs directly on the Raspberry Pi to classify what moved.

The challenge: YOLO's default format (PyTorch) doesn't run on ARM processors. Solution: export the model to **ONNX format** on a Windows machine, copy the 12MB file to the Pi, and use ONNX Runtime instead of PyTorch. No GPU needed — inference runs on CPU in ~200ms per frame.

```
Motion Region → YOLOv8-nano (ONNX) → "person" (94% confidence)
```

The Pi now knows **what** it's looking at, not just that something moved.

### Step 3 — Describe what you see

The frame is sent over WiFi to a **Windows laptop with an RTX 3060 GPU** running **Qwen2.5-VL 7B** — a Vision-Language Model that can look at an image and write a description in natural language.

```
Camera Frame → HTTP POST → GPU Server → Qwen2.5-VL 7B
                                              ↓
"A person in a yellow and gray t-shirt standing outdoors in a
 residential area, holding a small object in their right hand.
 The background shows a house with a sloped roof and some greenery."
```

This happens in a **background thread** on the Pi — the detection pipeline never waits for the VLM. The Pi keeps capturing and classifying while the GPU server processes the previous frame.

The description is also converted into a **384-dimensional embedding vector** using sentence-transformers, enabling semantic similarity search later.

### Step 4 — Extract structured knowledge

A raw text description is useful, but not queryable. A second AI model — **Qwen2.5 32B** running on a MacBook M1 Pro — reads the description and extracts structured entities and relationships.

```
VLM Description → Qwen2.5 32B → Structured Entities

Entities:
  [person]   "person" — attributes: yellow shirt, dark shorts
  [location] "shed" — attributes: dark colored, sloped roof
  [object]   "small object" — attributes: held in right hand

Relationships:
  person → NEAR → shed
  person → HOLDING → small object
  person → WEARING → yellow shirt
```

Why a separate model? The VLM is optimized for seeing images. The reasoning model is optimized for understanding text and producing structured output. Splitting them plays to each model's strength.

### Step 5 — Store as a knowledge graph

Everything goes into **Neo4j**, a graph database. Each observation becomes a connected web of nodes and relationships.

```
(Camera: pi-wildlife)
    │
    └── CAPTURED ──► (Observation: 2026-09-13 11:05:39)
                         │
                         ├── DESCRIBED_BY ──► (Description: "A person in a yellow...")
                         │                        └── embedding: [0.134, 0.080, ...]
                         │
                         ├── CLASSIFIED_AS ──► (Species: person)
                         │
                         ├── OCCURRED_DURING ──► (TimeWindow: 2026-09-13_11)
                         │
                         ├── CONTAINS ──► (Entity: person | yellow shirt, dark shorts)
                         │                    └── NEAR ──► (Entity: shed)
                         │                    └── HOLDING ──► (Entity: small object)
                         │
                         └── image_path: "data/images/20260913T110539_person.jpg"
```

The graph also has a **vector index** on description embeddings. This enables semantic search — finding observations by meaning, not just keywords.

### Step 6 — Ask questions in plain English

A **natural language query agent** takes your question, generates a Neo4j Cypher query, runs it, and explains the results — showing the actual camera images.

```
You:   "Who was near the shed at 11am?"

Agent: [generates Cypher query]
       [runs against Neo4j]
       [if query fails → sends error back to model → self-corrects → retries]
       [explains results]

       "At 11:05 AM on September 13, a person wearing a yellow and gray
        t-shirt and dark shorts was detected standing near the shed,
        holding a small object in their right hand."

       [displays camera image]
```

The agent has **self-correction** — if the generated database query has a syntax error, the error message is fed back to the model, which fixes the query and retries. This makes the system robust without hardcoding query templates.

---

## The Setup — 3 Devices, All Local

```
┌──────────────────┐        WiFi         ┌────────────────────────┐        WiFi         ┌──────────────────┐
│                  │     HTTP POST       │                        │     Ollama API      │                  │
│  Raspberry Pi 4  │ ─────────────────► │  Windows Laptop        │ ─────────────────► │  MacBook M1 Pro  │
│                  │                     │  (RTX 3060 GPU)        │                     │  (32GB RAM)      │
│  Pi Camera       │                     │                        │                     │                  │
│  Motion Detection│                     │  Qwen2.5-VL 7B (GPU)  │                     │  Qwen2.5 32B     │
│  YOLOv8 ONNX     │                     │  Neo4j Graph Database  │                     │  Entity Extractor│
│                  │                     │  Sentence Embeddings   │                     │  Query Agent     │
│  ~$75 device     │                     │  FastAPI Server        │                     │  Self-Correction │
│                  │                     │  Streamlit Dashboard   │                     │                  │
└──────────────────┘                     └────────────────────────┘                     └──────────────────┘

     EDGE DEVICE                              GPU SERVER                              REASONING SERVER
  Detect & Classify                      Describe & Store                          Extract & Answer
```

**Why 3 devices?** Each has different hardware strengths:

| Device | Strength | Role |
|--------|----------|------|
| Raspberry Pi 4 | Always-on, low power, camera interface | Capture + on-device detection |
| Windows RTX 3060 | 6GB GPU VRAM, fast inference | VLM image understanding |
| MacBook M1 Pro | 32GB unified memory | Large reasoning model |

The VLM runs on the GPU because image processing benefits from parallel computation. The 32B reasoning model runs on the MacBook because it needs more RAM than VRAM. The Pi handles edge detection because it's connected to the camera and can filter out 95% of frames before sending anything over the network.

---

## Before vs. After

| | Before | After |
|---|--------|-------|
| **Data** | Raw video files | Structured knowledge graph |
| **Storage** | Gigabytes of video | Nodes, relationships, embeddings |
| **Access** | Scrub through footage manually | Ask questions in English |
| **Understanding** | Pixels | Descriptions, entities, relationships |
| **Search** | Timestamp only | Semantic ("find scenes like X") |
| **Cost** | Cloud API bills | $0 — runs entirely local |

---

## Key Technical Decisions

**ONNX export for Pi** — Instead of trying to install PyTorch on ARM (which crashed), export the YOLO model to ONNX format on a capable machine and copy the 12MB file. ONNX Runtime works on any architecture.

**Background threads for network calls** — The Pi's capture loop never blocks on network. Frames are sent to the GPU server in daemon threads. If the server is slow or down, the Pi keeps detecting.

**Solidity filtering** — A simple geometric measure (contour area / convex hull area) that separates solid objects (people, animals) from irregular shapes (leaves, shadows). One line of math eliminates most false positives.

**Split VLM and reasoning** — The vision model sees the image. The text model reads the description. Each model does what it's best at, and they run on different hardware optimized for their workload.

**Vector index for semantic search** — The sentence embedding of each description is stored in Neo4j's native vector index. Searching for "person near building" finds relevant observations even if those exact words don't appear in the description.

**Self-correcting query agent** — Instead of hardcoding query templates, the LLM generates database queries from natural language. When a query fails, the error is sent back to the model as context, and it generates a corrected query. This handles edge cases without manual coding.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Camera | Raspberry Pi 4 + Camera Module 3 | Low-cost, always-on, GPIO camera |
| Motion | OpenCV MOG2 + contour analysis | Lightweight, runs on Pi CPU |
| Detection | YOLOv8-nano ONNX (12MB) | Small enough for Pi ARM, accurate enough for person/animal |
| VLM | Qwen2.5-VL 7B via Ollama | Best open VLM that fits in 6GB VRAM |
| Reasoning | Qwen2.5 32B via Ollama | Strong structured output, fits in 32GB unified memory |
| Embeddings | all-MiniLM-L6-v2 | Fast, 384-dim, good semantic similarity |
| Graph DB | Neo4j + vector index | Native graph queries + vector search in one database |
| API | FastAPI | Async, fast, auto-docs |
| Dashboard | Streamlit | Chat UI with inline images, quick to build |

---

## Concepts This Project Covers

1. **Edge AI** — Running neural networks on resource-constrained devices (ONNX on ARM)
2. **Vision-Language Models** — Using multimodal AI to describe images in natural language
3. **Distributed AI Systems** — Splitting models across devices based on hardware capabilities
4. **Knowledge Graphs** — Storing real-world observations as connected entities and relationships
5. **Vector Search / RAG** — Embedding text for semantic retrieval, augmenting LLM answers with retrieved data
6. **Agentic AI** — LLM autonomously generates and corrects database queries
7. **Real-time Stream Processing** — Continuous camera → detection → VLM → storage pipeline
8. **Model Serving** — Ollama as a network-accessible inference server
9. **Quantization Tradeoffs** — Choosing model sizes (3B vs 7B vs 32B) based on available hardware
10. **Full-stack AI Application** — From hardware (Pi camera) to UI (Streamlit chat)

---

## What's Next

- **Re-identification** — Track the same person across multiple observations using appearance embeddings
- **Temporal patterns** — Detect routines ("someone walks by every morning at 8am")
- **Multi-camera** — Add more Pi cameras, correlate observations across locations
- **Alerting** — Push notifications when specific entities or unusual activity are detected
- **Fine-tuning** — Train a small model on this specific camera's scenes for better descriptions

---

*Built by Hemanth Challapalli*
*GitHub: github.com/ChallapalliHemanthsaikumar/antvision*
