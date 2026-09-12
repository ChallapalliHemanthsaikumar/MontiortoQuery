# Wildlife Vision System — Architecture

A 3-device edge-to-cloud AI system that captures wildlife and room activity, describes scenes with a Vision Language Model, stores events in a graph database with semantic search, and answers natural language queries about what happened and when.

---

## System Architecture

```
    EDGE DEVICE                   GPU SERVER                    DATABASE / AGENT SERVER
  Raspberry Pi 4               Windows Laptop                    2nd Laptop
                                RTX 3060 6GB
 +-----------------+       +--------------------+        +-------------------------+
 | Pi Camera       |       | Ollama             |        | Neo4j Graph DB          |
 |   |             |       |   Qwen2.5-VL       |        |   Nodes + Edges         |
 |   v             |       |   (scene describe)  |        |   Vector Index          |
 | Motion Filter   |       |                    |        |                         |
 |   MOG2 + solid. | HTTP  | FastAPI            |  HTTP  | FastAPI                 |
 |   + multi-frame |------>|   /analyze          |------->|   /store                |
 |   |             | POST  |   receives frame    |  POST  |   stores event graph    |
 |   v             |       |   returns desc.     |        |                         |
 | YOLOv8n (ONNX)  |       | Embeddings         |        | LLM Query Agent         |
 |   detect + class|       |   sentence-transf.  |        |   natural language in   |
 |   |             |       |   384-dim vectors   |        |   Cypher queries out    |
 |   v             |       |                    |        |                         |
 | Send frame +    |       +--------------------+        | Web Dashboard           |
 | metadata        |                                     |   timeline + search     |
 +-----------------+                                     |   image viewer          |
                                                         +-------------------------+
```

---

## Device Specifications

| Device | Hardware | Role | Always On |
|--------|----------|------|-----------|
| Raspberry Pi 4 | ARM Cortex-A72, 4-8GB RAM, Pi Camera v3 | Edge capture, motion filtering, YOLO detection | Yes |
| Windows Laptop | i7-12700H, 16GB RAM, RTX 3060 6GB VRAM | VLM inference (Qwen2.5-VL), embedding generation | During capture sessions |
| 2nd Laptop | Any modern laptop, 8GB+ RAM | Neo4j, query agent, web dashboard | During capture + query sessions |

---

## Data Flow

```
[1] Pi Camera captures frame (every 5s)
         |
[2] Daylight check --- too dark ---> sleep 60s, retry
         |
[3] MOG2 background subtraction
         |
[4] Smart motion filter (solidity > 0.3, multi-frame confirm)
         |  no motion
         +-----------> discard
         |
[5] YOLOv8-nano ONNX inference
         |
         |  detection result
         v
[6] POST /analyze to GPU server
         |  payload: JPEG frame + JSON metadata
         |  (yolo_class, confidence, brightness, motion_pct, timestamp)
         |
         v
[7] GPU SERVER: Qwen2.5-VL describes the scene
         |  "A red-tailed hawk perched on the fence post,
         |   looking toward the bird feeder. Clear sky, late afternoon."
         |
[8] GPU SERVER: sentence-transformers encodes description to 384-dim vector
         |
         v
[9] POST /store to Database server
         |  payload: description, embedding, metadata, image bytes
         |
         v
[10] NEO4J: Create Event node, link to Entity/Species/TimeSlot nodes
         |  Store embedding in vector index
         |
         v
[11] USER QUERY via dashboard or chat:
         "What happened at 8:30am?"
         |
[12] LLM AGENT translates to Cypher + vector search
         |
[13] Neo4j returns matching events with descriptions and image paths
         |
[14] DASHBOARD displays timeline, images, and summary
```

---

## Neo4j Graph Schema

### Nodes

| Label | Properties | Description |
|-------|-----------|-------------|
| `Event` | `id`, `timestamp`, `image_path`, `vlm_description`, `yolo_class`, `yolo_confidence`, `brightness`, `motion_pct`, `trigger_type`, `embedding` (vector) | A single captured and analyzed frame |
| `Entity` | `id`, `type` (person/animal), `name` (optional), `first_seen`, `last_seen`, `appearance_count` | A tracked individual (person or animal) |
| `Species` | `name`, `category` (bird/mammal/reptile), `common_in_area` (bool) | Animal species reference |
| `TimeSlot` | `date`, `hour`, `period` (dawn/morning/afternoon/dusk/night) | Time bucketing for pattern queries |
| `Camera` | `id`, `location`, `position`, `resolution` | Camera source metadata |

### Edges

| Type | From | To | Properties | Description |
|------|------|----|-----------|-------------|
| `DETECTED_IN` | Entity | Event | `confidence`, `bbox` | An entity was seen in this event |
| `FOLLOWED_BY` | Event | Event | `gap_seconds` | Temporal sequence between events |
| `SAME_AS` | Entity | Entity | `similarity_score`, `method` | Re-identification link |
| `OCCURRED_DURING` | Event | TimeSlot | | Event-to-time relationship |
| `IS_SPECIES` | Entity | Species | `confidence` | Species classification |
| `CAPTURED_BY` | Event | Camera | | Which camera took the frame |

### Indexes

```cypher
-- Timestamp index for time-range queries
CREATE INDEX event_timestamp FOR (e:Event) ON (e.timestamp);

-- Vector index for semantic search over descriptions
CREATE VECTOR INDEX event_embedding FOR (e:Event) ON (e.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

-- Entity lookup
CREATE INDEX entity_type FOR (n:Entity) ON (n.type);

-- Species lookup
CREATE INDEX species_name FOR (s:Species) ON (s.name);
```

### Example Graph

```
(:Camera {id:"pi-room"})
       |
       | CAPTURED_BY
       v
(:Entity {type:"person", name:"Hemanth"})
       |
       | DETECTED_IN (confidence: 0.94)
       v
(:Event {timestamp:"2026-09-12T08:32:15", vlm_description:"A man in green shirt..."})
       |                          |
       | OCCURRED_DURING          | FOLLOWED_BY (gap: 300s)
       v                          v
(:TimeSlot {date:"2026-09-12",  (:Event {timestamp:"2026-09-12T08:37:15"...})
  hour:8, period:"morning"})           |
                                       | DETECTED_IN
                                       v
                                 (:Entity {type:"person", name:"Guest"})
```

---

## API Endpoints

### Pi to GPU Server

**POST /analyze**

Pi sends a captured frame with YOLO metadata for VLM analysis.

Request:
```
POST http://<gpu-server-ip>:8000/analyze
Content-Type: multipart/form-data

Fields:
  frame:            JPEG image bytes
  timestamp:        ISO 8601 UTC string
  yolo_class:       "person" | "bird" | "cat" | ...
  yolo_confidence:  float (0-1)
  brightness:       float (0-255)
  motion_pct:       float (0-100)
  frame_num:        int
  camera_id:        string
  trigger_type:     "wildlife_animal" | "wildlife_person" | "unclassified_motion"
```

Response:
```json
{
  "description": "A man in a green shirt walking toward the door...",
  "embedding": [0.023, -0.041, ...],
  "entities": [
    {"type": "person", "label": "adult male", "bbox": [120, 80, 340, 420]}
  ],
  "tags": ["person", "walking", "indoor", "door"],
  "processing_time_ms": 2340
}
```

### GPU Server to Database Server

**POST /store**

GPU server sends the analyzed event for graph storage.

Request:
```json
{
  "timestamp": "2026-09-12T08:32:15.000Z",
  "image_path": "wildlife_person/20260912T083215_f45.jpg",
  "image_bytes": "<base64 encoded JPEG>",
  "yolo_class": "person",
  "yolo_confidence": 0.94,
  "brightness": 124.0,
  "motion_pct": 7.3,
  "trigger_type": "wildlife_person",
  "camera_id": "pi-room",
  "vlm_description": "A man in a green shirt walking toward the door...",
  "embedding": [0.023, -0.041, ...],
  "entities": [
    {"type": "person", "label": "adult male", "bbox": [120, 80, 340, 420]}
  ],
  "tags": ["person", "walking", "indoor"]
}
```

Response:
```json
{
  "event_id": "evt_20260912T083215",
  "entities_created": 1,
  "edges_created": 3,
  "is_new_entity": false,
  "matched_entity": "entity_hemanth"
}
```

### Dashboard Query API

**POST /query**

Natural language query interface.

Request:
```json
{
  "question": "What happened at 8:30am this morning?"
}
```

Response:
```json
{
  "answer": "At 8:32am, a person in a green shirt walked toward the door. At 8:37am, a second person was detected near the desk area.",
  "events": [
    {
      "timestamp": "2026-09-12T08:32:15",
      "description": "A man in a green shirt walking toward the door...",
      "image_url": "/images/wildlife_person/20260912T083215_f45.jpg",
      "yolo_class": "person"
    },
    {
      "timestamp": "2026-09-12T08:37:15",
      "description": "A person sitting at the desk area...",
      "image_url": "/images/wildlife_person/20260912T083715_f51.jpg",
      "yolo_class": "person"
    }
  ],
  "cypher_query": "MATCH (e:Event) WHERE e.timestamp >= '2026-09-12T08:25:00' AND e.timestamp <= '2026-09-12T08:35:00' RETURN e ORDER BY e.timestamp"
}
```

**GET /timeline?date=2026-09-12&start=08:00&end=12:00**

Returns all events in a time range for dashboard rendering.

**GET /images/{path}**

Serves stored images.

**POST /search**

Semantic search over descriptions.

Request:
```json
{
  "query": "birds eating at feeder",
  "limit": 10
}
```

---

## Tech Stack

| Layer | Component | Technology | Version |
|-------|-----------|-----------|---------|
| Edge | Camera | picamera2 | system |
| Edge | Motion detection | OpenCV (MOG2) | 4.10 |
| Edge | Object detection | YOLOv8-nano ONNX | 8.x |
| Edge | HTTP client | requests | 2.x |
| GPU | VLM | Qwen2.5-VL via Ollama | 7B 4-bit |
| GPU | API server | FastAPI + uvicorn | 0.100+ |
| GPU | Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 2.x |
| DB | Graph database | Neo4j Community | 5.x |
| DB | API server | FastAPI + uvicorn | 0.100+ |
| DB | Query agent | LLM with tool use (Ollama or API) | -- |
| DB | Dashboard | FastAPI + Jinja2 + HTMX | -- |
| All | Language | Python | 3.11 |

---

## Project Structure

```
antvision/
|
|-- edge/                           # existing Pi camera + upload code
|   |-- camera/
|   |   |-- pi_camera.py
|   |   +-- capture.py
|   +-- image_uploader.py
|
|-- wildlife/                       # existing wildlife pipeline
|   |-- smart_motion.py             # MOG2 + solidity + multi-frame
|   |-- classify.py                 # YOLOv8 classifier
|   |-- daylight.py                 # brightness check
|   |-- storage_manager.py          # disk guards
|   |-- data_logger.py              # CSV + JSON logging
|   +-- main.py                     # capture pipeline entry point
|
|-- vision_server/                  # NEW: GPU server (RTX 3060 laptop)
|   |-- server.py                   # FastAPI app: /analyze endpoint
|   |-- vlm.py                      # Qwen2.5-VL wrapper via Ollama
|   |-- embeddings.py               # sentence-transformers encoding
|   |-- config.py                   # model settings, endpoints
|   +-- requirements.txt            # fastapi, uvicorn, ollama, sentence-transformers
|
|-- graph_server/                   # NEW: Database + Agent (2nd laptop)
|   |-- server.py                   # FastAPI app: /store, /query, /search, /timeline
|   |-- neo4j_client.py             # Neo4j driver, graph CRUD operations
|   |-- schema.py                   # node/edge definitions, index creation
|   |-- agent.py                    # LLM query agent with tool use
|   |-- tools/                      # agent tool definitions
|   |   |-- time_query.py           # query events by time range
|   |   |-- entity_query.py         # query by entity/species
|   |   |-- semantic_search.py      # vector similarity search
|   |   +-- pattern_query.py        # graph traversal for patterns
|   |-- dashboard/                  # web UI
|   |   |-- templates/
|   |   |   |-- index.html          # main dashboard page
|   |   |   |-- timeline.html       # event timeline view
|   |   |   +-- query.html          # natural language query page
|   |   +-- static/
|   |       |-- style.css
|   |       +-- app.js
|   |-- image_store/                # stored images (served by FastAPI)
|   |-- config.py                   # Neo4j credentials, model settings
|   +-- requirements.txt            # fastapi, neo4j, ollama, sentence-transformers
|
|-- docs/
|   |-- WILDLIFE_VISION_SYSTEM.md   # this document
|   |-- ADVANCED_AI_SYSTEMS_GUIDE.md
|   +-- DRONE_ADVANCE_AI.md
|
+-- docker-compose.yml              # optional: Neo4j + services containerized
```

---

## Build Phases

### Phase 1 -- Ollama + Qwen on RTX 3060 (Week 1)

Goal: VLM running locally, can describe wildlife images.

```bash
# On GPU server (Windows laptop)
# Install Ollama from https://ollama.com
ollama pull qwen2.5vl:7b

# Test with a wildlife image
ollama run qwen2.5vl:7b "Describe what you see in this image" --image wildlife_frame.jpg
```

Validation: Feed 10 captured wildlife images, verify descriptions are accurate and detailed.

### Phase 2 -- FastAPI Bridge: Pi to GPU Server (Week 1-2)

Goal: Pi sends frames over WiFi, GPU server returns VLM descriptions.

Build:
- `vision_server/server.py` -- FastAPI with `/analyze` endpoint
- `vision_server/vlm.py` -- Ollama Python client calling Qwen2.5-VL
- `vision_server/embeddings.py` -- sentence-transformers for 384-dim vectors
- Update `wildlife/main.py` on Pi to POST frames when triggered

```bash
# On GPU server
cd vision_server
pip install fastapi uvicorn ollama sentence-transformers
uvicorn server:app --host 0.0.0.0 --port 8000

# On Pi (test)
curl -X POST http://<gpu-ip>:8000/analyze \
  -F "frame=@test_image.jpg" \
  -F "yolo_class=person" \
  -F "timestamp=2026-09-12T08:32:15Z"
```

Validation: Pi sends a frame, receives description + embedding in under 5 seconds.

### Phase 3 -- Neo4j Setup + Graph Schema (Week 2-3)

Goal: Graph database running with correct schema and indexes.

```bash
# On database server (2nd laptop)
# Install Neo4j Community Edition
# Or run via Docker:
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/wildlife123 \
  -v neo4j_data:/data \
  neo4j:5

# Create schema
python graph_server/schema.py
```

Build:
- `graph_server/neo4j_client.py` -- connection, CRUD for all node/edge types
- `graph_server/schema.py` -- create constraints, indexes, vector index

Validation: Manually insert 5 test events, query by time and by similarity.

### Phase 4 -- VLM to Neo4j Pipeline (Week 3)

Goal: End-to-end flow from Pi capture to graph storage.

Build:
- `graph_server/server.py` -- FastAPI with `/store` endpoint
- Update `vision_server/server.py` to forward results to graph server
- Image storage on database server filesystem

Validation: Walk in front of camera. Event appears in Neo4j browser (localhost:7474) with description, image path, and correct time slot linkage.

### Phase 5 -- Query Agent (Week 3-4)

Goal: Ask natural language questions, get answers with images.

Build:
- `graph_server/agent.py` -- LLM with Cypher generation and tool use
- `graph_server/tools/` -- time query, entity query, semantic search, pattern query
- `/query` endpoint that accepts natural language

Agent tools:
```python
tools = [
    {
        "name": "query_by_time",
        "description": "Find events in a time range",
        "parameters": {"start": "ISO timestamp", "end": "ISO timestamp"}
    },
    {
        "name": "query_by_entity",
        "description": "Find events involving a specific entity or species",
        "parameters": {"entity_type": "person|bird|cat|...", "name": "optional"}
    },
    {
        "name": "semantic_search",
        "description": "Search events by description similarity",
        "parameters": {"query": "natural language description", "limit": "int"}
    },
    {
        "name": "count_events",
        "description": "Count events matching criteria",
        "parameters": {"entity_type": "optional", "date": "optional"}
    }
]
```

Validation: Ask "What happened this morning?" and receive a correct summary with images.

### Phase 6 -- Web Dashboard (Week 4-5)

Goal: Browser-based UI for timeline, search, and image viewing.

Pages:
- `/` -- overview: today's stats, recent events, species counts
- `/timeline` -- chronological event feed with thumbnails and descriptions
- `/search` -- semantic search bar with result cards
- `/chat` -- natural language query interface
- `/entity/{id}` -- entity detail: all appearances, timeline, linked events

Build:
- `graph_server/dashboard/` -- Jinja2 templates + HTMX for interactivity
- FastAPI serves both API and dashboard on the same port

Validation: Open dashboard in browser, see today's events with images and descriptions. Ask a question in chat and get a useful answer.

### Phase 7 -- Re-identification + Patterns (Week 5-6)

Goal: Track individuals across visits, discover behavioral patterns.

Build:
- Visual similarity matching: compare VLM descriptions + YOLO crops across events
- Create `SAME_AS` edges between Entity nodes when match confidence > threshold
- Pattern queries: "When does this hawk usually appear?" via graph traversal
- Activity heatmap: events-by-hour aggregation

Validation: Same person appears in multiple events over different days, system links them and can answer "How many times has this person visited this week?"

---

## Running the System

### Start order

```bash
# 1. Database server (2nd laptop)
docker start neo4j                              # or start Neo4j service
cd graph_server && uvicorn server:app --host 0.0.0.0 --port 8001

# 2. GPU server (RTX 3060 laptop)
ollama serve                                     # start Ollama daemon
cd vision_server && uvicorn server:app --host 0.0.0.0 --port 8000

# 3. Edge device (Pi)
python -m wildlife.main --live \
  --experiment session_001 \
  --rotate 180 \
  --use-classifier \
  --classifier-model yolov8n.onnx \
  --vision-server http://<gpu-ip>:8000
```

### Network

All three devices must be on the same local network. Use static IPs or mDNS names.

```
Pi:           192.168.1.100   (or hemanth.local)
GPU Server:   192.168.1.101   (or kavitha-pc.local)
DB Server:    192.168.1.102   (or db-laptop.local)
```

### Monitoring

```bash
# Check GPU server load
nvidia-smi -l 1

# Check Neo4j
# Open http://<db-ip>:7474 in browser

# Check event count
curl http://<db-ip>:8001/stats

# Open dashboard
# http://<db-ip>:8001/
```

---

## Example Queries and Expected Cypher

**"What happened at 8:30am?"**
```cypher
MATCH (e:Event)
WHERE e.timestamp >= datetime('2026-09-12T08:25:00Z')
  AND e.timestamp <= datetime('2026-09-12T08:35:00Z')
RETURN e.timestamp, e.vlm_description, e.image_path
ORDER BY e.timestamp
```

**"Show me all birds this week"**
```cypher
MATCH (ent:Entity)-[:IS_SPECIES]->(s:Species {category: 'bird'})
MATCH (ent)-[:DETECTED_IN]->(e:Event)
WHERE e.timestamp >= datetime('2026-09-08T00:00:00Z')
RETURN e.timestamp, e.vlm_description, e.image_path, s.name
ORDER BY e.timestamp
```

**"How many people visited today?"**
```cypher
MATCH (ent:Entity {type: 'person'})-[:DETECTED_IN]->(e:Event)
WHERE e.timestamp >= datetime('2026-09-12T00:00:00Z')
RETURN count(DISTINCT ent) AS unique_people, count(e) AS total_appearances
```

**"Has this hawk been here before?"**
```cypher
MATCH (ent:Entity)-[:IS_SPECIES]->(s:Species {name: 'hawk'})
MATCH (ent)-[:DETECTED_IN]->(e:Event)
RETURN ent.id, ent.first_seen, ent.last_seen, ent.appearance_count,
       collect(e.timestamp) AS visit_times
```

**"What animals appear together?"**
```cypher
MATCH (e:Event)<-[:DETECTED_IN]-(ent1:Entity)
MATCH (e)<-[:DETECTED_IN]-(ent2:Entity)
WHERE ent1 <> ent2
MATCH (ent1)-[:IS_SPECIES]->(s1:Species)
MATCH (ent2)-[:IS_SPECIES]->(s2:Species)
RETURN s1.name, s2.name, count(e) AS co_occurrences
ORDER BY co_occurrences DESC
```

**"Show me the sequence of events this morning"**
```cypher
MATCH (e1:Event)-[:FOLLOWED_BY*1..20]->(e2:Event)
MATCH (e1)-[:OCCURRED_DURING]->(t:TimeSlot {date: '2026-09-12', period: 'morning'})
WITH collect(DISTINCT e1) + collect(DISTINCT e2) AS all_events
UNWIND all_events AS e
RETURN DISTINCT e.timestamp, e.vlm_description, e.image_path
ORDER BY e.timestamp
```

**Semantic search: "birds eating at feeder"**
```cypher
CALL db.index.vector.queryNodes('event_embedding', 10, $query_embedding)
YIELD node, score
RETURN node.timestamp, node.vlm_description, node.image_path, score
ORDER BY score DESC
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Pi capture-to-send | < 500ms | YOLO ONNX inference + HTTP POST |
| VLM description | < 5s | Qwen2.5-VL 7B 4-bit on RTX 3060 |
| Embedding generation | < 100ms | all-MiniLM-L6-v2 on GPU |
| Graph storage | < 200ms | Neo4j write + index update |
| End-to-end latency | < 6s | From capture to stored in graph |
| Query response | < 3s | Natural language to answer |
| Dashboard load | < 1s | Timeline page with 50 events |
| Storage per event | ~150KB | Image (~100KB) + metadata (~50KB) |
| Events per day (busy) | ~500 | With motion filtering active |
