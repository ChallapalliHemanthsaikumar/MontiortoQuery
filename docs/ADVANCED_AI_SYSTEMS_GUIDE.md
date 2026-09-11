# Advanced AI Systems Guide — From Edge to Cloud

Your current stack: Pi Camera → OpenCV motion → YOLO detection → local storage → AWS S3.
Where you're going: Pi Camera → YOLO → GPU Server (VLM + LLM + Agent) → Smart Actions → Cloud.

This document maps every concept you need, why it matters, and how it connects to your real project.

---

## Part 1: Making WildlifeCam Smarter with VLM + LLM

### The Problem Right Now

YOLO gives you: `bird 94%`, `cat 87%`, `dog 91%`

YOLO does NOT give you:
- "Is that a hawk or a robin?" (fine-grained species)
- "The deer is eating from the garden" (scene understanding)
- "Two squirrels are chasing each other up the oak tree" (behavior)
- "This is the same cat from yesterday" (re-identification)
- "Alert: a coyote is near the bird feeder" (reasoning + action)

### The Solution: Three-Stage Pipeline

```
STAGE 1: Edge (Pi)                    STAGE 2: GPU Server (Laptop)           STAGE 3: Agent
─────────────────                     ──────────────────────────             ─────────────
Pi Camera                             Receive frame + YOLO results          Receive VLM description
    │                                      │                                    │
Motion Filter                         Run VLM on the frame                  Compare with history
(solidity + multi-frame)              "Describe what you see"               "Is this new?"
    │                                      │                                    │
YOLO-nano                             Get rich description:                 Decide action:
"bird detected 94%"                   "A red-tailed hawk perched            - Log to database
    │                                  on fence post, looking at             - Send notification
Send frame + metadata ──────────────> bird feeder area"                     - Update species count
over WiFi to GPU server                    │                                    │
                                      Run LLM reasoning:                    Execute:
                                      "Hawk near feeder = threat            - Push notification
                                       to songbirds"                        - Store in RAG
                                           │                                - Trigger action
                                      Send to Agent ────────────────────────>
```

### How to Connect Pi to GPU Laptop

**Option A: REST API (simplest)**
```
Pi sends HTTP POST with image → Laptop runs Flask/FastAPI server → Returns results
```
The laptop runs a server like:
- FastAPI endpoint that accepts image bytes
- Runs YOLO + VLM + LLM
- Returns JSON with detections + description + reasoning

**Option B: MQTT/Message Queue**
```
Pi publishes to MQTT topic → Laptop subscribes → Processes → Publishes result
```
Better for multiple cameras, decoupled, survives network hiccups.

**Option C: gRPC (fastest)**
```
Pi sends frame via gRPC → Laptop processes → Streams results back
```
Lower latency than REST, good for real-time video.

**For your project: start with Option A (FastAPI).** It's the simplest and you can upgrade later.

### What VLM to Run on RTX 3060

Your RTX 3060 has 12GB VRAM. Here's what fits:

| Model | Size | VRAM | What It Does |
|-------|------|------|-------------|
| YOLO v8-nano | 6MB | ~0.5GB | Object detection (already have) |
| LLaVA 7B (4-bit) | ~4GB | ~5GB | Vision + Language — describe images |
| Qwen2.5-VL 7B (4-bit) | ~4GB | ~5GB | Vision + Language — newer, better |
| Phi-3.5-Vision (4-bit) | ~2.5GB | ~4GB | Smaller but fast VLM |
| Llama 3.1 8B (4-bit) | ~4.5GB | ~5.5GB | Text-only LLM for reasoning |

**Can you run YOLO + VLM + LLM simultaneously on 12GB?**
- YOLO (0.5GB) + Phi-3.5-Vision (4GB) + Llama-8B (5.5GB) = ~10GB. Tight but possible.
- YOLO (0.5GB) + Qwen2.5-VL 7B (5GB) = ~5.5GB. Use the VLM for both vision AND reasoning. Simpler.

**Recommendation: Start with one VLM (Qwen2.5-VL or LLaVA) that handles both vision and reasoning.** Split into VLM + separate LLM later when you understand the tradeoffs.

---

## Part 2: The Concepts You Need — Connected to Your Project

### 2.1 Model Architecture Fundamentals

#### Transformers — The Foundation of Everything

Every model you'll use (YOLO's backbone, LLMs, VLMs) is built on transformers.

**What you need to know:**
- **Self-attention**: Each token looks at every other token to decide what's important. Cost: O(n^2) where n = sequence length. This is why long contexts are expensive.
- **KV-cache**: During generation, the model stores key-value pairs from previous tokens so it doesn't recompute them. This is why VRAM usage grows as you generate more text.
- **Why this matters for your project**: When your VLM describes a wildlife image, the prompt (system instructions + image tokens + question) creates a KV-cache. Longer prompts = more VRAM = fewer concurrent requests your 3060 can handle.

#### Vision Transformers (ViT) — How Models See Images

YOLO uses a CNN backbone. VLMs use Vision Transformers.

**How a VLM processes your wildlife frame:**
1. Image is split into patches (e.g., 14x14 pixel patches)
2. Each patch becomes a token (just like a word in text)
3. A 640x480 image at 14x14 patches = ~1,500 image tokens
4. These tokens go into the transformer alongside text tokens
5. The model attends to both image and text tokens together

**Why this matters**: A single image adds ~1,500 tokens to your context. If you send 5 frames for temporal analysis, that's ~7,500 tokens just for images. Your 3060 will feel this.

#### Mixture of Experts (MoE) — Why Some Models Are Fast Despite Being Huge

**What MoE is:**
- A model has many "expert" sub-networks (e.g., 8 experts)
- For each token, a router picks only 2 experts to activate
- Total parameters: 47B. Active parameters per token: 13B
- Result: model quality of a 47B model, speed closer to a 13B model

**Example: Mixtral 8x7B**
- 8 experts, each ~7B parameters
- Only 2 active per token
- Total: ~47B params, Active: ~13B params
- Fits in ~26GB VRAM (4-bit) — too big for your 3060 alone
- But the concept matters because it's how frontier models scale

**Why this matters for your project:**
- MoE models give you better quality per compute dollar
- Mixtral 8x7B (4-bit) won't fit on your 3060, but smaller MoE models will emerge
- Understanding MoE helps you pick the right model: dense 7B vs MoE 14B-active-from-47B — different tradeoffs
- If you move to cloud GPUs later, MoE models give the best quality/cost ratio

#### Quantization — Running Big Models on Small GPUs

**What it is:** Reducing the precision of model weights to use less memory.

```
Full precision (FP32):  32 bits per weight  →  7B model = ~28GB
Half precision (FP16):  16 bits per weight  →  7B model = ~14GB
8-bit (INT8):           8 bits per weight   →  7B model = ~7GB
4-bit (INT4/GPTQ/AWQ):  4 bits per weight  →  7B model = ~3.5GB
```

**Quality loss at each level:**
- FP16: virtually none
- INT8: minimal (<1% quality drop on most benchmarks)
- INT4: noticeable on complex reasoning, fine for classification/description
- 2-bit: significant quality loss, only for experiments

**Why this matters**: Your entire multi-model strategy depends on quantization. Without 4-bit quantization, a 7B VLM wouldn't fit alongside YOLO on your 3060.

**Tools to know:**
- `bitsandbytes` — easy 4/8-bit quantization in Python
- `GPTQ` — post-training quantization, very popular
- `AWQ` — activation-aware quantization, slightly better quality than GPTQ
- `llama.cpp / GGUF` — quantized models that run on CPU+GPU hybrid

### 2.2 Inference Optimization

#### KV-Cache — Why Your LLM Gets Slower Over Time

During text generation:
1. First token: process entire prompt. Slow (prefill phase).
2. Each next token: only process the new token, but look up all previous keys/values.
3. The KV-cache grows with every token generated.
4. VRAM usage: `2 * num_layers * hidden_dim * sequence_length * bytes_per_value`

**For a 7B model at 4-bit, generating 500 tokens:**
- KV-cache alone: ~200-400MB
- This is ON TOP of the model weights

**Why this matters**: If your wildlife agent generates long descriptions for every frame, the KV-cache grows, VRAM fills up, and eventually you hit OOM or start swapping to CPU (10x slower).

**What to do about it:**
- Keep prompts short and structured
- Use `max_tokens` limits
- Clear KV-cache between frames (don't accumulate context)
- Use sliding window attention models (Mistral) that cap KV-cache size

#### Speculative Decoding — Generating Tokens Faster

**The problem:** LLMs generate one token at a time. Each token requires a full forward pass through the model. A 7B model might do 30 tokens/sec on your 3060.

**The trick:**
1. A small "draft" model (e.g., 0.5B) quickly proposes 5 tokens
2. The large model verifies all 5 in ONE forward pass (parallel verification)
3. If 4 out of 5 are correct, you just generated 4 tokens in the time of ~1.5 forward passes

**Speed improvement:** 1.5-2.5x faster generation, same output quality.

**Why this matters for your project:** Your wildlife descriptions need to be fast (animal might leave). Speculative decoding on the GPU server means faster VLM responses without buying a bigger GPU.

**How to try it:**
- vLLM supports speculative decoding out of the box
- Pair a 7B model with a 0.5B draft model
- Measure: tokens/sec with and without, VRAM usage for both

#### Batching — Processing Multiple Frames at Once

**Static batching:** Collect N frames, process all at once. Simple but wasteful — fast frames wait for slow ones.

**Continuous batching (what vLLM does):** New requests join the batch as soon as a slot opens. No waiting.

**Why this matters:** If you add a second camera (front yard + back yard), continuous batching lets both cameras share the GPU efficiently. One VLM serving both cameras, not two separate VLM instances.

#### TensorRT — Making YOLO Scream Fast

YOLO on PyTorch on your 3060: ~30 FPS at 640x640.
YOLO on TensorRT on your 3060: ~80-120 FPS at 640x640.

**What TensorRT does:**
- Fuses operations (Conv + BatchNorm + ReLU → one kernel)
- Optimizes for your specific GPU architecture
- Uses FP16/INT8 automatically where safe
- Pre-allocates memory for the exact model graph

**How to convert:**
```bash
yolo export model=yolov8n.pt format=engine  # creates yolov8n.engine
```

**Why this matters:** If you're running YOLO + VLM + LLM on one GPU, making YOLO 3-4x faster means it uses the GPU for less time per frame, leaving more time for VLM/LLM.

### 2.3 Serving Infrastructure

#### vLLM — Production LLM Serving

vLLM is a high-performance inference server. It handles:
- Continuous batching (multiple requests share GPU)
- PagedAttention (efficient KV-cache memory management)
- Speculative decoding
- Multi-model serving
- OpenAI-compatible API

**Why this matters:** Instead of loading a model in Python and calling it directly (slow, no batching, wasteful), vLLM gives you a proper server that your Pi can call via HTTP.

```
Pi → HTTP POST /v1/chat/completions → vLLM on laptop → Response
```

Your Pi code doesn't even know it's talking to a local model — it looks exactly like calling the OpenAI API.

#### Ollama — Easiest Way to Start

If vLLM feels complex, Ollama is simpler:
```bash
ollama run llava    # download and run LLaVA VLM
ollama run llama3.1 # download and run Llama 3.1
```

It gives you a local API at `localhost:11434`. Your Pi can call it over the network.

**Start with Ollama to prototype, move to vLLM when you need performance.**

### 2.4 Agentic AI — Making the System Autonomous

You already know agents. Here's how they connect to your physical system.

#### The Wildlife Agent Architecture

```
PERCEPTION (what the system sees)
    │
    ├── YOLO: "bird detected, 94% confidence, bbox [120,80,340,290]"
    ├── VLM:  "A blue jay perched on the wooden fence post, facing left"
    └── Motion: "4.2% frame motion, 1 region, brightness 142"
        │
        ▼
MEMORY (what the system knows)
    │
    ├── Short-term: "Last 10 detections in the past hour"
    ├── Long-term: "Blue jays visit at 7am and 4pm daily"
    ├── RAG: vector store of all past sightings + descriptions
    └── Species database: local knowledge of backyard wildlife
        │
        ▼
REASONING (what the system thinks)
    │
    ├── LLM: "Blue jay at feeder is normal. But there's also a hawk
    │         in the background — this is unusual and could threaten
    │         the songbirds."
    ├── Pattern matching: "Hawk sightings increased 3x this week"
    └── Priority assessment: "High priority — potential predator"
        │
        ▼
ACTION (what the system does)
    │
    ├── Save high-res frame to S3
    ├── Log to species database
    ├── Send push notification: "Hawk spotted near feeder"
    ├── Update dashboard statistics
    └── Trigger sound deterrent (if you add a speaker)
```

#### Tool Use — How the Agent Takes Actions

The LLM doesn't send notifications directly. It calls tools:

```json
{
  "tool": "send_notification",
  "arguments": {
    "priority": "high",
    "title": "Predator Alert",
    "body": "Red-tailed hawk spotted near bird feeder. 3rd sighting this week.",
    "image_url": "s3://bucket/hawk_20260910.jpg"
  }
}
```

Tools you'd build:
- `save_to_database(species, confidence, description, image_path)`
- `query_sighting_history(species, days_back)`
- `send_notification(priority, title, body)`
- `update_dashboard(event_type, data)`
- `compare_with_previous(current_image, species)` — re-identification

#### RAG — Giving the Agent Memory

**Retrieval-Augmented Generation** lets the agent look up past sightings:

```
Agent question: "Have I seen this hawk before?"
    │
    ▼
Embed current image description → vector
    │
    ▼
Search vector database for similar past descriptions
    │
    ▼
Top 3 matches:
  1. "Red-tailed hawk on fence post, Sep 8, 4:12pm" (similarity: 0.94)
  2. "Red-tailed hawk circling above yard, Sep 6, 3:45pm" (similarity: 0.87)
  3. "Cooper's hawk in oak tree, Sep 3, 11:20am" (similarity: 0.72)
    │
    ▼
Agent reasoning with context:
  "This appears to be the same red-tailed hawk from Sep 6 and Sep 8.
   It's visiting more frequently — every 2 days.
   The Cooper's hawk from Sep 3 was a different species."
```

**Vector database options:**
- ChromaDB (simplest, local, Python)
- Qdrant (faster, runs in Docker)
- pgvector (if you already use PostgreSQL)

### 2.5 Training and Fine-Tuning

#### Why Fine-Tune?

YOLO-nano with COCO knows 80 classes. It knows "bird" but not "blue jay vs cardinal vs hawk." Your backyard has maybe 15 specific species.

**Options ranked by effort:**

1. **Zero-shot VLM** (no training): Ask LLaVA "What species of bird is this?" — works okay for common species, fails for similar-looking ones.

2. **Few-shot prompting** (no training): Give the VLM 3-4 example images in the prompt. Better but uses lots of tokens/VRAM.

3. **Fine-tune YOLO on your data** (moderate effort): Collect 50-100 labeled images per species from your camera. Fine-tune YOLO to detect your specific backyard species. Result: fast, accurate, runs on Pi.

4. **LoRA fine-tune a VLM** (more effort): Fine-tune LLaVA/Qwen-VL on your wildlife images with species labels. Result: VLM that's an expert on your local wildlife.

#### LoRA — Fine-Tuning Without Breaking the Bank

**Low-Rank Adaptation**: Instead of updating all 7 billion parameters, LoRA adds small trainable matrices (~1-10M parameters) alongside the frozen base model.

```
Original: 7B parameters, all frozen
LoRA adapter: ~10M parameters, trainable
Total training VRAM: ~10-12GB (fits on your 3060!)
Merged model: still 7B, but now knows your species
```

**Why this matters:** You can fine-tune a VLM on your actual backyard wildlife images. 50 labeled images per species, a few hours of training on your 3060, and suddenly the model knows "that's a dark-eyed junco, not a house sparrow."

### 2.6 GPU and Hardware Deep Knowledge

#### GPU Architecture — What's Actually Inside Your RTX 3060

```
RTX 3060 12GB
├── 3584 CUDA cores          — parallel math (matrix multiply for neural nets)
├── 112 Tensor cores         — specialized for FP16/INT8 matrix ops (2-4x faster for AI)
├── 28 RT cores              — ray tracing (not relevant for AI)
├── 12GB GDDR6 VRAM          — where model weights + KV-cache + activations live
├── 360 GB/s memory bandwidth — how fast data moves between VRAM and compute
└── 170W TDP                 — power budget
```

**The bottlenecks you'll hit:**

1. **VRAM capacity** (12GB): determines which models fit
   - YOLO-nano: 0.5GB
   - 7B model (4-bit): 3.5-5GB
   - KV-cache for 4K context: 0.2-0.4GB
   - CUDA overhead: ~0.5-1GB
   - Available for models: ~10-11GB usable

2. **Memory bandwidth** (360 GB/s): determines token generation speed
   - Each token generated must read ALL model weights from VRAM
   - 7B model at 4-bit = 3.5GB read per token
   - Theoretical max: 360/3.5 = ~100 tokens/sec
   - Real-world: 30-50 tokens/sec (overhead, KV-cache reads, etc.)

3. **Compute** (Tensor cores): determines prefill speed
   - Processing the initial prompt is compute-bound (matrix math)
   - Token generation is memory-bandwidth-bound (weight reading)
   - This is why first-token latency is different from per-token latency

#### The Experiments to Run (and What They'll Teach You)

**Experiment 1: VRAM vs Model Size**
```
Load models one at a time, measure VRAM:
  yolov8n          →  0.5GB
  yolov8s          →  1.0GB
  yolov8m          →  2.5GB
  llama-7B-Q4      →  4.5GB
  llama-7B-Q8      →  8.5GB
  llama-7B-FP16    →  14GB (won't fit!)

Lesson: Quantization is the difference between "fits" and "doesn't fit"
```

**Experiment 2: Throughput vs Concurrency**
```
Run YOLO alone:                    80 FPS
Run YOLO + VLM:                    YOLO 60 FPS, VLM 5 tokens/sec
Run YOLO + VLM + LLM:             YOLO 40 FPS, VLM 3 tok/s, LLM 8 tok/s
Run YOLO + VLM + LLM (2 clients): everything 50% slower

Lesson: GPU time-sharing has real costs. Models don't just share — they compete.
```

**Experiment 3: Latency Breakdown**
```
Pi captures frame:                  50ms
Network transfer (WiFi):           20-100ms
YOLO inference:                    30ms
VLM inference (describe image):    800-2000ms
LLM reasoning:                     500-1500ms
Agent tool execution:              50-200ms
Total end-to-end:                  1.5-4 seconds

Lesson: The VLM is the bottleneck. Optimize there first.
```

**Experiment 4: Context Length vs VRAM**
```
Generate with 512 context:    4.5GB VRAM
Generate with 2048 context:   5.2GB VRAM
Generate with 4096 context:   6.1GB VRAM
Generate with 8192 context:   7.8GB VRAM
Generate with 16384 context:  11.2GB VRAM (barely fits!)

Lesson: KV-cache growth is linear but significant. Long agent conversations eat VRAM.
```

---

## Part 3: The Gaps You Might Be Missing

Based on your trajectory (GenAI → agents → AWS → CV → Pi → edge), here are the areas most people skip:

### Gap 1: Networking Between Devices

You know AWS networking. But Pi-to-laptop local networking is different:
- **mDNS/Avahi**: Find your laptop on the network by name (e.g., `hemanth-laptop.local`) instead of memorizing IP addresses
- **Latency**: WiFi adds 5-50ms per hop. If your pipeline has 4 network calls, that's 20-200ms just in networking
- **Reliability**: WiFi drops packets. Your pipeline needs retry logic and timeouts
- **Security**: Even on your home network, API endpoints should use authentication

### Gap 2: Model Evaluation — How Do You Know It's Better?

You added solidity filtering — but how much better is it? You need:
- **Ground truth**: Manually label 100 images as "real animal" or "false positive"
- **Precision**: Of all triggers, what % were real animals?
- **Recall**: Of all real animal appearances, what % did you capture?
- **F1 score**: The balance between precision and recall

```
Before (motion only):    Precision 15%, Recall 95%, F1 0.26
After (solidity+multi):  Precision 45%, Recall 85%, F1 0.59
After (+ YOLO):          Precision 82%, Recall 70%, F1 0.75
After (+ VLM verify):    Precision 95%, Recall 65%, F1 0.77
```

The tradeoff: more filtering = fewer false positives but also some missed animals.

### Gap 3: Profiling and Debugging GPU Workloads

When something is slow, you need to know WHY:
- `nvidia-smi` — basic GPU monitoring (utilization, VRAM, temperature)
- `nvidia-smi dmon` — continuous monitoring
- `torch.cuda.memory_summary()` — detailed VRAM breakdown in PyTorch
- `torch.profiler` — trace exactly which operations are slow
- `nsight systems` — NVIDIA's full profiler (advanced)

### Gap 4: Containerization for Multi-Model Serving

Running YOLO + VLM + agent as separate Python scripts is fragile. Production approach:
```
Docker Compose:
  ├── yolo-service (container 1, GPU)
  ├── vlm-service  (container 2, GPU)
  ├── agent        (container 3, CPU)
  ├── vector-db    (container 4, CPU)
  └── dashboard    (container 5, CPU)
```

Each service has its own dependencies, restarts independently, scales separately.

### Gap 5: Data Pipeline — From Raw Frames to Training Data

Your camera generates data. That data should feed back into making the model better:

```
Camera captures → YOLO detects → VLM describes → Human reviews
         │                                              │
         └──────── Training data ◄──────────────────────┘
                        │
                   Fine-tune YOLO + VLM
                        │
                   Deploy updated model
                        │
                   Better detections → Better data → cycle continues
```

This is the **data flywheel** — the system gets better the more it runs.

---

## Part 4: Your Build Order

Don't try everything at once. Here's the sequence that builds on what you have:

### Phase 1: VLM on Laptop (Week 1-2)
- Install Ollama on your RTX 3060 laptop
- Run LLaVA or Qwen2.5-VL locally
- Test manually: feed wildlife images, see what it says
- Measure: VRAM usage, inference time, description quality

### Phase 2: Pi-to-Laptop Connection (Week 2-3)
- Build a FastAPI server on the laptop that accepts images
- Pi sends motion-triggered frames to the laptop via HTTP
- Laptop runs YOLO + VLM, returns species + description
- Pi logs the enriched results to CSV

### Phase 3: Agent Layer (Week 3-4)
- Add an LLM-based agent on the laptop
- Agent receives YOLO + VLM results
- Agent decides: log, notify, compare with history
- Add tool functions: save to DB, send notification, query history

### Phase 4: RAG Memory (Week 4-5)
- Set up ChromaDB on the laptop
- Store every sighting: embedding of VLM description + metadata
- Agent queries RAG: "Have I seen this animal before?"
- Agent identifies patterns: "Hawks visit Tue/Thu afternoons"

### Phase 5: Benchmarking (Week 5-6)
- Run the GPU experiments from Part 2.6
- Measure everything: VRAM, latency, throughput, FPS
- Find the bottleneck
- Optimize: TensorRT for YOLO, quantization for VLM, batching

### Phase 6: Fine-Tuning (Week 6-8)
- Label 500 images from your camera (species, behavior)
- Fine-tune YOLO on your specific species
- LoRA fine-tune VLM on your wildlife descriptions
- Measure improvement: before vs after accuracy

### Phase 7: Production Infrastructure (Week 8-10)
- Docker Compose for all services
- Monitoring dashboard (Grafana)
- Auto-restart on failure
- Multi-camera support

### Phase 8: Cloud Scale (Week 10+)
- When the 3060 isn't enough, move inference to AWS
- GPU instances (g5.xlarge with A10G)
- Compare: local cost vs cloud cost for your workload
- Distributed inference for larger models

---

## Part 5: Quick Reference — Every Concept Connected

| Concept | What It Is | How It Connects to Your Project |
|---------|-----------|-------------------------------|
| Transformer | Attention-based neural network | Inside every model you use |
| ViT | Vision Transformer | How VLMs process your camera frames |
| MoE | Mixture of Experts routing | Better models at same compute cost |
| Quantization (4/8-bit) | Reduced precision weights | Fitting models on your 3060 |
| KV-cache | Stored attention keys/values | Why long agent conversations use VRAM |
| Speculative decoding | Draft model + verify | Faster VLM responses |
| Continuous batching | Dynamic request scheduling | Multi-camera support on one GPU |
| TensorRT | GPU-optimized inference | 3-4x faster YOLO |
| vLLM | LLM serving engine | Production model serving |
| LoRA | Low-rank fine-tuning | Teaching models your local species |
| RAG | Retrieval-augmented generation | Agent memory of past sightings |
| Tool use | LLM calling functions | Agent taking actions (notify, save, query) |
| Vector DB | Embedding similarity search | Finding similar past sightings |
| CUDA | GPU programming interface | Understanding GPU bottlenecks |
| Tensor parallelism | Splitting model across GPUs | Running models too big for one GPU |
| NCCL | Multi-GPU communication | Fast data transfer between GPUs |
| PagedAttention | Virtual memory for KV-cache | Efficient VRAM usage in vLLM |
| Flash Attention | Optimized attention computation | Faster and less memory for long contexts |
| GGUF | Quantized model format | Running models via llama.cpp |
| Embedding model | Text/image → vector | Powering RAG similarity search |
| Data flywheel | Capture → label → train → deploy | System improves itself over time |

---

## Part 6: The Story This Tells

When someone asks "What have you built?", you say:

> I built an autonomous wildlife monitoring system. A Raspberry Pi camera runs lightweight motion detection at the edge — I engineered solidity filtering and multi-frame temporal consistency to eliminate false positives from leaves and wind. When real motion is detected, the frame is sent to a local GPU server running YOLOv8 for detection and a Vision Language Model for scene understanding. An LLM-based agent reasons about the sighting — comparing it against a RAG database of past observations, identifying species patterns, and taking actions like notifications. I benchmarked the entire pipeline: VRAM utilization, inference latency, throughput under concurrent load, and identified that the VLM is the bottleneck at 1.5s per frame. I optimized with TensorRT for YOLO (3x speedup), 4-bit quantization for the VLM, and continuous batching for multi-camera support. The system runs unattended for weeks, improving over time through a data flywheel — captured images feed back into fine-tuning with LoRA.

That's one project. It covers: edge computing, computer vision, GPU engineering, inference optimization, multimodal AI, agentic systems, RAG, fine-tuning, and production infrastructure.
