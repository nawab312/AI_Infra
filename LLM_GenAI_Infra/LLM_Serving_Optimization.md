# LLM Serving & Optimization — Complete Study Guide

> **Context:** This is Section 7.1 of the DevOps/SRE → AI Infra / MLOps Engineer transition roadmap.
> This is the "how do we actually run these massive models in production" layer — and it's where your infrastructure skills translate most directly.

---

## Table of Contents

1. [Core Concepts — How LLM Inference Works](#1-core-concepts--how-llm-inference-works)
2. [Serving Frameworks](#2-serving-frameworks)
3. [Inference Optimization Techniques](#3-inference-optimization-techniques)
4. [Hardware Knowledge](#4-hardware-knowledge)
5. [Production Deployment Patterns](#5-production-deployment-patterns)
6. [Benchmarking & Performance Tuning](#6-benchmarking--performance-tuning)
7. [Hands-On Projects](#7-hands-on-projects)

---

## 1. Core Concepts — How LLM Inference Works

Before touching any framework, you need to understand what's actually happening when an LLM generates text. This section is the foundation everything else builds on.

### 1.1 The Autoregressive Generation Loop

LLMs generate text **one token at a time**. This is fundamentally different from most ML models you may have worked with, which produce their entire output in a single forward pass.

```
Input: "The capital of France is"

Step 1: Model processes all input tokens → predicts next token: "Paris"
Step 2: Model processes "The capital of France is Paris" → predicts: ","
Step 3: Model processes "The capital of France is Paris," → predicts: "a"
Step 4: Model processes "The capital of France is Paris, a" → predicts: "city"
... continues until stop condition (max tokens, EOS token, stop sequence)

Each next word = full model run again
So if output = 500 tokens → 500 times model runs
And each step depends on previous output → can’t parallelize
```

**Why this matters for infrastructure:**
Each step requires a full forward pass through the model. A 70B parameter model generating 500 tokens means 500 sequential forward passes. You cannot parallelize across tokens — each token depends on all previous tokens. This is why LLM serving is fundamentally **memory-bandwidth bound**, not compute-bound during generation.

### 1.2 Prefill vs Decode Phases

Every LLM inference request has two distinct phases with very different computational characteristics:

**Prefill Phase (also called "prompt processing" or "context encoding"):**
- Processes ALL input tokens in parallel (the prompt, system message, context)
- Compute-intensive — lots of matrix multiplications on the GPU
- Produces the KV cache entries for all input tokens
- Happens once per request
- Latency here determines the **Time to First Token (TTFT)**

**Decode Phase (also called "generation" or "autoregressive decoding"):**
- Generates tokens one at a time
- Memory-bandwidth-intensive — each step reads the entire model weights and KV cache but only produces one token
- The GPU is underutilized during decode because the computation is small relative to the data movement
- Latency per step determines the **Inter-Token Latency (ITL)** — what the user perceives as "typing speed"

```
Timeline of a single request:

|←——— Prefill ———→|←———————— Decode ————————————→|
|  Process prompt   | Token 1 | Token 2 | ... | Token N |
|  (parallel, fast) | (sequential, one at a time)        |
|                   |                                      |
|  TTFT measured    | ITL measured between each token      |
|  at this boundary |                                      |
```

**Why this distinction matters for infra engineers:**

1. **Prefill is compute-bound.** Long prompts (think RAG with 10K tokens of context) stress the GPU's compute capacity. Short prompts are cheap.

2. **Decode is memory-bandwidth-bound.** The GPU spends most of its time reading weights from VRAM, not computing. This is why higher memory bandwidth GPUs (H100 > A100) give better decode performance.

3. **They interfere with each other.** If a serving system is processing a long prefill for one request, decode steps for other in-flight requests get delayed — causing latency spikes. Modern frameworks handle this with **chunked prefill** (breaking large prefills into smaller chunks interleaved with decode steps).

4. **Different optimization strategies.** Prefill benefits from compute optimization (FlashAttention, kernel fusion). Decode benefits from memory optimization (quantization, smaller KV cache).

Some advanced architectures even **disaggregate** prefill and decode onto separate GPU pools — prefill on compute-heavy GPUs, decode on memory-bandwidth-optimized GPUs.

### 1.3 KV Cache — The Critical Memory Bottleneck

The KV cache is the single most important concept for understanding LLM serving infrastructure. If you understand KV cache, you understand why vLLM exists.

**What is the KV cache?**

In the Transformer's attention mechanism, each token produces a Key (K) and Value (V) vector at every layer. During autoregressive generation, the model needs the K and V vectors of ALL previous tokens to generate the next token.

Without caching, generating token N would require recomputing the K and V vectors for tokens 1 through N-1 — an O(n²) operation that would make generation impossibly slow.

The **KV cache** stores these K and V vectors so they only need to be computed once.

**How big is the KV cache?**

```
KV Cache Size = 2 × num_layers × num_heads × head_dim × sequence_length × batch_size × bytes_per_element

Example: Llama 3 70B with FP16
= 2 × 80 layers × 8 KV heads (GQA) × 128 head_dim × seq_len × batch × 2 bytes
= 327,680 bytes per token per request
≈ 320 KB per token per request

For a single request with 4096 context:
= 320 KB × 4096 = 1.28 GB — just for KV cache of ONE request!

For 32 concurrent requests at 4096 context:
= 1.28 GB × 32 = 40.96 GB — just KV cache, before model weights!
```

**The memory crisis:**
A 70B parameter model in FP16 needs ~140 GB just for weights. Add KV cache for concurrent requests and you're easily exceeding the VRAM of even an 8×A100 (640 GB total) setup. This is why KV cache management is the central challenge of LLM serving.

**KV cache memory fragmentation (the problem vLLM solves):**

Before vLLM, serving frameworks allocated KV cache memory **contiguously** — like malloc in C. Each request got a pre-allocated block sized for the maximum possible sequence length.

```
Before vLLM — Contiguous KV Cache Allocation:

GPU Memory:
|  Model Weights  | Req 1 (max 2048) | Req 2 (max 2048) | Req 3 (max 2048) | WASTED |
                  |■■■■░░░░░░░░░░░░░|■■░░░░░░░░░░░░░░░|■■■■■■■░░░░░░░░░░|
                   ■ = actually used   ░ = reserved but unused

Problems:
1. Internal fragmentation: Each request reserves max length but uses much less
2. External fragmentation: Free space exists but can't be used
3. Typical memory utilization: 20-40% of allocated KV cache space
```

This meant only 20-40% of GPU memory allocated for KV cache was actually being used. The rest was wasted on reservations for tokens that might never be generated.

### 1.4 Model Size vs GPU Memory (VRAM)

**Quick math every AI infra engineer must know:**

```
Model Memory ≈ Parameters × Bytes per Parameter

Precision    | Bytes | 7B Model | 13B Model | 70B Model
-------------|-------|----------|-----------|----------
FP32         | 4     | 28 GB    | 52 GB     | 280 GB
FP16/BF16    | 2     | 14 GB    | 26 GB     | 140 GB
INT8         | 1     | 7 GB     | 13 GB     | 70 GB
INT4         | 0.5   | 3.5 GB   | 6.5 GB    | 35 GB
```

**But that's just model weights.** Total VRAM needed:

```
Total VRAM = Model Weights + KV Cache + Activation Memory + Framework Overhead

Model Weights: Fixed cost, depends on precision
KV Cache: Variable, grows with concurrent requests × context length
Activation Memory: Temporary buffers during computation (~10-20% overhead)
Framework Overhead: CUDA context, memory allocator overhead (~1-2 GB)
```

**Practical GPU planning:**

| Model Size | Precision | Min GPUs (A100 80GB) | Min GPUs (H100 80GB) | Notes |
|-----------|-----------|---------------------|----------------------|-------|
| 7-8B | FP16 | 1 | 1 | Fits easily on a single GPU |
| 7-8B | INT4 | 1 | 1 | Leaves lots of room for KV cache |
| 13B | FP16 | 1 | 1 | Tight fit, limited concurrent requests |
| 34B | FP16 | 1-2 | 1 | Depends on context length needed |
| 70B | FP16 | 2-4 | 2 | 2 GPUs minimum, 4 for good throughput |
| 70B | INT4 | 1 | 1 | Fits on 1 GPU, quality tradeoff |
| 405B | FP16 | 8-16 | 8 | Multi-node required |
| 405B | FP8 | 4-8 | 4-8 | H100's native FP8 support helps |

### 1.5 Throughput vs Latency Tradeoffs

These are the two fundamental metrics that pull in opposite directions:

**Latency metrics:**
- **TTFT (Time to First Token):** How long until the first token appears. Critical for interactive applications. Target: < 500ms for chat, < 1s for most apps.
- **ITL (Inter-Token Latency):** Time between consecutive generated tokens. Determines perceived "typing speed." Target: < 50ms for smooth streaming (roughly 20 tokens/sec).
- **E2E Latency:** Total time from request to complete response.

**Throughput metrics:**
- **Tokens/second (aggregate):** Total tokens generated per second across all requests. The key metric for cost efficiency.
- **Requests/second:** How many concurrent requests can be served.

**The fundamental tradeoff:**

Batching more requests together **increases throughput** (GPU compute is shared more efficiently) but **increases latency** (each individual request waits longer as the GPU is shared).

```
Batch Size 1:   Low throughput, lowest latency
Batch Size 8:   Good throughput, moderate latency
Batch Size 32:  High throughput, higher latency
Batch Size 128: Maximum throughput, potentially unacceptable latency
```

**How to think about this as an infra engineer:**
- **Real-time chat applications:** Optimize for latency. Smaller batches, faster GPUs, maybe sacrifice some throughput.
- **Batch processing (summarization, embeddings):** Optimize for throughput. Large batches, maximize GPU utilization, latency doesn't matter.
- **API serving (mixed workloads):** Balance both. Use continuous batching (next section) to dynamically adjust.

### 1.6 Batching Strategies

**Static Batching (Naive Approach):**

Collect N requests, process them together as a batch. All requests in the batch must wait for the longest one to finish before any response is sent.

```
Static Batch:
Request A: "Hi"        → generates 5 tokens   [done at step 5]
Request B: "Tell me..." → generates 200 tokens [done at step 200]
Request C: "What is..."  → generates 50 tokens  [done at step 50]

All three must wait until step 200 before results are returned.
Request A wastes 195 decode steps worth of GPU time.
```

**Problems:** Requests that finish early waste GPU resources waiting. New requests can't join until the entire batch completes. GPU utilization fluctuates wildly.

**Continuous Batching (also called "iteration-level batching" or "inflight batching"):**

The key innovation that makes modern LLM serving practical. Instead of batching at the request level, batch at the **token-generation step level**.

```
Continuous Batching:

Step 1: [A, B, C] — all three generate their next token
Step 2: [A, B, C] — all three generate their next token
...
Step 5: [A, B, C] — A finishes! Slot opens.
Step 6: [D, B, C] — New request D immediately joins the batch
...
Step 50: [D, B, E] — C finished, E joined
```

**Benefits:**
- Completed requests immediately free their slot for new requests
- GPU utilization stays high because slots are always filled
- Latency is much lower because requests don't wait for slow batch-mates
- Throughput is higher because the GPU never idles waiting for a batch to complete

All modern serving frameworks (vLLM, TGI, TensorRT-LLM) implement continuous batching. This is not optional — it's table stakes for production LLM serving.

---

## 2. Serving Frameworks

### 2.1 vLLM — The Industry Standard

vLLM is the most widely deployed open-source LLM serving framework. It was created by UC Berkeley researchers and the key innovation is **PagedAttention**.

#### PagedAttention — The Core Innovation

PagedAttention applies the concept of **virtual memory paging** (from operating systems) to KV cache management. If you've studied OS memory management, this will click immediately.

```
Traditional KV Cache: Contiguous allocation (like malloc)
|████░░░░░░░░░░░░|████████░░░░░░░░|██░░░░░░░░░░░░░░|
 Request 1 (wasted) Request 2 (wasted) Request 3 (wasted)

PagedAttention: Block-based allocation (like virtual memory pages)
Block Table for Request 1: [Block 3, Block 7, Block 1]
Block Table for Request 2: [Block 5, Block 2, Block 8, Block 4]
Block Table for Request 3: [Block 6, Block 9]

Physical GPU Memory (blocks):
|B1|B2|B3|B4|B5|B6|B7|B8|B9|  |  |  |
 R1 R2 R1 R2 R2 R3 R1 R2 R3  free free free
```

**How it works:**
1. KV cache memory is divided into fixed-size **blocks** (like OS memory pages, typically 16 tokens per block)
2. Each request has a **block table** (like a page table) that maps logical positions to physical blocks
3. Blocks are allocated **on demand** as tokens are generated — no pre-allocation waste
4. When a request finishes, its blocks are immediately returned to the free pool
5. Blocks don't need to be contiguous in GPU memory — the block table handles the mapping

**Results:**
- Memory utilization jumps from 20-40% to **near 100%**
- 2-4x more concurrent requests on the same hardware
- Near-zero memory waste from fragmentation

**Bonus: Copy-on-Write for Shared Prefixes**

When multiple requests share the same system prompt (very common in production — every request to your chatbot starts with the same system instructions), PagedAttention can **share** the KV cache blocks for that prefix.

```
Request 1: [System Prompt] + "What is RAG?"
Request 2: [System Prompt] + "Explain transformers"
Request 3: [System Prompt] + "Help me with code"

Without sharing: 3 copies of system prompt KV cache
With sharing:    1 copy, all three requests reference the same blocks
                 (copy-on-write if any request modifies shared blocks)
```

This is called **prefix caching** and can save enormous amounts of memory when your application has standardized system prompts.

#### vLLM Architecture

```
┌─────────────────────────────────────────────────┐
│                   vLLM Server                     │
│                                                   │
│  ┌──────────────┐    ┌────────────────────────┐  │
│  │  API Server   │    │     LLM Engine          │  │
│  │  (FastAPI)    │───▶│                          │  │
│  │              │    │  ┌────────────────────┐  │  │
│  │  OpenAI-     │    │  │   Scheduler        │  │  │
│  │  compatible  │    │  │   (continuous      │  │  │
│  │  /v1/chat/   │    │  │    batching)       │  │  │
│  │  completions │    │  └────────┬───────────┘  │  │
│  └──────────────┘    │           │               │  │
│                      │  ┌────────▼───────────┐  │  │
│                      │  │   Model Executor    │  │  │
│                      │  │   (GPU workers)     │  │  │
│                      │  │                      │  │  │
│                      │  │  ┌────────────────┐ │  │  │
│                      │  │  │ PagedAttention  │ │  │  │
│                      │  │  │ KV Cache Mgr    │ │  │  │
│                      │  │  └────────────────┘ │  │  │
│                      │  └────────────────────┘  │  │
│                      └────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

#### vLLM Key Configuration Parameters

**Critical parameters to understand and tune:**

```bash
# Basic serving command
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \          # Shard model across 4 GPUs
    --gpu-memory-utilization 0.90 \     # Use 90% of GPU memory for KV cache
    --max-model-len 8192 \              # Maximum sequence length
    --max-num-seqs 256 \                # Maximum concurrent sequences
    --enable-prefix-caching \           # Share KV cache for common prefixes
    --quantization awq \                # Use AWQ quantization
    --dtype float16 \                   # Model weight precision
    --port 8000                         # API port
```

**Parameter deep dive:**

| Parameter | What It Does | Tuning Guidance |
|-----------|-------------|-----------------|
| `tensor-parallel-size` | Splits model across N GPUs. Each GPU holds 1/N of each layer. | Set to number of GPUs. Must evenly divide `num_attention_heads`. |
| `gpu-memory-utilization` | Fraction of GPU memory for KV cache (after model weights). | Default 0.9. Lower if you get OOM. Higher for more concurrent requests. |
| `max-model-len` | Maximum context length the server will accept. | Lower = more concurrent requests (less KV cache per request). Set to your actual max need. |
| `max-num-seqs` | Maximum number of concurrent sequences. | Higher = more throughput but more memory. Start with 256, adjust based on benchmarks. |
| `enable-prefix-caching` | Reuse KV cache for shared prompt prefixes. | Always enable if requests share system prompts. Major memory saving. |
| `quantization` | Model weight compression (awq, gptq, fp8, etc.). | AWQ and GPTQ for INT4, FP8 for H100s. Quality-speed tradeoff. |
| `enforce-eager` | Disable CUDA graph optimization. | Use for debugging. Remove in production. |
| `swap-space` | CPU RAM (GB) to swap KV cache when GPU is full. | Set to 4-16 GB. Allows handling burst traffic by offloading to CPU. |
| `max-num-batched-tokens` | Maximum tokens processed in a single batch iteration. | Higher = better throughput, more memory. Tune based on GPU compute capacity. |

#### Deploying vLLM with Docker and Kubernetes

**Docker deployment:**

```dockerfile
# Production Dockerfile
FROM vllm/vllm-openai:latest

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run vLLM server
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "meta-llama/Llama-3.1-8B-Instruct", \
     "--tensor-parallel-size", "1", \
     "--gpu-memory-utilization", "0.9", \
     "--max-model-len", "4096"]
```

```bash
# Run with GPU access
docker run --gpus all \
    -p 8000:8000 \
    -v /path/to/model/cache:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```

**Kubernetes deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-70b
spec:
  replicas: 1  # Each replica needs its own GPU set
  selector:
    matchLabels:
      app: vllm-llama-70b
  template:
    metadata:
      labels:
        app: vllm-llama-70b
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "meta-llama/Llama-3.1-70B-Instruct"
        - "--tensor-parallel-size"
        - "4"
        - "--gpu-memory-utilization"
        - "0.90"
        - "--max-model-len"
        - "8192"
        - "--enable-prefix-caching"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 4  # Request 4 GPUs
          requests:
            memory: "64Gi"
            cpu: "16"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
        - name: shm
          mountPath: /dev/shm  # Required for tensor parallelism
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120  # Models take time to load
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "16Gi"  # Shared memory for NCCL
      nodeSelector:
        gpu-type: a100-80gb  # Target specific GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama-70b
spec:
  selector:
    app: vllm-llama-70b
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

**Key Kubernetes considerations for vLLM:**
- **`/dev/shm` (shared memory):** Must be large (8-16 GB) for NCCL communication between GPUs during tensor parallelism. Without this, multi-GPU serving will fail or crawl.
- **`initialDelaySeconds`:** Set high (120-300s). Large models take minutes to load into GPU memory.
- **Node selection:** Use node selectors or node affinity to ensure pods land on nodes with the right GPU type and count.
- **Persistent volumes:** Cache model weights on PVCs so they're not re-downloaded on every pod restart.
- **HPA (Horizontal Pod Autoscaler):** Scale based on custom metrics (request queue depth, GPU utilization) rather than CPU/memory.

#### vLLM's OpenAI-Compatible API

vLLM exposes an API that's a drop-in replacement for OpenAI's API:

```python
# Your application code doesn't change — just change the base URL
from openai import OpenAI

client = OpenAI(
    base_url="http://vllm-service:8000/v1",  # Point to vLLM
    api_key="not-needed"                       # vLLM doesn't require a key
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Explain Kubernetes"}],
    max_tokens=500,
    temperature=0.7
)
```

This means you can swap between OpenAI and self-hosted models without changing application code — just change the base URL. This is a massive operational advantage.

### 2.2 Text Generation Inference (TGI) — Hugging Face's Solution

TGI is Hugging Face's production-ready LLM serving framework. Written in Rust for performance.

**Architecture differences from vLLM:**

| Aspect | vLLM | TGI |
|--------|------|-----|
| Language | Python + CUDA C++ | Rust + Python |
| KV cache management | PagedAttention (paged blocks) | Contiguous pre-allocation with dynamic management |
| Batching | Continuous batching | Continuous batching (token-level scheduling) |
| API | OpenAI-compatible | Custom API + OpenAI-compatible (newer versions) |
| Flash Attention | Supported | Built-in, first-class |
| Speculative decoding | Supported | First-class support |
| Model support | Broader (more architectures) | Focused on Hugging Face models |
| Deployment | Docker, Kubernetes | Docker, Kubernetes, Hugging Face Inference Endpoints |

**Key TGI features:**

**Flash Attention Integration:** TGI was one of the first frameworks to deeply integrate Flash Attention — an optimized attention algorithm that reduces memory usage from O(n²) to O(n) and significantly speeds up the prefill phase.

**Watermark-based Batching:** TGI uses watermark levels for batch management — when the number of waiting requests exceeds a high watermark, batching becomes more aggressive; below a low watermark, it prioritizes latency.

**Speculative Decoding (first-class support):**
Uses a small "draft" model to propose multiple tokens, then the large model verifies them in a single forward pass. Can give 2-3x speedup for certain workloads.

```
Without speculative decoding:
Large model: Token1 → Token2 → Token3 → Token4  (4 forward passes)

With speculative decoding:
Draft model: proposes [Token1, Token2, Token3, Token4]  (fast, small model)
Large model: verifies all 4 in ONE forward pass
If all accepted: 4 tokens for the cost of ~1 large model forward pass
If Token3 rejected: keep Token1, Token2, regenerate from Token3
```

**When to choose TGI over vLLM:**
- You're already deeply integrated with Hugging Face ecosystem
- You need native speculative decoding support
- You want to deploy via Hugging Face Inference Endpoints (managed)
- You prefer Rust-based performance characteristics
- Your model is a standard Hugging Face model

**When vLLM wins:**
- Broader model architecture support
- PagedAttention gives better memory efficiency under high concurrency
- More active open-source community and faster feature development
- Better prefix caching support
- Simpler OpenAI-compatible API

### 2.3 TensorRT-LLM — NVIDIA's Optimized Engine

TensorRT-LLM is NVIDIA's solution for maximum inference performance on NVIDIA hardware. It compiles models into highly optimized CUDA kernels.

**How TensorRT-LLM differs:**

Unlike vLLM and TGI which run models directly from their Hugging Face format, TensorRT-LLM **compiles** the model into an optimized engine:

```
Standard serving (vLLM/TGI):
Hugging Face Model → Load directly → Serve

TensorRT-LLM:
Hugging Face Model → Convert → Build TRT Engine → Serve
                     (minutes)  (minutes to hours)   (optimized)
```

The compilation step takes time but produces a model that's heavily optimized:
- Kernel fusion (combining multiple operations into single GPU kernels)
- Precision calibration (automatically determining which layers can be quantized)
- Memory layout optimization for specific GPU architectures
- Custom CUDA kernels for attention patterns

**Key TensorRT-LLM concepts:**

**Engine Building:**
```bash
# Example: Build a TensorRT-LLM engine for Llama 70B
# Step 1: Convert checkpoint
python convert_checkpoint.py \
    --model_dir /models/llama-70b \
    --output_dir /engines/llama-70b/checkpoint \
    --tp_size 4 \
    --dtype float16

# Step 2: Build engine (this is the compilation step)
trtllm-build \
    --checkpoint_dir /engines/llama-70b/checkpoint \
    --output_dir /engines/llama-70b/engine \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --use_paged_kv_cache
```

**In-Flight Batching:** TensorRT-LLM's version of continuous batching. Requests can join and leave the batch at each decode iteration.

**FP8 Quantization:** On H100 and newer GPUs, TensorRT-LLM can use native FP8 precision — getting near-INT8 performance with near-FP16 quality. This is a hardware-specific optimization that vLLM and TGI can't match as effectively.

**Integration with Triton Inference Server:**
In production, TensorRT-LLM engines are typically served through NVIDIA's Triton Inference Server, which adds:
- Multi-model serving (run multiple models on the same GPU)
- Dynamic batching at the request level
- Model versioning and A/B testing
- Metrics and monitoring (Prometheus compatible)
- gRPC and HTTP APIs
- Model ensemble pipelines

```
Production TensorRT-LLM Stack:

Client → Triton Inference Server → TensorRT-LLM Engine → GPU
              │
              ├── Model repository (versioned engines)
              ├── Metrics endpoint (/metrics)
              ├── Health endpoint (/v2/health)
              └── Multi-model scheduling
```

**When TensorRT-LLM wins:**
- Maximum throughput on NVIDIA hardware (10-30% faster than vLLM in benchmarks)
- You're on H100/H200/B200 and want FP8 optimization
- You need multi-model serving via Triton
- Latency is absolutely critical (kernel-level optimization)
- You have a dedicated ML platform team that can manage the complexity

**When TensorRT-LLM loses:**
- Much more complex setup (compilation step, engine building)
- Engine must be rebuilt for different batch sizes or sequence lengths
- Less flexible — changing model parameters requires recompilation
- Smaller open-source community
- NVIDIA-only (no AMD or other accelerator support)

### 2.4 Framework Selection Matrix

| Requirement | Best Choice | Why |
|-------------|-------------|-----|
| Quick deployment, broad model support | vLLM | Load and serve in one command |
| Maximum throughput on NVIDIA hardware | TensorRT-LLM | Kernel-level optimization |
| Hugging Face ecosystem integration | TGI | Native HF model support |
| Multi-model serving | TensorRT-LLM + Triton | Triton handles multi-model scheduling |
| OpenAI API drop-in replacement | vLLM | Best OpenAI-compatible API |
| Speculative decoding | TGI | First-class support |
| Memory-constrained (maximize concurrency) | vLLM | PagedAttention is best-in-class |
| Team lacks ML expertise | vLLM | Simplest to deploy and operate |
| Team has strong NVIDIA/CUDA expertise | TensorRT-LLM | Can leverage kernel optimizations |

---

## 3. Inference Optimization Techniques

### 3.1 Quantization — The Most Important Optimization

Quantization reduces the precision of model weights (and sometimes activations) from higher-precision formats (FP32, FP16) to lower-precision formats (INT8, INT4, FP8). This is the single most impactful optimization for inference.

**Why quantization works:**
Neural network weights are stored as floating-point numbers, but they don't actually need 16 or 32 bits of precision. Most weights cluster around small values, and rounding them to lower precision causes minimal quality loss.

**Precision formats explained:**

```
FP32 (32-bit float):
[1 sign bit] [8 exponent bits] [23 mantissa bits]
Range: ±3.4 × 10³⁸, Precision: ~7 decimal digits
Memory: 4 bytes per parameter

FP16 (16-bit float):
[1 sign bit] [5 exponent bits] [10 mantissa bits]
Range: ±65,504, Precision: ~3 decimal digits
Memory: 2 bytes per parameter

BF16 (Brain Float 16):
[1 sign bit] [8 exponent bits] [7 mantissa bits]
Same range as FP32, lower precision
Memory: 2 bytes per parameter
Preferred for training (maintains range, good enough precision)

FP8 (E4M3):
[1 sign bit] [4 exponent bits] [3 mantissa bits]
Memory: 1 byte per parameter
Native on H100+, excellent quality-speed tradeoff

INT8 (8-bit integer):
Range: -128 to 127
Memory: 1 byte per parameter
Good quality, 2x memory reduction from FP16

INT4 (4-bit integer):
Range: -8 to 7 (or 0-15 unsigned)
Memory: 0.5 bytes per parameter
Significant quality loss, 4x memory reduction from FP16
Needs careful calibration
```

**Quantization methods comparison:**

| Method | Bits | Type | Calibration | Quality | Speed | Best For |
|--------|------|------|-------------|---------|-------|----------|
| **FP16/BF16** | 16 | Weight-only | None | Baseline | Baseline | Default, no quantization needed |
| **FP8** | 8 | Weight + Activation | Minimal | Near FP16 | ~2x FP16 | H100/H200, best quality-speed ratio |
| **GPTQ** | 4 | Weight-only, post-training | Calibration dataset required | Good | ~3-4x FP16 | General INT4 quantization |
| **AWQ** | 4 | Weight-only, activation-aware | Calibration dataset required | Better than GPTQ | ~3-4x FP16 | Preferred INT4 for serving |
| **GGUF** | 2-8 | Weight-only, mixed precision | Pre-computed | Varies | Good on CPU | llama.cpp, CPU/Mac inference |
| **bitsandbytes** | 4/8 | Weight-only, runtime | None (on-the-fly) | Good | Slower than GPTQ/AWQ | Quick experimentation, fine-tuning (QLoRA) |
| **SqueezeLLM** | 3-4 | Weight-only, sparse | Calibration dataset required | Good | ~4x FP16 | Extreme compression |

**AWQ vs GPTQ (the two you'll use most):**

**GPTQ (GPT-Quantization):** Quantizes weights layer-by-layer using a calibration dataset. Minimizes the output error of each layer independently. Fast to apply but can accumulate errors across layers.

**AWQ (Activation-Aware Weight Quantization):** Instead of treating all weights equally, AWQ identifies the small percentage of weights that are most important (based on activation magnitudes) and preserves their precision. Less important weights get aggressively quantized. Typically produces better quality than GPTQ at the same bit width.

**Practical guidance:**
- Start with **FP16/BF16** (no quantization) if you have enough GPU memory
- Use **FP8** on H100/H200 — nearly free performance boost with minimal quality loss
- Use **AWQ 4-bit** when you need to fit larger models on fewer GPUs
- Use **GPTQ 4-bit** as an alternative if AWQ models aren't available for your model
- Use **GGUF** only for CPU inference or Apple Silicon (llama.cpp ecosystem)
- Use **bitsandbytes** only for fine-tuning (QLoRA), not for production serving

### 3.2 Speculative Decoding

**The insight:** During autoregressive decoding, the large model is memory-bandwidth-bound and underutilizing GPU compute. What if we could "speculatively" generate multiple tokens and verify them cheaply?

**How it works:**

```
Step 1: Draft model (small, fast) generates K candidate tokens
        Draft: "The" → "capital" → "of" → "France" → "is" → "Paris"
        (6 tokens generated very quickly by a ~1B parameter model)

Step 2: Target model (large, accurate) verifies ALL candidates in ONE forward pass
        Target evaluates: P("capital"|"The"), P("of"|"The capital"),
                          P("France"|"The capital of"), etc.

Step 3: Accept tokens where target agrees with draft
        If target would have generated the same token: ACCEPT
        If target disagrees: REJECT, resample from target, discard rest

Best case: All 6 tokens accepted → 6 tokens for cost of 1 target forward pass
Worst case: First token rejected → same speed as normal decoding
Average: 2-4 tokens accepted → 2-4x speedup
```

**Key parameters:**
- **Draft model:** Must be from the same model family (e.g., Llama 3 8B as draft for Llama 3 70B). Smaller = faster but lower acceptance rate.
- **Speculation length (K):** How many tokens to draft. Higher K = more potential speedup but lower acceptance probability. Typically K=3-6.
- **Acceptance rate:** Depends on how well the draft model matches the target. Same-family models: 70-90%. Different-family: 40-60%.

**When speculative decoding helps most:**
- Simple, predictable text (high acceptance rate)
- Low-concurrency scenarios (GPU compute is available for the draft model)
- When latency matters more than throughput

**When it doesn't help:**
- High concurrency (GPU is already fully utilized)
- Creative/diverse text (low acceptance rate)
- Very small models (already fast, overhead of draft model not worth it)

### 3.3 KV Cache Optimization

Beyond PagedAttention, there are architectural innovations that reduce KV cache size:

**Multi-Query Attention (MQA):**
Instead of each attention head having its own K and V projections, ALL heads share a SINGLE K and V. Reduces KV cache size by `num_heads`x.
- Used by: PaLM, Falcon
- Tradeoff: Slight quality loss, massive memory saving

**Grouped-Query Attention (GQA):**
A compromise — heads are divided into groups, and each group shares K/V projections. If you have 32 heads grouped into 8 groups, KV cache is reduced 4x.
- Used by: Llama 2 70B, Llama 3, Mistral
- Tradeoff: Minimal quality loss, significant memory saving

```
Multi-Head Attention (MHA):  32 heads × 32 KV pairs = 32 KV cache entries
Grouped-Query Attention (GQA): 32 heads × 8 KV pairs  = 8 KV cache entries (4x smaller)
Multi-Query Attention (MQA): 32 heads × 1 KV pair   = 1 KV cache entry  (32x smaller)
```

**Sliding Window Attention:**
Instead of attending to ALL previous tokens, only attend to the last W tokens. KV cache is bounded at W entries regardless of sequence length.
- Used by: Mistral (W=4096)
- Tradeoff: Loses very long-range dependencies, but bounded memory

**Prefix Caching (already covered in vLLM section):**
Reuse KV cache for shared prefixes across requests.

### 3.4 Model Parallelism for Serving

When a model doesn't fit on a single GPU, you need parallelism:

**Tensor Parallelism (TP) — Used for Serving:**

Splits each layer's weight matrices across multiple GPUs. Each GPU processes a portion of every layer.

```
Single GPU:
Layer 1: [Full weight matrix A] → [Full output]

Tensor Parallel across 4 GPUs:
GPU 0: [A_chunk_0] → [partial output 0] ──┐
GPU 1: [A_chunk_1] → [partial output 1] ──┤── AllReduce → [Full output]
GPU 2: [A_chunk_2] → [partial output 2] ──┤
GPU 3: [A_chunk_3] → [partial output 3] ──┘
```

**Key characteristics:**
- Requires high-bandwidth interconnect between GPUs (NVLink). Without NVLink, the AllReduce communication becomes the bottleneck.
- Each GPU holds 1/N of the model + 1/N of the KV cache
- All GPUs must be on the same node (cross-node TP is too slow for inference)
- TP degree must evenly divide `num_attention_heads`

**Pipeline Parallelism (PP) — Less Common for Serving:**

Splits layers across GPUs. GPU 0 runs layers 1-20, GPU 1 runs layers 21-40, etc.

```
Pipeline Parallel:
GPU 0: Layers 1-20   → output → GPU 1
GPU 1: Layers 21-40  → output → GPU 2
GPU 2: Layers 41-60  → output → GPU 3
GPU 3: Layers 61-80  → final output
```

**Key characteristics:**
- Can work across nodes (only sends activations between stages, not AllReduce)
- Creates "pipeline bubbles" — GPU 0 is idle while GPU 3 is computing
- Less efficient for inference than TP (bubbles waste time)
- Used primarily when you can't fit the model with TP alone (e.g., 405B model across 2 nodes × 8 GPUs = TP=8, PP=2)

**Practical serving parallelism:**
- **7-13B models:** Single GPU, no parallelism needed
- **34B models:** TP=2 (2 GPUs on same node)
- **70B models:** TP=4 (4 GPUs on same node)
- **70B quantized INT4:** TP=1 or TP=2
- **405B models:** TP=8, PP=2 (2 nodes × 8 GPUs)

### 3.5 Structured Output Optimization

When you need the LLM to output valid JSON, SQL, or other structured formats:

**Constrained Decoding / Grammar-Guided Generation:**
Instead of sampling freely from the model's vocabulary, restrict the next token to only tokens that would produce valid output according to a grammar (JSON schema, regex, CFG).

```
Normal decoding: Sample from all 32,000 tokens
Constrained decoding: At each step, mask tokens that would produce invalid JSON

Example generating JSON:
Step 1: Must start with '{' → mask all tokens except '{'
Step 2: Must be a key → mask everything except '"'
Step 3: Inside key string → allow alphabet/numbers
...
```

**Tools:**
- **Outlines** (by .txt): Grammar-guided generation library, integrates with vLLM
- **vLLM's guided decoding:** Built-in support for JSON schema and regex constraints
- **Instructor:** Higher-level library that handles structured output with retry logic

**Why this matters for production:** Without constrained decoding, you rely on the LLM to "hopefully" produce valid JSON, then parse and retry on failure. With constrained decoding, valid output is guaranteed, eliminating retries and improving reliability.

---

## 4. Hardware Knowledge

### 4.1 GPU Comparison for LLM Inference

| GPU | VRAM | Memory BW | FP16 TFLOPS | FP8 TFLOPS | NVLink | Cost/hr (Cloud) | Best For |
|-----|------|-----------|-------------|------------|--------|-----------------|----------|
| **A100 40GB** | 40 GB | 1.6 TB/s | 312 | N/A | 600 GB/s | ~$2-3 | Budget inference, smaller models |
| **A100 80GB** | 80 GB | 2.0 TB/s | 312 | N/A | 600 GB/s | ~$3-4 | Standard inference workhorse |
| **H100 SXM** | 80 GB | 3.35 TB/s | 990 | 1,979 | 900 GB/s | ~$4-8 | High-throughput inference, FP8 |
| **H200** | 141 GB | 4.8 TB/s | 990 | 1,979 | 900 GB/s | ~$6-10 | Large models, memory-bound workloads |
| **B200** | 192 GB | 8.0 TB/s | 2,250 | 4,500 | 1,800 GB/s | (newest) | Maximum performance |
| **L40S** | 48 GB | 864 GB/s | 366 | 733 | None | ~$1-2 | Cost-efficient inference for smaller models |
| **A10G** | 24 GB | 600 GB/s | 125 | N/A | None | ~$1 | Small models, 7B quantized |

**Key insight: Memory bandwidth is the bottleneck for decode.**

During the decode phase, the GPU reads the entire model weights from VRAM for each token generated. The rate at which it can read those weights is limited by memory bandwidth.

```
Theoretical max decode tokens/sec ≈ Memory Bandwidth / Model Size in Memory

A100 80GB with Llama 70B FP16 (140 GB across 2 GPUs):
  2.0 TB/s × 2 GPUs / 140 GB ≈ 28 tokens/sec per request

H100 80GB with Llama 70B FP16 (140 GB across 2 GPUs):
  3.35 TB/s × 2 GPUs / 140 GB ≈ 47 tokens/sec per request

H100 80GB with Llama 70B FP8 (70 GB across 1 GPU):
  3.35 TB/s / 70 GB ≈ 47 tokens/sec per request (same speed, half the GPUs!)
```

This is why quantization is so powerful — it directly reduces the data that needs to be read from memory, directly increasing decode speed.

### 4.2 NVLink and NVSwitch

**NVLink:** High-speed point-to-point connection between GPUs. Essential for tensor parallelism.

```
Without NVLink (PCIe only):
GPU ↔ PCIe ↔ CPU ↔ PCIe ↔ GPU
Bandwidth: ~32 GB/s (PCIe Gen4 x16)

With NVLink (A100):
GPU ↔ NVLink ↔ GPU
Bandwidth: 600 GB/s (18.75x faster!)

With NVLink (H100):
GPU ↔ NVLink ↔ GPU
Bandwidth: 900 GB/s
```

**NVSwitch:** Allows full-bisection bandwidth between all GPUs in a node. Instead of point-to-point NVLink connections, NVSwitch creates a "GPU fabric" where any GPU can communicate with any other GPU at full NVLink speed.

**Why this matters:** Tensor parallelism requires AllReduce operations at every layer. If your GPU interconnect is slow (PCIe), the communication overhead kills performance. This is why you should:
- Always use NVLink-connected GPUs for multi-GPU serving
- Never try tensor parallelism across nodes (use pipeline parallelism instead for cross-node)
- Check that your cloud instance type has NVLink (not all multi-GPU instances do)

### 4.3 CPU Offloading Strategies

When GPU memory is insufficient:

**KV Cache Offloading:** Move KV cache of less-active requests to CPU RAM. When those requests become active again, swap back to GPU. vLLM's `--swap-space` parameter controls this.

**Weight Offloading:** Keep some model layers in CPU RAM, moving them to GPU only during computation. Extremely slow — only useful for experimentation, not production.

**Mixed CPU/GPU Inference:** Keep the entire model in CPU RAM and use GPU only for specific layers. Used by llama.cpp's GPU layer offloading. Not suitable for production LLM serving.

### 4.4 Cost-Performance Analysis

**Cost optimization strategies:**

1. **Right-size your GPU:** Don't use H100s for a 7B model. An A10G at 1/8th the cost will serve it just fine.

2. **Quantize aggressively:** INT4 on a single A100 might outperform FP16 on two A100s — at half the cost.

3. **Use spot/preemptible instances:** LLM serving is stateless (the model is read-only). Spot instances with fast restart can save 60-70% on GPU costs.

4. **Batch offline workloads:** If you have summarization or embedding jobs, batch them and run on cheaper GPUs during off-peak hours.

5. **Model routing:** Send simple queries to a small model (7-13B), complex queries to a large model (70B+). Most queries don't need the largest model.

```
Cost per 1M tokens (approximate, self-hosted, cloud GPUs):

70B FP16 on 4× A100:  ~$0.50-1.00 / 1M tokens
70B INT4 on 1× A100:  ~$0.15-0.30 / 1M tokens
8B FP16 on 1× A10G:   ~$0.05-0.10 / 1M tokens
8B INT4 on 1× A10G:   ~$0.02-0.05 / 1M tokens

vs. OpenAI GPT-4o:     ~$2.50-10.00 / 1M tokens
vs. OpenAI GPT-4o-mini: ~$0.15-0.60 / 1M tokens
```

Self-hosting becomes cost-effective when you have consistent, high-volume traffic. For low or bursty traffic, API providers are often cheaper.

---

## 5. Production Deployment Patterns

### 5.1 Single-Model Serving

The simplest pattern. One model behind a load balancer.

```
                    ┌──────────────┐
                    │ Load Balancer │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼───────┐ ┌──▼────────┐
        │ vLLM Pod 1 │ │ vLLM Pod 2│ │ vLLM Pod 3│
        │ (4× GPU)   │ │ (4× GPU)  │ │ (4× GPU)  │
        └────────────┘ └───────────┘ └───────────┘
```

### 5.2 Model Router (Multi-Model)

Route requests to different models based on complexity, cost, or capability.

```
                    ┌──────────────┐
                    │    Router     │
                    │ (LiteLLM /   │
                    │  custom)     │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
     ┌─────▼─────┐   ┌────▼────┐   ┌──────▼──────┐
     │  Small 8B  │   │ Medium  │   │  Large 70B   │
     │  (simple)  │   │  34B    │   │  (complex)   │
     │  1× GPU    │   │ 2× GPU  │   │  4× GPU      │
     └────────────┘   └─────────┘   └──────────────┘
```

**Routing logic examples:**
- Token count < 50 and simple question → Small model
- Code generation or reasoning → Large model
- Default → Medium model
- Fallback if large model is overloaded → Medium model

### 5.3 LLM Gateway Pattern

A centralized gateway that handles cross-cutting concerns:

```
Client → LLM Gateway → Model Backends
              │
              ├── Rate limiting (per user, per org)
              ├── Authentication & authorization
              ├── Request/response logging
              ├── Caching (exact match + semantic)
              ├── Cost tracking (token counting)
              ├── Fallback routing (primary → backup)
              ├── Load balancing
              ├── PII redaction
              └── Metrics & tracing
```

**Tools:** LiteLLM (open source), Portkey, Kong AI Gateway, custom FastAPI middleware.

---

## 6. Benchmarking & Performance Tuning

### 6.1 Key Metrics to Measure

| Metric | What It Measures | Target (Chat) | Target (Batch) |
|--------|-----------------|----------------|-----------------|
| TTFT | Time from request to first token | < 500ms | Don't care |
| ITL | Time between consecutive tokens | < 50ms | Don't care |
| Tokens/sec (per request) | Generation speed for a single user | > 20 tok/s | > 50 tok/s |
| Tokens/sec (aggregate) | Total throughput across all requests | Maximize | Maximize |
| Requests/sec | How many concurrent requests handled | Maximize | Maximize |
| GPU Utilization | % of GPU compute being used | > 60% | > 80% |
| GPU Memory Utilization | % of VRAM being used | 80-95% | 80-95% |
| P99 Latency | Tail latency (worst case) | < 2s TTFT | Don't care |

### 6.2 Benchmarking Tools

**vLLM's built-in benchmark:**
```bash
# Throughput benchmark (offline)
python -m vllm.entrypoints.openai.api_server --model ... &

python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --num-prompts 1000 \
    --input-len 512 \
    --output-len 128

# Serving benchmark (online, with concurrent requests)
python benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --num-prompts 500 \
    --request-rate 10 \      # 10 requests per second
    --input-len 512 \
    --output-len 128
```

**LLMPerf:** Open-source LLM benchmarking tool that measures TTFT, ITL, throughput across providers.

**Custom load testing:** Use tools like Locust or k6 to simulate realistic traffic patterns against your vLLM deployment.

### 6.3 Tuning Checklist

When optimizing a vLLM deployment, work through this sequence:

```
1. RIGHT-SIZE THE MODEL
   □ Is this the smallest model that meets quality requirements?
   □ Would quantization (AWQ/FP8) maintain acceptable quality?
   □ Benchmark: FP16 vs AWQ vs FP8 on YOUR eval set

2. OPTIMIZE GPU MEMORY
   □ Set gpu-memory-utilization to 0.90-0.95
   □ Set max-model-len to your ACTUAL maximum (not model's max)
   □ Enable prefix caching if requests share system prompts
   □ Add swap-space (4-16 GB) for burst handling

3. MAXIMIZE THROUGHPUT
   □ Increase max-num-seqs (monitor for latency degradation)
   □ Increase max-num-batched-tokens
   □ Enable chunked prefill for mixed workloads

4. REDUCE LATENCY
   □ Use CUDA graphs (disable enforce-eager)
   □ Consider speculative decoding for latency-sensitive paths
   □ Reduce tensor-parallel-size if possible (less communication overhead)

5. MONITOR AND ITERATE
   □ Track TTFT P50/P95/P99 over time
   □ Track ITL distribution
   □ Track GPU memory utilization (should be 85-95%)
   □ Track request queue depth (growing queue = under-provisioned)
   □ Alert on latency SLO violations
```

---

## 7. Hands-On Projects

### Project 1: Deploy and Benchmark vLLM
Deploy Llama 3.1 8B with vLLM in Docker. Benchmark with default settings, then systematically tune parameters. Compare FP16 vs AWQ INT4 on throughput and quality.

### Project 2: Multi-GPU Tensor Parallel Serving
Deploy Llama 3.1 70B across 4 GPUs with tensor parallelism. Understand NVLink requirements. Benchmark and compare with a 2-GPU setup.

### Project 3: Production vLLM on Kubernetes
Deploy vLLM on Kubernetes with proper health checks, resource requests, PVC for model cache, Prometheus metrics, and HPA based on GPU utilization.

### Project 4: LLM Gateway
Build a FastAPI-based LLM gateway that routes between a small (8B) and large (70B) model based on query complexity. Add caching, rate limiting, and cost tracking.

### Project 5: Quantization Comparison
Take a model. Quantize it to FP16, FP8, AWQ INT4, and GPTQ INT4. Benchmark each on throughput, latency, and quality (using an eval dataset). Document the tradeoffs.

---

> **Next:** Once you're solid on serving, move to [7.2 — RAG Architecture] to learn how to build applications on top of these served models, or to [7.3 — Fine-Tuning Infrastructure] to learn how to customize models before serving them.
