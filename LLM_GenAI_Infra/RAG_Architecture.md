# RAG Architecture (Retrieval-Augmented Generation) — Complete Study Guide

> **Context:** This is Section 7.2 of the DevOps/SRE → AI Infra / MLOps Engineer transition roadmap.
> RAG is the most common LLM application pattern in production today. If you learn one thing deeply in GenAI infrastructure, make it this.

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [Embeddings — Deep Dive](#2-embeddings--deep-dive)
3. [Chunking Strategies](#3-chunking-strategies)
4. [Vector Databases](#4-vector-databases)
5. [Retrieval Strategies](#5-retrieval-strategies)
6. [Advanced RAG Patterns](#6-advanced-rag-patterns)
7. [RAG Evaluation](#7-rag-evaluation)
8. [Production RAG Infrastructure](#8-production-rag-infrastructure)
9. [Hands-On Projects](#9-hands-on-projects)

---

## 1. Foundational Concepts

### 1.1 Why RAG Exists

Large Language Models have three fundamental limitations that RAG solves:

**Knowledge Cutoff:** Every LLM is trained on data up to a certain date. It literally does not know anything that happened after that point. If your company published a new policy yesterday, the LLM has no idea it exists.

**Hallucination:** When an LLM doesn't know something, it doesn't say "I don't know." It confidently generates plausible-sounding but completely fabricated information. In high-stakes domains (legal, medical, financial), this is unacceptable.

**No Access to Private Data:** LLMs are trained on public internet data. They have zero knowledge of your company's internal documents, Confluence pages, Slack messages, customer records, or proprietary databases.

RAG solves all three by retrieving relevant information at query time and injecting it into the prompt before the LLM generates a response.

### 1.2 The Basic RAG Pipeline

```
User Query
    │
    ▼
┌─────────────────┐
│  1. EMBED QUERY  │  Convert the user's question into a vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. RETRIEVE     │  Search vector DB for similar document chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. AUGMENT      │  Inject retrieved chunks into the LLM prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. GENERATE     │  LLM generates answer grounded in retrieved context
└─────────────────┘
```

**Step-by-step breakdown:**

1. **Embed the Query:** The user's question is converted into a dense vector (an array of floating-point numbers, e.g., 1536 dimensions) using an embedding model. This vector captures the semantic meaning of the question.

2. **Retrieve:** The query vector is compared against pre-computed vectors of all your document chunks stored in a vector database. The top-k most similar chunks are returned (typically k = 3 to 20).

3. **Augment:** The retrieved chunks are inserted into the LLM's prompt, usually in a structured format like:
   ```
   Use the following context to answer the question.
   Context:
   [chunk 1]
   [chunk 2]
   [chunk 3]
   Question: {user's original question}
   ```

4. **Generate:** The LLM generates a response grounded in the provided context, dramatically reducing hallucination and providing up-to-date, domain-specific answers.

### 1.3 RAG vs Fine-Tuning — When to Use Each

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Best for** | Adding new knowledge, accessing private data | Changing model behavior, style, format |
| **Data requirement** | Documents in any format | Curated instruction/response pairs |
| **Cost** | Lower (no training, just inference + retrieval) | Higher (GPU hours for training) |
| **Update frequency** | Real-time (just update the document store) | Requires retraining |
| **Hallucination control** | High (answers are grounded in retrieved docs) | Moderate (can still hallucinate) |
| **Latency** | Slightly higher (retrieval adds ~100-500ms) | Lower (no retrieval step) |
| **Transparency** | High (you can show source documents) | Low (knowledge is baked into weights) |

**When to combine them:** Fine-tune a model to follow your output format and domain terminology, then use RAG to inject current knowledge. This is the "best of both worlds" approach used in many production systems.

### 1.4 Naive RAG vs Advanced RAG vs Modular RAG

**Naive RAG:**
The basic pipeline described above. Simple embed → retrieve → generate. Problems: poor retrieval quality for complex queries, no handling of ambiguous questions, no verification of retrieved context relevance, rigid pipeline with no feedback loops.

**Advanced RAG:**
Adds pre-retrieval and post-retrieval optimizations:
- **Pre-retrieval:** Query rewriting, query expansion, HyDE, step-back prompting
- **Retrieval:** Hybrid search, multi-index, recursive retrieval
- **Post-retrieval:** Reranking, contextual compression, relevance filtering
- **Generation:** Citation generation, self-consistency checking

**Modular RAG:**
Treats RAG as a set of composable modules rather than a fixed pipeline. Any module can be added, removed, or replaced. Modules include: routing (decide which retrieval strategy to use), judging (evaluate if retrieval is needed at all), adapting (rewrite the query for better retrieval). Think of it as microservices architecture applied to RAG.

---

## 2. Embeddings — Deep Dive

### 2.1 What Embedding Models Do

An embedding model converts text (words, sentences, paragraphs) into a fixed-size vector of floating-point numbers. These vectors are positioned in a high-dimensional space such that semantically similar texts are close together.

```
"How do I reset my password?"  →  [0.023, -0.041, 0.089, ..., 0.012]  (1536 dims)
"I forgot my login credentials" →  [0.025, -0.038, 0.091, ..., 0.015]  (1536 dims)
"What's the weather today?"    →  [0.891, 0.234, -0.567, ..., 0.445]  (1536 dims)
```

The first two vectors will be very close together (high cosine similarity), while the third will be far away — even though they share no keywords.

**Key insight for infra engineers:** Embedding is a relatively cheap, fast operation compared to LLM generation. A single embedding API call typically takes 10–50ms and costs a fraction of a cent. The infrastructure challenge is in storing and searching billions of these vectors efficiently.

### 2.2 Popular Embedding Models

**Proprietary:**

| Model | Provider | Dimensions | Context Window | Notes |
|-------|----------|------------|----------------|-------|
| text-embedding-3-large | OpenAI | 3072 (configurable) | 8191 tokens | Supports dimension reduction via API parameter |
| text-embedding-3-small | OpenAI | 1536 (configurable) | 8191 tokens | Cheaper, slightly lower quality |
| embed-v4 | Cohere | 1024 | 512 tokens | Excellent multilingual support, input_type parameter |
| Voyage-3 | Voyage AI | 1024 | 32000 tokens | Strong for code and long documents |

**Open Source (self-hostable — important for your infra skills):**

| Model | Dimensions | Context | Notes |
|-------|------------|---------|-------|
| BGE-large-en-v1.5 (BAAI) | 1024 | 512 tokens | Top performer, Apache 2.0 license |
| BGE-M3 (BAAI) | 1024 | 8192 tokens | Multi-lingual, multi-granularity, multi-functionality |
| E5-large-v2 (Microsoft) | 1024 | 512 tokens | Requires "query:" and "passage:" prefixes |
| GTE-large (Alibaba) | 1024 | 8192 tokens | Strong all-rounder |
| Jina-embeddings-v3 | 1024 | 8192 tokens | Task-specific LoRA adapters built-in |
| nomic-embed-text-v1.5 | 768 | 8192 tokens | Fully open source including training data |

**Infra Perspective — Self-Hosting Embeddings:**
You will likely need to self-host embedding models for cost, latency, and data privacy reasons. Tools to know:
- **TEI (Text Embeddings Inference)** — Hugging Face's optimized embedding server. Supports batching, GPU acceleration, and quantization. Deploy with Docker or Kubernetes.
- **Infinity** — Another high-performance embedding server.
- **Ollama** — Supports some embedding models locally.
- Typical deployment: TEI container on a GPU node behind an internal load balancer.

### 2.3 How to Evaluate Embeddings

**MTEB (Massive Text Embedding Benchmark):**
The standard benchmark for comparing embedding models. Covers multiple tasks: retrieval, classification, clustering, pair classification, reranking, STS (semantic textual similarity), summarization. Check the MTEB leaderboard on Hugging Face before choosing a model.

**Task-Specific Evaluation:**
MTEB gives you a general ranking, but what matters is how well the model performs on YOUR data. Create a small evaluation set:
1. Take 50–100 real user queries from your domain
2. For each query, manually identify the correct document chunks
3. Run each embedding model and measure: Recall@k (did the correct chunk appear in the top-k results?), MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain)

### 2.4 Dimensionality Tradeoffs

| Dimensions | Storage per Vector | Search Speed | Quality | Use Case |
|------------|-------------------|--------------|---------|----------|
| 384 | 1.5 KB | Fastest | Good | High-volume, cost-sensitive |
| 768 | 3 KB | Fast | Better | Balanced default |
| 1024 | 4 KB | Moderate | Very Good | Most production systems |
| 1536 | 6 KB | Slower | Excellent | When quality is critical |
| 3072 | 12 KB | Slowest | Marginal gains | Rarely worth the cost |

**Practical math:** 10 million vectors at 1536 dimensions = ~60 GB of raw vector data (before index overhead). At 384 dimensions = ~15 GB. This matters for your infrastructure planning.

**Matryoshka Representation Learning (MRL):**
Some newer models (like OpenAI's text-embedding-3) support dimension truncation — you can train at 3072 dims but use only 512 or 256 at search time with minimal quality loss. This is a powerful trick for balancing cost and quality.

### 2.5 Fine-Tuning Embeddings on Domain-Specific Data

When off-the-shelf embeddings don't perform well enough on your domain (e.g., medical, legal, or highly technical content):

**Approaches:**
- **Contrastive fine-tuning:** Provide pairs of (query, positive document) and optionally (query, negative document). The model learns to bring relevant pairs closer together.
- **Tools:** Sentence-Transformers library, Hugging Face `transformers` + custom training loop
- **Data requirements:** As few as 100–500 high-quality pairs can significantly improve domain retrieval
- **Hard negatives mining:** Finding documents that are superficially similar but not actually relevant — these are the most valuable training examples

### 2.6 Multimodal Embeddings

**CLIP (Contrastive Language-Image Pre-training):**
Maps both images and text into the same vector space. You can search for images using text queries and vice versa. Useful for: product catalogs, design asset search, medical imaging + report retrieval.

**ColPali / ColQwen:**
Newer approach — embed entire document pages as images rather than extracting text. Handles tables, charts, and complex layouts much better than text-only approaches.

**Infra consideration:** Multimodal embeddings require GPU for both indexing and potentially for query embedding. Plan your GPU allocation accordingly.

---

## 3. Chunking Strategies

> This is where most RAG systems succeed or fail. Bad chunking = bad retrieval = bad answers, regardless of how good your LLM is.

### 3.1 Fixed-Size Chunking with Overlap

The simplest approach. Split text into chunks of N characters/tokens with an overlap of M characters/tokens.

```
Document: "AAAAABBBBBCCCCCDDDDDEEEEE"
Chunk size: 10, Overlap: 3

Chunk 1: "AAAAABBBBB"
Chunk 2: "BBBCCCCCDD"
Chunk 3: "DDDDDEEEE"
```

**Parameters to tune:**
- **Chunk size:** 256–1024 tokens is the sweet spot for most use cases. Smaller chunks = more precise retrieval but less context. Larger chunks = more context but noisier retrieval.
- **Overlap:** Typically 10–20% of chunk size. Prevents important information from being split across chunk boundaries.

**Pros:** Simple, predictable, fast.
**Cons:** Splits sentences and paragraphs mid-thought, ignores document structure entirely.

### 3.2 Recursive Character Splitting

LangChain's default approach. Uses a hierarchy of separators to split text:

```
Separators (in order of priority):
1. "\n\n"  (paragraph breaks)
2. "\n"    (line breaks)
3. ". "    (sentence boundaries)
4. " "     (word boundaries)
5. ""      (character level — last resort)
```

The algorithm tries to split on paragraph breaks first. If any resulting chunk is still too large, it recursively splits on the next separator level. This preserves natural text boundaries much better than fixed-size chunking.

**When to use:** Good default for unstructured text documents (articles, reports, documentation).

### 3.3 Semantic Chunking

Splits text based on meaning rather than character count. The algorithm:

1. Split text into sentences
2. Compute embeddings for each sentence
3. Calculate cosine similarity between consecutive sentences
4. When similarity drops below a threshold, insert a chunk boundary

This ensures each chunk contains a coherent "thought" or topic.

**Tools:** LlamaIndex's `SemanticSplitterNodeParser`, LangChain's `SemanticChunker`.

**Pros:** Produces semantically coherent chunks.
**Cons:** Requires embedding computation during chunking (slower, costs money), chunk sizes are variable and unpredictable.

### 3.4 Document-Structure-Aware Chunking

Leverages the inherent structure of documents:

**For Markdown/HTML:** Split on headers (H1, H2, H3), preserving the header hierarchy as metadata. Each section becomes a chunk.

**For PDFs:** Use document layout analysis (tools like `unstructured.io`, `docling`, `marker`) to identify: headers, paragraphs, tables, lists, figures, captions. Keep each structural element as a unit; never split a table across chunks.

**For Code:** Split on function/class boundaries rather than arbitrary line counts.

**This is the approach you should default to** when your documents have clear structure. It dramatically outperforms character-based splitting.

### 3.5 Agentic Chunking

Use an LLM to decide chunk boundaries. The LLM reads the document and determines where one "topic" ends and another begins.

**Approach:**
1. Feed the document to an LLM in sections
2. Ask: "Does the following text belong to the same topic as the previous chunk, or does it start a new topic?"
3. Based on the LLM's response, either append to the current chunk or start a new one

**Also used for:** Generating a summary or "proposition" for each chunk that can be embedded alongside the raw text, improving retrieval quality.

**Pros:** Best quality chunking.
**Cons:** Expensive (LLM call per chunk decision), slow, non-deterministic.

### 3.6 Chunk Size Optimization

There is no universal best chunk size. It depends on:

| Factor | Smaller Chunks (128–256 tokens) | Larger Chunks (512–1024 tokens) |
|--------|-------------------------------|-------------------------------|
| **Retrieval precision** | Higher — less noise per chunk | Lower — more irrelevant text |
| **Context completeness** | Lower — might miss surrounding context | Higher — more complete answer |
| **Best for** | Factual Q&A, specific lookups | Summarization, complex reasoning |
| **Number of chunks needed** | More chunks in top-k (use k=10–20) | Fewer chunks needed (k=3–5) |
| **Storage** | More vectors, more metadata | Fewer vectors |

**How to find your optimal size:** Create a test set of 50+ queries with known correct answers. Test chunk sizes of 128, 256, 512, 768, 1024. Measure retrieval recall and end-to-end answer quality. The right size is domain-specific.

### 3.7 Metadata Enrichment

Every chunk should carry metadata. This is critical for filtering, citation, and debugging.

**Essential metadata per chunk:**
- `source`: filename or URL
- `page_number`: where in the document this chunk came from
- `section_title`: the heading this chunk falls under
- `chunk_index`: position within the document
- `document_date`: when the source was published/updated
- `document_type`: policy, FAQ, manual, etc.

**Advanced metadata:**
- `summary`: LLM-generated summary of the chunk (stored alongside the chunk text)
- `hypothetical_questions`: LLM-generated questions this chunk could answer (embed these for better retrieval)
- `entities`: extracted named entities (people, products, dates)
- `language`: for multilingual document stores

**Why this matters for infra:** Metadata filtering happens BEFORE vector search. If a user asks "What was the 2024 Q3 revenue?", you can filter to only chunks with `document_date` in Q3 2024 before doing vector similarity search. This dramatically improves both speed and accuracy.

---

## 4. Vector Databases

### 4.1 Core Concepts You Need to Understand

**Vector Similarity Search:**
Finding the nearest neighbors to a query vector in high-dimensional space. Two main distance metrics:
- **Cosine Similarity:** Measures the angle between vectors. Most common for text embeddings. Range: -1 to 1 (1 = identical).
- **Euclidean Distance (L2):** Measures straight-line distance. Used when magnitude matters.
- **Dot Product:** Like cosine similarity but not normalized. Faster to compute.

**Approximate Nearest Neighbor (ANN) Algorithms:**
Exact nearest neighbor search is O(n) — you must compare against every vector. With millions of vectors, this is too slow. ANN algorithms trade a small amount of accuracy for massive speed improvements.

**HNSW (Hierarchical Navigable Small World):**
The most popular ANN algorithm. Think of it as a multi-layer skip list for vectors.
- **How it works:** Builds a multi-layer graph where each layer is progressively sparser. Search starts at the top (sparse) layer and navigates down to the bottom (dense) layer.
- **Parameters to tune:**
  - `M` (connections per node): Higher = better recall, more memory. Default: 16.
  - `ef_construction` (build-time beam width): Higher = better index quality, slower build. Default: 200.
  - `ef_search` (query-time beam width): Higher = better recall, slower search. Default: 100.
- **Tradeoff:** HNSW uses a lot of memory (the entire index must fit in RAM) but provides excellent query speed.

**IVF (Inverted File Index):**
Clusters vectors into buckets (Voronoi cells). At query time, only searches the nearest clusters.
- **Parameters:** `nlist` (number of clusters), `nprobe` (number of clusters to search at query time)
- **Tradeoff:** Less memory than HNSW, but typically lower recall at the same speed.

**Product Quantization (PQ):**
Compresses vectors by splitting them into sub-vectors and quantizing each. Reduces memory by 4–32x with moderate quality loss. Often combined with IVF (IVF_PQ).

**DiskANN:**
Stores the index on SSD rather than RAM. Enables searching billions of vectors on a single machine. Slightly slower than HNSW but dramatically cheaper at scale.

### 4.2 Pinecone

**What it is:** Fully managed, serverless vector database. You don't manage any infrastructure.

**Key concepts:**
- **Indexes:** The top-level container. Choose dimension size and distance metric at creation time. Cannot be changed later.
- **Namespaces:** Logical partitions within an index. Use for multi-tenancy (one namespace per customer) or separating document types.
- **Metadata filtering:** Each vector can have metadata key-value pairs. Filters are applied BEFORE vector search, narrowing the search space.
- **Hybrid search:** Pinecone supports sparse-dense vectors. Dense vectors capture semantic meaning; sparse vectors (like BM25) capture keyword matches. Combined, they give you the best of both worlds.
- **Serverless vs Pod-based:** Serverless is cheaper for variable workloads (pay per query). Pod-based gives predictable performance.

**When to choose Pinecone:** You want zero operational overhead, your team is small, you're okay with vendor lock-in, and you need to ship fast. Typical use: startup building an AI product.

**Infra considerations:** No self-hosting option. Data leaves your network. Pricing can spike with high query volumes. Limited control over performance tuning.

### 4.3 Weaviate

**What it is:** Open-source vector database with a rich feature set. Can be self-hosted or used as a managed service (Weaviate Cloud).

**Key concepts:**
- **Schema/Collections:** Define your data structure with properties and their types. Weaviate is more opinionated about schema than other vector DBs.
- **Vectorizers:** Built-in modules that automatically embed your text on insert (text2vec-openai, text2vec-transformers, etc.). You can also bring your own vectors.
- **HNSW configuration:** Per-collection tuning of `efConstruction`, `maxConnections`, `ef`, and vector cache size.
- **Multi-tenancy:** Native support for isolating data per tenant. Critical for SaaS applications.
- **Hybrid BM25 + Vector:** Built-in keyword search alongside vector search with configurable alpha weighting.
- **Generative modules:** Can call LLMs directly from queries (RAG built into the database).

**Self-hosted deployment (your focus area):**
- Docker single-node for development
- Kubernetes (Helm chart) for production
- Replication: Multi-node clusters with configurable consistency levels
- Resource planning: HNSW index is memory-bound; plan ~2–4x the raw vector size for index overhead

**When to choose Weaviate:** You need self-hosting, want built-in vectorization, need multi-tenancy, or want a feature-rich "batteries-included" solution.

### 4.4 Milvus / Zilliz

**What it is:** High-performance open-source vector database designed for massive scale. Zilliz is the managed cloud version.

**Key concepts:**
- **Collections and Partitions:** Collections hold vectors; partitions subdivide collections for filtered search efficiency.
- **Multiple index types:** More index options than any other vector DB:
  - `FLAT` — Exact search, no approximation. For small datasets or ground truth testing.
  - `IVF_FLAT` — Good balance of speed and recall for medium datasets.
  - `IVF_PQ` — Memory-efficient for large datasets, some quality loss.
  - `HNSW` — Best recall, highest memory usage.
  - `DiskANN` — For billion-scale datasets that don't fit in memory.
  - `GPU_IVF_FLAT`, `GPU_CAGRA` — GPU-accelerated indexing and search.
- **Consistency levels:** Strong, bounded staleness, session, eventual. Choose based on your read-after-write requirements.
- **Segment architecture:** Data is stored in sealed (immutable, indexed) and growing (mutable, not yet indexed) segments.

**Cluster deployment:**
- Components: Proxy, Query Node, Data Node, Index Node, Root Coord, Query Coord, Data Coord, Index Coord
- Dependencies: etcd (metadata), MinIO/S3 (storage), Pulsar/Kafka (log)
- Kubernetes deployment via Milvus Operator or Helm
- Each component can scale independently

**When to choose Milvus:** You need billion-scale vector search, want GPU acceleration, need flexible index types, or require strong consistency guarantees. It's the most "enterprise-grade" open-source option.

**Infra considerations:** More complex to operate than Weaviate or Qdrant. More components = more things to monitor and debug. Worth it at scale; overkill for <1M vectors.

### 4.5 pgvector

**What it is:** A PostgreSQL extension that adds vector similarity search to your existing Postgres database.

**Key concepts:**
- **Vector column type:** `embedding vector(1536)` — add a vector column to any table.
- **Index types:**
  - `ivfflat` — IVF-based, faster to build, lower recall
  - `hnsw` — Higher recall, slower to build, more memory
- **Performance tuning:**
  - `maintenance_work_mem` — Increase for faster index builds (e.g., `2GB` for million-scale)
  - `ef_construction` (HNSW) — Higher = better quality index (default: 64, try 128–256)
  - `m` (HNSW) — Connections per node (default: 16, try 32–64)
  - `probes` (IVF) — Clusters to search at query time (default: 1, try 10–50)
- **Hybrid queries:** The killer feature — combine vector search with standard SQL WHERE clauses, JOINs, aggregations in a single query.

**When pgvector is "good enough":**
- < 5 million vectors
- You already run PostgreSQL
- You need strong ACID guarantees
- Your data has rich relational structure alongside vectors
- You want to avoid adding another database to your stack

**When to graduate to a dedicated vector DB:**
- > 10 million vectors and growing
- You need sub-10ms query latency at scale
- You need advanced features like multi-tenancy, GPU search, or DiskANN
- Your query volume exceeds what a single Postgres instance can handle

**Infra perspective:** pgvector is the pragmatic choice. For most early-stage RAG systems, it's all you need. Your existing Postgres monitoring, backup, and HA setup (patroni, CloudNativePG) all apply.

### 4.6 Other Vector Databases Worth Knowing

**Qdrant:**
- Written in Rust, strong performance
- Rich filtering with payload indexing
- Named vectors (store multiple vector types per point)
- Good documentation, growing community
- Self-hosted via Docker/K8s, or Qdrant Cloud

**Chroma:**
- Lightweight, embedded vector DB
- Perfect for prototyping and development
- Can run in-memory or with persistent storage
- Not suitable for production at scale

**Redis Vector Search (Redis Stack):**
- If you already run Redis, add vector search capability
- Good for low-latency, smaller-scale use cases
- Flat and HNSW index support

### 4.7 Vector Database Selection Matrix

| Requirement | Best Choice |
|-------------|-------------|
| Zero ops, ship fast | Pinecone |
| Self-hosted, feature-rich | Weaviate |
| Billion-scale, max performance | Milvus |
| Already using Postgres, < 5M vectors | pgvector |
| Rust-based, clean API | Qdrant |
| Prototyping / local development | Chroma |
| Already using Redis | Redis Vector Search |

---

## 5. Retrieval Strategies

### 5.1 Hybrid Search (Dense + Sparse)

**The problem with pure vector search:** Semantic search is great for finding conceptually similar documents, but it can miss exact keyword matches. If a user asks about "error code E-4012", vector search might return chunks about error handling in general, missing the specific error code.

**The problem with pure keyword search (BM25):** Keyword search finds exact matches but misses semantic equivalents. "How to fix authentication issues" won't match a document about "resolving login failures."

**Hybrid search combines both:**

```
Final Score = α × vector_score + (1 - α) × bm25_score
```

Where α (alpha) is a weighting parameter:
- α = 1.0 → Pure vector search
- α = 0.0 → Pure keyword search
- α = 0.5 → Equal weighting (common starting point)
- α = 0.7 → Favor semantic (good default for most Q&A)

**Implementation options:**
- Weaviate: Built-in hybrid search with configurable alpha
- Pinecone: Sparse-dense vectors in the same index
- Qdrant: Supports sparse vectors alongside dense
- DIY: Run BM25 (Elasticsearch/OpenSearch) and vector search separately, merge results with Reciprocal Rank Fusion (RRF)

**Reciprocal Rank Fusion (RRF):**
A simple, effective way to merge ranked lists from different search systems:
```
RRF_score(doc) = Σ 1 / (k + rank_in_list_i)
```
Where k is a constant (typically 60). This doesn't require score normalization, making it practical for combining results from different systems.

### 5.2 Reranking

**The concept:** First-stage retrieval (vector search) is fast but approximate. A reranker is a more expensive but more accurate model that re-scores the top-k retrieved documents.

```
Query → Vector Search (retrieve top 50) → Reranker (re-score, return top 5) → LLM
```

**Why rerankers are more accurate:**
- Embedding models process query and document independently (bi-encoder). They can't capture fine-grained query-document interactions.
- Rerankers process query and document together (cross-encoder). They can attend to specific interactions between query terms and document terms.

**Popular rerankers:**
- **Cohere Rerank:** API-based, very easy to use, consistently strong
- **BGE-reranker-v2-m3:** Open source, multilingual, self-hostable
- **FlashRank:** Lightweight, fast, good for latency-sensitive applications
- **Jina Reranker:** Good balance of speed and quality
- **ColBERT:** Late-interaction model, faster than cross-encoders with comparable quality

**Practical guidance:**
- Retrieve 20–50 candidates with vector search (fast, cheap)
- Rerank to top 3–10 with a cross-encoder (slower, more accurate)
- The reranking step typically adds 50–200ms latency
- The quality improvement is significant — often 10–30% better retrieval metrics

### 5.3 Multi-Query Retrieval

**The problem:** A single user query might not capture all aspects of what they're looking for. "Tell me about the company's sustainability initiatives" is a broad query that might miss specific topics like carbon offsetting, supply chain ethics, or renewable energy usage.

**The approach:**
1. Use an LLM to generate 3–5 variations of the original query
2. Run vector search for each variation
3. Take the union of all results (deduplicated)

**Example:**
```
Original: "What is our parental leave policy?"

Generated variations:
1. "maternity leave benefits and duration"
2. "paternity leave policy for new fathers"
3. "parental leave eligibility requirements"
4. "family leave and childcare support policies"
```

Each variation retrieves different relevant chunks, giving broader coverage.

**Tradeoff:** More retrieval calls (cost + latency) for better recall. Typically 3–5 variations is the sweet spot.

### 5.4 HyDE (Hypothetical Document Embeddings)

**The insight:** Sometimes the query and the answer are expressed very differently. "Why is the sky blue?" has low embedding similarity to "Rayleigh scattering causes shorter wavelengths of light to scatter more than longer wavelengths..."

**The approach:**
1. Ask the LLM to generate a hypothetical answer to the query (without any context)
2. Embed the hypothetical answer (not the original query)
3. Use that embedding to search the vector database

The hypothetical answer, even if factually wrong, will be phrased more similarly to the actual documents in your database.

**When it works well:** Technical documentation, academic content, domain-specific language.
**When it doesn't work:** When the LLM's hypothetical answer is so far off that it misleads retrieval. Factual or very specific queries (dates, names, codes) don't benefit.

### 5.5 Parent-Document Retrieval

**The problem:** Small chunks give precise retrieval, but the LLM often needs surrounding context to generate a good answer.

**The approach:**
1. Store documents at two levels: small chunks (for retrieval) and parent documents/sections (for context)
2. Search is done against the small chunks
3. When a small chunk is retrieved, return its parent document/section to the LLM

```
Parent Document (full section, ~2000 tokens)
├── Child Chunk 1 (256 tokens)  ← matched by vector search
├── Child Chunk 2 (256 tokens)
└── Child Chunk 3 (256 tokens)

Search matches Child Chunk 1 → Return entire Parent Document to LLM
```

**Implementation:** Store chunks with a `parent_id` reference. On retrieval, fetch the parent. LlamaIndex has this built-in as `AutoMergingRetriever`.

### 5.6 Contextual Compression

**The problem:** Retrieved chunks often contain a lot of irrelevant text alongside the relevant portion. Sending all of this to the LLM wastes tokens and can confuse the model.

**The approach:** After retrieval, use an LLM (or a smaller model) to extract only the relevant sentences from each chunk, given the query.

```
Query: "What is the return policy for electronics?"

Retrieved chunk (500 tokens):
"Our store is open Monday through Friday, 9am to 5pm. We offer a wide
range of electronics including TVs, laptops, and smartphones. For
electronics, customers may return items within 30 days of purchase with
original receipt. Items must be in original packaging. Restocking fees
may apply for opened items..."

Compressed (50 tokens):
"Electronics may be returned within 30 days with original receipt.
Items must be in original packaging. Restocking fees may apply for
opened items."
```

**Tradeoff:** Additional LLM call adds latency and cost, but the LLM's final answer is better because the context is more focused.

### 5.7 Multi-Index Strategies

**The concept:** Different types of documents may need different retrieval strategies. Route queries to the appropriate index based on intent.

**Example architecture:**
```
User Query
    │
    ▼
┌──────────────┐
│ Query Router  │ (LLM or classifier)
└──────┬───────┘
       │
  ┌────┼────────────┐
  │    │             │
  ▼    ▼             ▼
FAQ   Technical    Policy
Index  Docs Index   Index
(small chunks,  (large chunks,  (full document,
fast retrieval)  code-aware)     metadata-filtered)
```

**Routing approaches:**
- **LLM-based routing:** Ask the LLM to classify the query intent and pick the index
- **Embedding-based routing:** Embed the query and compare against representative embeddings for each index
- **Keyword-based routing:** Simple regex/keyword rules for obvious cases

---

## 6. Advanced RAG Patterns

### 6.1 Self-RAG

**The concept:** The model itself decides:
1. Whether retrieval is needed for a given query
2. Which retrieved passages are relevant
3. Whether its generated response is supported by the evidence

**How it works:**
The model generates special "reflection tokens" during generation:
- `[Retrieve]` → Yes/No — should I retrieve for this query?
- `[IsRel]` → Is this retrieved passage relevant?
- `[IsSup]` → Is my response supported by the retrieved evidence?
- `[IsUse]` → Is my response useful to the user?

**Infra implications:** Requires a fine-tuned model (not just prompt engineering). The original Self-RAG paper fine-tunes Llama 2. More complex to deploy but reduces unnecessary retrieval calls and improves output quality.

### 6.2 Corrective RAG (CRAG)

**The concept:** After retrieval, evaluate whether the retrieved documents are good enough. If not, take corrective action.

**The pipeline:**
```
Query → Retrieve → Evaluate Retrieval Quality
                        │
              ┌─────────┼─────────┐
              │         │          │
          Correct    Ambiguous   Incorrect
              │         │          │
         Use docs   Refine +    Web search
         as context  retrieve    fallback
                    again
```

**Evaluation methods:**
- Use a lightweight classifier to score retrieval relevance
- Set confidence thresholds: > 0.8 = correct, 0.4–0.8 = ambiguous, < 0.4 = incorrect
- For ambiguous results, rewrite the query and try again
- For incorrect results, fall back to web search or inform the user

### 6.3 Graph RAG

**The concept:** Instead of (or in addition to) chunking documents and doing vector search, build a knowledge graph from your documents and use graph traversal for retrieval.

**How it works:**
1. **Entity extraction:** Use an LLM to extract entities and relationships from documents
2. **Graph construction:** Build a knowledge graph where nodes = entities, edges = relationships
3. **Community detection:** Identify clusters of related entities
4. **Community summarization:** Generate summaries for each community
5. **Query:** For a query, identify relevant communities and use their summaries as context

**When Graph RAG shines:**
- Complex queries that require connecting information across multiple documents
- "What are the main themes across all our customer feedback?"
- Questions about relationships: "Who works with whom on what?"
- When simple chunk retrieval doesn't capture the big picture

**Tools:** Microsoft's `graphrag` library, Neo4j + LangChain integration, LlamaIndex's `KnowledgeGraphIndex`.

**Infra considerations:** Graph construction is expensive (many LLM calls). The graph needs to be rebuilt when documents change. Graph databases (Neo4j, Amazon Neptune) add another component to manage.

### 6.4 Agentic RAG

**The concept:** An AI agent autonomously decides how to retrieve information, performing multi-step retrieval when needed.

**Example flow:**
```
User: "Compare our Q3 2024 revenue with our top competitor's Q3 2024 revenue"

Agent thinks: I need two pieces of information
  Step 1: Search internal docs for "Q3 2024 revenue" → Found: $45M
  Step 2: The internal docs don't have competitor data
  Step 3: Search web for "competitor Q3 2024 revenue" → Found: $52M
  Step 4: Synthesize comparison
```

**Key capabilities:**
- **Tool selection:** The agent can choose between different retrieval tools (internal vector DB, web search, SQL database, API calls)
- **Query decomposition:** Breaking complex queries into sub-queries
- **Iterative refinement:** If the first retrieval doesn't answer the question, try different queries
- **Source verification:** Cross-referencing information from multiple sources

**Frameworks:** LangChain Agents, LlamaIndex Agents, custom ReAct loops.

### 6.5 Multi-Modal RAG

**The problem:** Documents contain tables, charts, diagrams, and images that carry critical information. Text-only RAG misses all of this.

**Approaches:**

**Option A — Extract and convert to text:**
- Use table extraction (Camelot, Tabula, Unstructured) to convert tables to markdown/CSV
- Use OCR for images with text
- Use image captioning models to describe visual content
- Index the extracted text alongside regular text chunks

**Option B — Embed natively as images:**
- Use multimodal embedding models (ColPali, ColQwen) to embed entire pages as images
- Retrieve based on image embeddings
- Send the retrieved page images directly to a multimodal LLM (GPT-4o, Claude) for reasoning

**Option C — Hybrid:**
- Extract what you can as structured text (tables, lists)
- Embed remaining visual content as images
- Use both text and image retrieval depending on the query

**Infra considerations:** Multi-modal RAG requires more storage (images are larger), more compute (vision models for embedding/processing), and more complex pipelines.

---

## 7. RAG Evaluation

### 7.1 What to Evaluate

RAG evaluation happens at two levels:

**Retrieval quality (did we find the right documents?):**
- **Context Precision:** What fraction of retrieved chunks are actually relevant?
- **Context Recall:** Did we retrieve all the chunks that are relevant?
- **Mean Reciprocal Rank (MRR):** How high in the ranked list does the first relevant result appear?
- **NDCG@k:** Measures ranking quality of the top-k results

**Generation quality (did the LLM produce a good answer?):**
- **Faithfulness:** Is the answer supported by the retrieved context? (No hallucination)
- **Answer Relevancy:** Does the answer actually address the question?
- **Answer Correctness:** Is the answer factually correct? (Requires ground truth)

### 7.2 Evaluation Frameworks

**RAGAS (Retrieval-Augmented Generation Assessment):**
The most popular RAG evaluation framework. Provides automated metrics:
- Faithfulness — uses LLM to check if each claim in the answer is supported by context
- Answer Relevancy — generates questions from the answer and checks if they match the original
- Context Precision — evaluates if relevant context is ranked higher
- Context Recall — checks if ground truth can be attributed to context
- Answer Correctness — compares against a reference answer

**DeepEval:**
Another evaluation framework with:
- Similar metrics to RAGAS
- Hallucination detection
- Toxicity and bias evaluation
- Conversation-level evaluation (for multi-turn RAG)
- Integration with CI/CD pipelines

**Promptfoo:**
More focused on prompt and configuration testing:
- Define test cases with expected outputs
- Run across multiple configurations (chunk sizes, embedding models, retrieval strategies)
- Built-in assertion types: contains, similar, LLM-graded
- Great for A/B testing RAG configurations

### 7.3 Building Your Evaluation Dataset

**The Golden Test Set:**
Create a set of 50–200 question-answer pairs with:
- The question
- The expected answer (ground truth)
- The source document(s) that contain the answer

**Types of questions to include:**
- Simple factual lookups: "What is the vacation policy?"
- Multi-document synthesis: "Compare policy A with policy B"
- Questions requiring reasoning: "Am I eligible for benefit X given condition Y?"
- Questions with no answer in the corpus: "What is the company's policy on time travel?" (should say "I don't know")
- Ambiguous questions: "Tell me about the policy" (should ask for clarification or handle gracefully)

**Automate evaluation in CI/CD:**
When you change chunking strategy, embedding model, or retrieval logic, automatically run your test suite and compare metrics against the previous version. Treat it like a regression test suite.

---

## 8. Production RAG Infrastructure

### 8.1 End-to-End Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                     │
│                                                           │
│  Documents → Parser → Chunker → Embedder → Vector DB     │
│  (S3/GCS)  (Unstructured/  (Your      (TEI/         (Weaviate/ │
│             Docling)       strategy)   OpenAI API)    Milvus)   │
│                                                           │
│  Orchestrated by: Airflow / Dagster / Prefect             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                         │
│                                                           │
│  User Query → [Query Rewrite] → Embed → Search → [Rerank]│
│            → Augment Prompt → LLM → Response              │
│                                                           │
│  Served by: FastAPI / LangServe behind a load balancer    │
│  Traced by: LangSmith / Langfuse / Arize Phoenix          │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Ingestion Pipeline Considerations

- **Document parsing:** Use `unstructured.io` or `docling` for handling PDFs, DOCX, PPTX, HTML. Handle failures gracefully — some documents will fail parsing.
- **Incremental ingestion:** Don't re-embed all documents when one changes. Track document hashes and only re-process modified documents.
- **Embedding throughput:** Batch embedding calls. TEI supports batching natively. OpenAI's API can process up to 2048 inputs per request.
- **Backfill strategy:** When you change your chunking strategy or embedding model, you need to re-process everything. Plan for this.
- **Deduplication:** Multiple documents may contain the same content. Deduplicate at the chunk level to avoid retrieval redundancy.

### 8.3 Operational Concerns

**Scaling:**
- Vector DB: Horizontal scaling (Milvus, Weaviate support sharding). Monitor query latency and memory usage.
- Embedding service: Scale based on ingestion throughput and query volume. GPU instances for self-hosted models.
- LLM serving: Separate concern (see Section 7.1), but the bottleneck is usually here.

**Monitoring (your SRE skills shine here):**
- Retrieval latency (p50, p95, p99)
- Vector DB query throughput and saturation
- Embedding service latency and error rates
- Reranker latency
- End-to-end RAG pipeline latency (query to response)
- Retrieval quality metrics over time (track faithfulness/relevancy scores)
- Token usage and cost per query
- Cache hit rates

**Failure modes to handle:**
- Vector DB unavailable → Fallback to direct LLM response (with disclaimer)
- Embedding service down → Queue ingestion, serve cached results for queries
- No relevant results found → Don't hallucinate; respond with "I don't have information about that"
- Retrieved context too large for LLM context window → Truncation strategy

---

## 9. Hands-On Projects

### Project 1: Basic RAG System
Build a Q&A system over a set of PDFs using LangChain, OpenAI embeddings, and Chroma. Focus on getting the full pipeline working end-to-end.

### Project 2: Production RAG with Evaluation
Upgrade to pgvector or Weaviate. Add hybrid search, reranking (Cohere Rerank), and build an evaluation pipeline with RAGAS. Deploy with Docker Compose.

### Project 3: Advanced RAG on Kubernetes
Deploy Milvus or Weaviate on Kubernetes. Self-host embeddings with TEI. Build an ingestion pipeline with Airflow. Add observability with Langfuse. Implement multi-index routing and agentic retrieval.

### Project 4: Multi-Modal RAG
Build a RAG system that handles PDFs with tables and charts. Use Unstructured.io for parsing, ColPali for page-level embeddings, and a multi-modal LLM for generation.

---

> **Next:** Once you're solid on RAG, move to [7.1 — LLM Serving & Optimization] to learn how to serve the models that power your RAG pipeline.
