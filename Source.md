# YouTube Learning Guide: DevOps/SRE → AI Infra / MLOps Engineer

> A curated list of the best free YouTube videos and channels, organized by phase, to cover every topic in your transition roadmap.

---

## Table of Contents

1. [Phase 1: ML Fundamentals](#phase-1-ml-fundamentals)
2. [Phase 2: Deep Learning & Neural Networks](#phase-2-deep-learning--neural-networks)
3. [Phase 3: Data Engineering for ML](#phase-3-data-engineering-for-ml)
4. [Phase 4: ML Training Infrastructure](#phase-4-ml-training-infrastructure)
5. [Phase 5: Model Packaging, Serving & Inference](#phase-5-model-packaging-serving--inference)
6. [Phase 6: MLOps Pipelines & Automation](#phase-6-mlops-pipelines--automation)
7. [Phase 7: LLM & GenAI Infrastructure](#phase-7-llm--genai-infrastructure)
8. [Phase 8: Monitoring, Observability & Reliability for ML](#phase-8-monitoring-observability--reliability-for-ml)
9. [Must-Follow YouTube Channels](#must-follow-youtube-channels)
10. [Bonus: Books to Read Alongside](#bonus-books-to-read-alongside)

---

## Phase 1: ML Fundamentals

### Core ML Theory & Intuition

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **Machine Learning for Everybody – Full Course** | freeCodeCamp | ~10 hrs | Complete beginner-friendly ML course covering supervised, unsupervised, evaluation metrics, and the full ML workflow |
| **StatQuest: Machine Learning** (full playlist) | StatQuest with Josh Starmer | ~30+ videos | The absolute best for building intuition. Each video explains ONE concept with crystal-clear visuals — regression, classification, bias-variance, cross-validation, regularization, all of it |
| **Machine Learning Specialization** (lectures on YouTube) | Andrew Ng / Stanford | ~33 hrs | The gold standard. Andrew Ng explains ML fundamentals from linear regression to recommender systems with mathematical precision but beginner-friendly delivery |
| **ML for Beginners** | Microsoft Developer | Playlist | Practical, project-based approach to learning ML fundamentals |

### Specific Topics to Search For on StatQuest

- "StatQuest Linear Regression" — how models learn relationships
- "StatQuest Logistic Regression" — classification fundamentals
- "StatQuest Random Forests" — ensemble methods
- "StatQuest Cross Validation" — how to evaluate properly
- "StatQuest Bias Variance Tradeoff" — the most important ML concept
- "StatQuest Regularization" — preventing overfitting (Ridge, Lasso)
- "StatQuest ROC and AUC" — evaluation metrics explained

---

## Phase 2: Deep Learning & Neural Networks

### Neural Network Foundations (START HERE)

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **Neural Networks** (4-part series) | 3Blue1Brown | ~1 hr total | Stunning visual explanations of what neural networks actually do, how backpropagation works, and gradient descent. The best visual introduction to deep learning ever made |
| **Neural Networks: Zero to Hero** (full playlist) | Andrej Karpathy | ~13 hrs | Build neural networks from scratch in Python — from micrograd (backpropagation engine) all the way to building a GPT. This is the single most recommended deep learning course on the internet. Former OpenAI research lead teaches you like no one else can |
| **Deep Learning Specialization** (lectures) | Andrew Ng / DeepLearning.AI | ~20+ hrs | Comprehensive coverage of CNNs, RNNs, sequence models, and optimization strategies |

### Transformer Architecture (Critical for LLM work)

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **Attention Is All You Need** (paper explained) | Yannic Kilcher | ~45 min | Deep walkthrough of the original Transformer paper, explained by a PhD researcher who makes dense papers accessible |
| **Transformer Neural Networks Explained** | Umar Jamil | ~1.5 hrs | The most thorough visual and code-level explanation of the Transformer architecture — attention mechanism, positional encoding, multi-head attention, all with code |
| **Let's Build GPT from Scratch** | Andrej Karpathy | ~2 hrs | Build a GPT model from scratch in Python. After watching this, you'll understand exactly how LLMs work at the code level |
| **Illustrated Guide to Transformers** | 3Blue1Brown | ~27 min | Beautiful visual explanation of how attention works |

### GPU & Hardware Fundamentals

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **How GPU Computing Works** | GTC / NVIDIA | ~30 min | Understand GPU architecture, CUDA cores, and why GPUs are essential for deep learning |
| **GPUs Explained** | Fireship | ~12 min | Quick, high-density overview of GPU computing concepts |
| **PyTorch in 100 Seconds** | Fireship | ~2 min | Ultra-quick intro before diving deeper |
| **PyTorch for Deep Learning — Full Course** | freeCodeCamp / Daniel Bourke | ~25 hrs | Comprehensive hands-on PyTorch course — the framework you'll use daily |

---

## Phase 3: Data Engineering for ML

### Data Pipelines & Orchestration

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **Apache Airflow Tutorial for Beginners** | freeCodeCamp | ~8 hrs | Full Airflow course — the most widely used ML pipeline orchestrator |
| **Apache Airflow Full Course** | TupleSpectra | ~4 hrs | Concentrated Airflow tutorial with practical DAGs |
| **Apache Spark Full Course** | freeCodeCamp | ~7 hrs | Learn Spark for large-scale data processing — essential for ML data pipelines |
| **Data Engineering Zoomcamp** (full playlist) | DataTalksClub | ~40+ hrs | Free end-to-end data engineering course covering everything from data warehouses to orchestration with Airflow, dbt, and Spark |

### Feature Stores & Data Versioning

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **What is a Feature Store?** | Feast (official) | ~20 min | Introduction to feature stores from the creators of Feast, the most popular open-source feature store |
| **DVC Tutorial — Data Version Control** | DVC (official channel) | Playlist | Learn data versioning — "Git for data" — essential for reproducible ML |
| **Great Expectations Tutorial** | Various | ~30 min | Data quality validation for ML pipelines |

---

## Phase 4: ML Training Infrastructure

### Distributed Training

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **Distributed Training with PyTorch** | PyTorch (official) | ~45 min | Official guide to distributed data parallel (DDP) training |
| **DeepSpeed Explained** | Microsoft Developer | ~30 min | Understanding ZeRO optimization stages for training large models |
| **How Distributed Training Works** | Weights & Biases | ~20 min | Clear explanation of data parallelism vs model parallelism |
| **FSDP (Fully Sharded Data Parallel) Tutorial** | PyTorch (official) | ~30 min | PyTorch's native solution for distributed training |

### Experiment Tracking

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **MLflow Tutorial** | freeCodeCamp | ~3 hrs | Complete MLflow course — experiment tracking, model registry, deployment |
| **Weights & Biases 101** | Weights & Biases | Playlist | Learn W&B from the creators — experiment tracking, sweeps, artifacts |
| **MLflow vs W&B vs Neptune Comparison** | Various | ~20 min | Understanding the experiment tracking landscape |

### GPU/Kubernetes for ML

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **Kubernetes for ML** | CNCF | Various talks | Conference talks on running ML workloads on K8s |
| **NVIDIA GPU Operator on Kubernetes** | NVIDIA Developer | ~30 min | Setting up GPU support in Kubernetes clusters |
| **Kubeflow Tutorial** | freeCodeCamp / Various | ~2 hrs | ML platform on Kubernetes — training, pipelines, serving |

---

## Phase 5: Model Packaging, Serving & Inference

### Model Serving Frameworks

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **vLLM: Easily Deploying & Serving LLMs** | NeuralNine | ~25 min | Practical vLLM tutorial covering setup, deployment, and the OpenAI-compatible API |
| **NVIDIA Triton Inference Server Tutorial** | NVIDIA Developer | ~1 hr | Official tutorial on Triton — the enterprise-grade multi-framework model server |
| **BentoML Tutorial** | BentoML (official) | Playlist | Learn to package and serve ML models as production APIs |
| **TorchServe Tutorial** | PyTorch (official) | ~30 min | PyTorch's native model serving framework |

### Inference Optimization

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **LLM Quantization Explained (GPTQ, AWQ, GGUF)** | Various | ~30 min | Understanding model compression techniques — critical for production serving |
| **ONNX Runtime Explained** | Microsoft Developer | ~20 min | Cross-framework model optimization |
| **What is Model Quantization?** | StatQuest / Various | ~15 min | The concept of reducing model precision for faster inference |

---

## Phase 6: MLOps Pipelines & Automation

### End-to-End MLOps

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **MLOps Zoomcamp** (full playlist) | DataTalksClub (Alexey Grigorev) | ~40+ hrs | **THE SINGLE BEST FREE MLOps COURSE.** Covers experiment tracking, orchestration, model deployment, monitoring — with hands-on homework and a capstone project. Runs annually with a community of 70k+ learners |
| **End-to-End Machine Learning: EDA to MLOps** | freeCodeCamp | ~3 hrs | Build a full ML project with ZenML and MLflow — from data exploration to deployment |
| **MLOps: YouTube Sentiment Analyzer** | freeCodeCamp | ~3 hrs | Learn MLOps by building a real project — Chrome extension that analyzes YouTube comment sentiment with a full pipeline |
| **MLOps Course — Production Grade Projects** | freeCodeCamp | ~3 hrs | Hands-on MLOps with ZenML, MLflow, customer satisfaction prediction project |
| **Made With ML — MLOps Course** | Goku Mohandas | Online (videos + text) | Comprehensive MLOps curriculum combining ML with software engineering — design, develop, deploy, iterate |

### CI/CD for ML

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **CI/CD for Machine Learning** | GitHub / Various | ~30 min | GitHub Actions for ML — automated testing, training, deployment |
| **ML Pipeline with GitHub Actions** | Various | ~20 min | Practical CI/CD pipelines for ML models |
| **ZenML Pipelines Tutorial** | ZenML (official) | Playlist | Building reproducible ML pipelines |

---

## Phase 7: LLM & GenAI Infrastructure

> This is the highest-demand phase. Spend extra time here.

### 7.1 RAG (Retrieval-Augmented Generation)

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **RAG from Scratch** (full playlist, ~14 videos) | LangChain (official) | ~5 hrs total | The most comprehensive RAG tutorial series — covers basic RAG, multi-query, RAG fusion, decomposition, step-back prompting, HyDE, routing, query structuring, indexing, CRAG, self-RAG, adaptive RAG |
| **RAG + LangChain Full Tutorial** | freeCodeCamp | ~3+ hrs | Build a complete RAG application from scratch |
| **Complete RAG Tutorial 2026 (Free Labs)** | Kode (YouTube) | ~2 hrs | Up-to-date crash course on RAG with hands-on labs |
| **RAG Explained — Retrieval Augmented Generation** | IBM Technology | ~10 min | Clean, concise explanation of RAG concepts for quick understanding |
| **RAG Pipeline Explained (Embeddings, HNSW, Vector DB)** | Umar Jamil | ~1.5 hrs | Deep technical dive into the RAG pipeline including the HNSW algorithm and how vector databases work under the hood |

### 7.2 Vector Databases

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **Vector Databases Simply Explained** | Fireship | ~8 min | Quick, high-density intro to vector DBs and embeddings |
| **Embeddings & Similarity Search with Qdrant** | Qdrant (official) | ~30 min | Hands-on demo of vector embeddings and similarity search |
| **Pinecone Tutorial** | James Briggs | Playlist | Series on building with Pinecone — the most popular managed vector DB |
| **pgvector Tutorial — Postgres for AI** | Various | ~30 min | Using PostgreSQL as a vector database — pragmatic for infra engineers |
| **Weaviate Tutorial** | Weaviate (official) | Playlist | Self-hosted vector database with Kubernetes deployment guides |

### 7.3 LLM Serving & Optimization

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **vLLM Tutorial** | NeuralNine | ~25 min | Practical vLLM deployment and serving |
| **PagedAttention & vLLM Explained** | Various | ~20 min | How vLLM's core innovation solves KV cache memory fragmentation |
| **TensorRT-LLM Tutorial** | NVIDIA Developer | ~45 min | NVIDIA's optimized inference engine for maximum throughput |
| **LLM Inference Optimization** | Weights & Biases | ~30 min | Overview of quantization, batching, and serving strategies |
| **Quantization Explained (INT8, INT4, GPTQ, AWQ)** | Various | ~30 min | Understanding model compression for production inference |

### 7.4 Fine-Tuning LLMs

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **Fine-Tune ANY LLM with LLaMA Factory (LoRA + QLoRA)** | Trelis Research | ~1 hr | The most complete LLaMA Factory tutorial — fine-tune any model with WebUI or CLI |
| **QLoRA Explained and Fine-Tuning Tutorial** | Trelis Research | ~45 min | Detailed QLoRA explanation with hands-on fine-tuning |
| **QLoRA — Fine-Tune an LLM on a Single GPU (with Code)** | Shaw Talebi (Data Entrepreneurs) | ~1 hr | Practical QLoRA implementation with Python code |
| **LoRA Explained** | Umar Jamil | ~1 hr | Deep mathematical and visual explanation of how LoRA works |
| **RLHF and DPO Explained** | Umar Jamil | ~1 hr | Alignment techniques — understanding how models are fine-tuned for safety and helpfulness |
| **DeepLearning.AI Short Courses** (LLM fine-tuning) | DeepLearning.AI | ~1-2 hrs each | Short, focused courses on fine-tuning with Hugging Face, evaluation, and prompt engineering — taught by industry experts |

### 7.5 LLM Agents & Orchestration

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **LangChain Full Course** | freeCodeCamp | ~3+ hrs | Complete LangChain course — chains, agents, tools, memory |
| **LangGraph Tutorial** | LangChain (official) | Playlist | Building stateful, multi-step agent workflows |
| **LlamaIndex Tutorial** | LlamaIndex (official) | Playlist | Data framework for RAG — indexes, query engines, agents |
| **AI Agents Course** | Weights & Biases | ~2 hrs (free) | Design, build, and evaluate production-grade AI agents |
| **Building AI Agents from Scratch** | Sam Witteveen | Playlist | Practical agent-building tutorials with multiple frameworks |
| **What is MCP (Model Context Protocol)?** | Various | ~15-20 min | Understanding the emerging standard for LLM-tool integration |

### 7.6 LLM Observability & Guardrails

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **LangSmith Tutorial** | LangChain (official) | ~30 min | Tracing, debugging, and evaluating LLM applications |
| **Langfuse — Open Source LLM Observability** | Langfuse (official) | ~20 min | Self-hosted alternative to LangSmith |
| **Prompt Engineering Course** | Weights & Biases | ~2 hrs (free) | System prompts, structural techniques, and model-specific strategies |
| **LLM Evaluation Frameworks** | Various | ~30 min | RAGAS, DeepEval, Promptfoo for testing RAG and LLM apps |

---

## Phase 8: Monitoring, Observability & Reliability for ML

| Video / Playlist | Creator | Duration | Why Watch It |
|---|---|---|---|
| **ML Model Monitoring — Data Drift & Concept Drift** | Various | ~30 min | Understanding why models degrade in production |
| **Evidently AI Tutorial** | Evidently AI (official) | Playlist | Open-source ML monitoring — data drift, model performance, data quality |
| **GPU Monitoring with DCGM** | NVIDIA Developer | ~20 min | Monitoring GPU utilization, memory, temperature in production |
| **SRE for ML Systems** | Google Cloud / Various | Conference talks | Applying SRE principles to ML infrastructure |
| **Prometheus + Grafana for ML** | Various | ~45 min | Setting up observability for ML serving infrastructure |

---

## Must-Follow YouTube Channels

### Tier 1 — Watch Everything (Directly Relevant)

| Channel | What They Cover | Why Essential |
|---|---|---|
| **Andrej Karpathy** | Neural networks, GPT, deep learning from scratch | The best teacher in deep learning. Period. Former OpenAI, Tesla AI lead |
| **3Blue1Brown** | Math, neural networks, linear algebra, calculus | Unmatched visual explanations of the math behind ML |
| **StatQuest with Josh Starmer** | Statistics, ML algorithms, deep learning | Every ML concept explained with zero assumptions. Beloved by the community |
| **DeepLearning.AI** | Short courses on LLMs, RAG, fine-tuning, agents, MLOps | Andrew Ng's channel — structured 1-2 hour courses taught by industry experts. Covers the latest topics rapidly |
| **Weights & Biases** | MLOps, experiment tracking, LLM engineering, RAG | Free courses on RAG, prompt engineering, agents, and LLM engineering |
| **freeCodeCamp** | Full courses on ML, MLOps, PyTorch, LangChain | 3-10 hour comprehensive courses, completely free |

### Tier 2 — Watch Selectively (Deep Dives)

| Channel | What They Cover | Why Follow |
|---|---|---|
| **Yannic Kilcher** | AI research paper breakdowns | 45-60 min deep dives into landmark papers. PhD-level explanations made accessible |
| **Umar Jamil** | Transformers, RAG, LoRA, CLIP, diffusion models | The most thorough visual + code explanations of GenAI architectures |
| **James Briggs** | Pinecone, RAG, vector databases, LangChain | Practical tutorials focused on building with vector DBs and RAG |
| **Sam Witteveen** | LangChain, agents, LLM applications | One of the earliest and most prolific LangChain/agent tutorial creators |
| **Fireship** | Quick explainers on any tech topic | "X in 100 seconds" format gives you fast mental models before diving deep |
| **DataTalksClub** | MLOps Zoomcamp, ML Zoomcamp, Data Engineering Zoomcamp | Free, structured, community-driven courses with homework and projects |

### Tier 3 — Conference Talks & Vendors (Stay Current)

| Channel | What They Cover | Why Follow |
|---|---|---|
| **NVIDIA Developer** | GPU computing, TensorRT, Triton, CUDA, GTC talks | Essential for understanding GPU infrastructure and optimization |
| **Google Cloud Tech** | Vertex AI, Gemini, GenAI on GCP | Cloud-specific ML platform tutorials |
| **AWS Events / re:Invent** | SageMaker, Bedrock, AWS ML infrastructure | If your org is on AWS, these are critical |
| **LangChain (official)** | RAG from scratch, LangGraph, LangSmith | The source of truth for the most popular LLM framework |
| **Hugging Face** | Transformers, model hub, training, inference | The center of the open-source ML ecosystem |
| **MLOps Community** | Interviews with ML engineers, MLOps practitioners | Real-world stories from people building ML systems in production — includes talks by Chip Huyen, leading ML systems thinker |

---

## Suggested Viewing Order

**Weeks 1-2: Foundations**
1. 3Blue1Brown — Neural Networks series (1 hr)
2. StatQuest — ML playlist, pick 10-15 key videos (5 hrs)
3. Fireship — "Machine Learning in 100 Seconds" (quick context)

**Weeks 3-5: Deep Learning**
4. Andrej Karpathy — Neural Networks: Zero to Hero (13 hrs, do over 2 weeks)
5. Umar Jamil — Transformer Explained (1.5 hrs)

**Weeks 6-8: MLOps Core**
6. DataTalksClub — MLOps Zoomcamp (start the playlist, do alongside exercises)
7. freeCodeCamp — MLOps Course (3 hrs)
8. freeCodeCamp — MLflow Tutorial (3 hrs)

**Weeks 9-12: LLM Infrastructure (The Money Phase)**
9. LangChain — RAG from Scratch playlist (5 hrs)
10. Umar Jamil — RAG Pipeline Explained (1.5 hrs)
11. NeuralNine — vLLM Tutorial (25 min)
12. Trelis Research — LoRA/QLoRA Fine-Tuning (1 hr)
13. Weights & Biases — AI Agents Course (2 hrs)
14. DeepLearning.AI — pick 3-4 relevant short courses

**Ongoing: Stay Current**
15. Yannic Kilcher — watch his latest paper breakdowns weekly
16. MLOps Community — watch 1-2 talks per week
17. NVIDIA Developer — GTC talks relevant to your work

---

## Bonus: Books to Read Alongside

| Book | Author | Maps To |
|---|---|---|
| **AI Engineering** (2025) | Chip Huyen | Phases 5-8 — the most-read book on O'Reilly's platform, covers production AI systems end-to-end |
| **Designing Machine Learning Systems** (2022) | Chip Huyen | Phases 3-6 — Amazon bestseller, holistic approach to ML systems in production |
| **Deep Learning** | Ian Goodfellow, Yoshua Bengio, Aaron Courville | Phase 2 — the "textbook" for deep learning theory |
| **Hands-On Machine Learning** (3rd ed) | Aurélien Géron | Phases 1-2 — practical ML with scikit-learn, Keras, and TensorFlow |
| **Machine Learning Engineering** | Andriy Burkov | Phases 4-6 — focused on the engineering side of ML |

---

> **Pro tip:** Don't just watch. For every video, pause and replicate the code. Build something small after each section. A portfolio of 3-4 hands-on projects is worth more than 100 hours of passive watching.
