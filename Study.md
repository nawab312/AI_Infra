# Complete Study Guide: DevOps/SRE → AI Infra / MLOps Engineer

> **The definitive roadmap.** Every concept, every tool, every technique you need to study — organized across 9 phases — to transition from DevOps/SRE/Cloud Engineering into AI Infrastructure or MLOps Engineering.

---

## Table of Contents

- [Phase 1: ML Fundamentals](#phase-1-ml-fundamentals)
- [Phase 2: Deep Learning & Neural Network Essentials](#phase-2-deep-learning--neural-network-essentials)
- [Phase 3: Data Engineering for ML](#phase-3-data-engineering-for-ml)
- [Phase 4: ML Training Infrastructure](#phase-4-ml-training-infrastructure)
- [Phase 5: Model Packaging, Serving & Inference](#phase-5-model-packaging-serving--inference)
- [Phase 6: MLOps Pipelines & Automation](#phase-6-mlops-pipelines--automation)
- [Phase 7: LLM & GenAI Infrastructure](#phase-7-llm--genai-infrastructure)
- [Phase 8: Monitoring, Observability & Reliability for ML](#phase-8-monitoring-observability--reliability-for-ml)
- [Phase 9: Platform Engineering for ML](#phase-9-platform-engineering-for-ml)
- [Cross-Cutting Topics](#cross-cutting-topics)

---

# Phase 1: ML Fundamentals

> **Goal:** Build enough ML intuition to understand what you're operationalizing. You don't need to become a data scientist, but you must speak the language fluently.
>
> **Time estimate:** 4-6 weeks

---

## 1.1 The Machine Learning Paradigm

### What Machine Learning Actually Is

Traditional programming writes explicit rules. Machine learning learns rules from data. Understand this shift deeply because it changes everything about how you build, test, deploy, and monitor software.

```
Traditional: Input + Rules → Output
ML:          Input + Output → Rules (a "model")
```

### Types of Machine Learning

**Supervised Learning — the most common in production:**
The model learns from labeled examples (input-output pairs). This is what you'll deal with 80% of the time.

- **Classification:** Predicting a category. Is this email spam or not? Is this transaction fraudulent? What type of object is in this image? The output is a discrete label.
  - Binary classification (yes/no, spam/ham)
  - Multi-class classification (cat/dog/bird/fish)
  - Multi-label classification (an image can be both "outdoor" and "sunny")

- **Regression:** Predicting a continuous number. What will the stock price be? How long will this delivery take? What temperature will it be tomorrow? The output is a number on a continuous scale.

**Unsupervised Learning — finding structure in unlabeled data:**
No labels. The model discovers patterns on its own.

- **Clustering:** Grouping similar data points together. Customer segmentation, anomaly detection, topic modeling. Algorithms: K-means, DBSCAN, hierarchical clustering.
- **Dimensionality Reduction:** Compressing high-dimensional data into fewer dimensions while preserving structure. PCA (Principal Component Analysis), t-SNE, UMAP. Used for visualization and as a preprocessing step.
- **Anomaly Detection:** Finding data points that don't fit the normal pattern. Fraud detection, system health monitoring (relevant to your SRE background).

**Semi-Supervised Learning:**
A mix — you have a small amount of labeled data and a large amount of unlabeled data. The model uses the unlabeled data to improve its understanding of the labeled data. Common in practice because labeling data is expensive.

**Reinforcement Learning (RL):**
An agent learns by interacting with an environment and receiving rewards or penalties. Used in robotics, game AI, recommendation systems, and notably in RLHF (Reinforcement Learning from Human Feedback) for fine-tuning LLMs. You don't need deep RL knowledge for MLOps, but understand the concept and how RLHF works.

**Self-Supervised Learning:**
The model creates its own labels from the data. This is how LLMs are trained — predict the next token in a sequence. The "label" is the next word, which comes for free from the training text. Understanding this is critical for Phase 7.

### The ML Workflow (End-to-End)

```
1. Problem Definition
   └→ What are we trying to predict? What data do we have?
   └→ Is ML even the right approach? (Sometimes a rules engine is better)

2. Data Collection & Preparation
   └→ Gather data from various sources
   └→ Clean: handle missing values, remove duplicates, fix inconsistencies
   └→ Explore: understand distributions, correlations, outliers (EDA)
   └→ Transform: normalize, encode categorical variables, create features

3. Feature Engineering
   └→ Create new features from raw data that help the model learn
   └→ Feature selection: pick the most informative features
   └→ This is often where domain expertise matters most

4. Model Selection & Training
   └→ Choose an algorithm (or multiple to compare)
   └→ Split data: train (70%), validation (15%), test (15%)
   └→ Train the model on training data
   └→ Tune hyperparameters using validation data

5. Evaluation
   └→ Test on held-out test data (never seen during training)
   └→ Measure metrics appropriate to the problem
   └→ Check for overfitting, bias, fairness

6. Deployment
   └→ Package the model for serving
   └→ Set up inference infrastructure
   └→ Integrate with application

7. Monitoring & Maintenance
   └→ Monitor model performance in production
   └→ Detect data drift, concept drift
   └→ Retrain when performance degrades
```

Steps 6 and 7 are where your DevOps/SRE skills directly apply. Steps 1-5 are what data scientists do, but you need to understand them to build the infrastructure that supports them.

## 1.2 Common ML Algorithms

You don't need to implement these from scratch, but you need to understand what each does, when it's used, and its computational requirements.

### Linear Models

**Linear Regression:**
The simplest ML model. Fits a line (or hyperplane) to predict a continuous value. y = w₁x₁ + w₂x₂ + ... + b. Fast to train, interpretable, but can only model linear relationships.

**Logistic Regression:**
Despite the name, it's a classification algorithm. Uses a sigmoid function to output a probability between 0 and 1. Fast, interpretable, good baseline for binary classification.

### Tree-Based Models

**Decision Trees:**
A flowchart-like structure where each node splits the data based on a feature threshold. Easy to interpret but prone to overfitting.

**Random Forests:**
An ensemble of many decision trees, each trained on a random subset of data and features. Votes are averaged. Much more robust than a single tree. One of the most reliable "just works" algorithms.

**Gradient Boosted Trees (XGBoost, LightGBM, CatBoost):**
Trees are built sequentially, each one correcting the errors of the previous. State-of-the-art for tabular/structured data. XGBoost and LightGBM are the workhorses of production ML for non-deep-learning tasks. Very relevant because many production ML systems still use these, not deep learning.

### Support Vector Machines (SVM)

Finds the hyperplane that maximally separates classes. Effective in high-dimensional spaces. Less common today but still used in some specialized applications.

### K-Nearest Neighbors (KNN)

Classifies a data point based on the majority class of its K nearest neighbors. Simple, no training phase, but slow at inference time for large datasets. Conceptually important for understanding vector similarity search (relevant to vector databases in Phase 7).

### Neural Networks

Covered in depth in Phase 2. At this point, just understand that neural networks are function approximators composed of layers of interconnected nodes, and they power all of deep learning.

## 1.3 Training Concepts

### The Training Loop

```python
# Pseudocode for the fundamental training loop
for epoch in range(num_epochs):           # Iterate over the dataset multiple times
    for batch in data_loader:             # Process data in mini-batches
        predictions = model(batch.inputs) # Forward pass
        loss = loss_function(predictions, batch.labels)  # Compute error
        loss.backward()                   # Backward pass (compute gradients)
        optimizer.step()                  # Update model weights
        optimizer.zero_grad()             # Reset gradients for next batch
```

Understand each component:
- **Epoch:** One complete pass through the entire training dataset
- **Batch:** A subset of training data processed together (batch size is a hyperparameter)
- **Forward pass:** Input data flows through the model to produce predictions
- **Loss function:** Measures how wrong the predictions are (MSE for regression, cross-entropy for classification)
- **Backward pass (backpropagation):** Calculates how each weight contributed to the error
- **Optimizer:** Updates weights to reduce the error (SGD, Adam, AdamW)
- **Learning rate:** How big the weight updates are (too big = unstable, too small = slow)

### Overfitting vs Underfitting

**Overfitting:** The model memorizes the training data instead of learning general patterns. Performs great on training data, poorly on new data. Like studying only the practice test answers instead of understanding the subject.

**Signs:** Training accuracy is high, validation accuracy is much lower.

**Solutions:**
- More training data
- Regularization (L1/L2 penalties on weights)
- Dropout (randomly disable neurons during training)
- Early stopping (stop training when validation loss stops improving)
- Data augmentation (create variations of training data)
- Simpler model architecture

**Underfitting:** The model is too simple to capture the patterns in the data. Poor performance on both training and validation data.

**Solutions:**
- More complex model
- More features
- Train longer
- Reduce regularization

### Bias-Variance Tradeoff

- **Bias:** Error from oversimplifying the model. High bias = underfitting.
- **Variance:** Error from the model being too sensitive to training data. High variance = overfitting.
- The goal is to find the sweet spot — complex enough to capture patterns, simple enough to generalize.

### Hyperparameters vs Parameters

- **Parameters:** Learned during training (weights, biases). The model figures these out.
- **Hyperparameters:** Set by you before training (learning rate, batch size, number of layers, regularization strength). You need to tune these.

**Hyperparameter tuning methods:**
- Grid search (try all combinations — expensive)
- Random search (sample random combinations — surprisingly effective)
- Bayesian optimization (smart search based on previous results — Optuna, Ray Tune)

## 1.4 Evaluation Metrics

### Classification Metrics

**Confusion Matrix — the foundation:**
```
                    Predicted
                  Pos     Neg
Actual  Pos    [ TP  |  FN  ]
        Neg    [ FP  |  TN  ]

TP = True Positive  (correctly predicted positive)
FP = False Positive (incorrectly predicted positive — "false alarm")
FN = False Negative (missed positive — "miss")
TN = True Negative  (correctly predicted negative)
```

**Accuracy:** (TP + TN) / Total. Misleading when classes are imbalanced. If 99% of transactions are legitimate, a model that always predicts "legitimate" has 99% accuracy but catches zero fraud.

**Precision:** TP / (TP + FP). Of all things predicted positive, how many actually were? High precision = few false alarms. Important when false positives are costly (spam filter sending important email to spam).

**Recall (Sensitivity):** TP / (TP + FN). Of all actual positives, how many did we catch? High recall = few misses. Important when false negatives are costly (missing a cancer diagnosis).

**F1 Score:** Harmonic mean of precision and recall. 2 × (Precision × Recall) / (Precision + Recall). Useful when you need to balance precision and recall.

**AUC-ROC:** Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between classes across all threshold values. AUC = 1.0 is perfect, AUC = 0.5 is random guessing. Threshold-independent, making it useful for comparing models.

### Regression Metrics

**MSE (Mean Squared Error):** Average of squared differences between predictions and actual values. Penalizes large errors heavily.

**RMSE (Root Mean Squared Error):** Square root of MSE. Same units as the target variable, making it more interpretable.

**MAE (Mean Absolute Error):** Average of absolute differences. Less sensitive to outliers than MSE.

**R² (R-squared):** Proportion of variance explained by the model. R² = 1.0 means perfect prediction, R² = 0 means the model is no better than predicting the mean.

## 1.5 Data Concepts

### Train/Validation/Test Splits

- **Training set (70-80%):** Model learns from this
- **Validation set (10-15%):** Used to tune hyperparameters and monitor for overfitting during training
- **Test set (10-15%):** Final evaluation only. NEVER used during training or tuning. Simulates real-world performance.

**Cross-validation:** K-fold cross-validation splits data into K folds, trains on K-1, validates on 1, rotates. More robust evaluation but K times slower.

### Data Leakage

The cardinal sin of ML. Information from the test set "leaks" into the training process, giving artificially high performance that won't generalize. Examples: normalizing before splitting (statistics computed on test data), using future data to predict past events, target leakage (a feature that directly encodes the target).

### Class Imbalance

When one class is much more common than others (99% legitimate, 1% fraud). Solutions: oversampling minority class (SMOTE), undersampling majority class, class weights in the loss function, using appropriate metrics (F1, AUC instead of accuracy).

### Feature Engineering

The process of creating new features from raw data to improve model performance. Often the single biggest lever for improving model quality. Examples: extracting day-of-week from a timestamp, computing ratios between features, one-hot encoding categorical variables, text tokenization and TF-IDF.

## 1.6 ML Frameworks & Tools to Know

| Tool | What It's For | Your Interaction |
|------|--------------|------------------|
| scikit-learn | Classical ML (trees, SVMs, preprocessing) | Understand the API pattern (fit/predict), used by data scientists |
| PyTorch | Deep learning (training and inference) | You'll deploy and serve PyTorch models |
| TensorFlow / Keras | Deep learning (alternative to PyTorch) | Less common in new projects, but exists in legacy systems |
| XGBoost / LightGBM | Gradient boosted trees | Often in production for tabular data |
| pandas | Data manipulation (DataFrames) | Data scientists use it constantly; you'll encounter it in pipeline code |
| NumPy | Numerical computing (arrays, linear algebra) | The foundation under everything |
| Jupyter Notebooks | Interactive experimentation | Data scientists develop in notebooks; you need to productionize this code |

---

# Phase 2: Deep Learning & Neural Network Essentials

> **Goal:** Understand how modern deep learning works — from basic neural networks to Transformers. This is essential because most production ML in 2025-2026 is deep learning, and virtually all GenAI is built on Transformers.
>
> **Time estimate:** 4-6 weeks

---

## 2.1 Neural Network Fundamentals

### The Neuron (Perceptron)

The basic building block. Takes multiple inputs, multiplies each by a weight, sums them, adds a bias, and passes through an activation function.

```
Inputs:     x₁, x₂, x₃
Weights:    w₁, w₂, w₃
Bias:       b
Output:     activation(w₁x₁ + w₂x₂ + w₃x₃ + b)
```

### Layers

- **Input Layer:** Raw features enter here
- **Hidden Layers:** Where the model learns representations. More layers = deeper model = can learn more complex patterns
- **Output Layer:** Produces the final prediction

**Dense (Fully Connected) Layer:** Every neuron connects to every neuron in the next layer. The simplest layer type.

### Activation Functions

Without activation functions, a neural network is just a linear transformation (stacking linear layers gives you another linear layer). Activation functions introduce non-linearity, allowing the network to learn complex patterns.

**ReLU (Rectified Linear Unit):** f(x) = max(0, x). The most widely used. Simple, fast, works well. Problem: "dying ReLU" — neurons that output 0 for all inputs stop learning.

**GELU (Gaussian Error Linear Unit):** Smooth approximation of ReLU. Used in Transformers (BERT, GPT). f(x) = x × Φ(x) where Φ is the Gaussian CDF.

**Sigmoid:** f(x) = 1/(1+e^(-x)). Outputs between 0 and 1. Used in binary classification output layers. Problem: vanishing gradients for very large or small inputs.

**Softmax:** Converts a vector of numbers into a probability distribution (all positive, sum to 1). Used as the output layer for multi-class classification and in attention mechanisms.

**Tanh:** f(x) = (e^x - e^(-x))/(e^x + e^(-x)). Outputs between -1 and 1. Used in some RNN architectures.

### Backpropagation

The algorithm that makes training possible. After a forward pass computes the loss, backpropagation calculates the gradient (partial derivative) of the loss with respect to every weight in the network using the chain rule of calculus. These gradients tell the optimizer which direction to adjust each weight to reduce the loss.

```
Forward pass:  Input → Layer 1 → Layer 2 → ... → Output → Loss
Backward pass: Loss → ∂L/∂w_output → ∂L/∂w_layer2 → ... → ∂L/∂w_layer1
                        (gradients flow backward through the network)
```

**Vanishing gradients:** In deep networks, gradients can become extremely small as they propagate backward, causing early layers to stop learning. Solutions: ReLU activation, batch normalization, residual connections, careful initialization.

**Exploding gradients:** Gradients become extremely large, causing unstable training. Solutions: gradient clipping, batch normalization, careful initialization.

### Optimizers

**SGD (Stochastic Gradient Descent):** The simplest optimizer. Updates weights in the direction opposite to the gradient: w = w - lr × gradient. Can be slow and get stuck in local minima.

**Adam (Adaptive Moment Estimation):** Maintains per-parameter adaptive learning rates using running averages of gradients and squared gradients. The default choice for most deep learning. Nearly always works "well enough."

**AdamW:** Adam with decoupled weight decay. Better regularization. The standard optimizer for training Transformers and LLMs.

**Learning Rate Scheduling:** The learning rate doesn't have to be constant. Common schedules: linear warmup then cosine decay, step decay, one-cycle policy. LLM training typically uses warmup followed by cosine decay.

### Loss Functions

**Cross-Entropy Loss:** For classification. Measures the difference between the predicted probability distribution and the true distribution. The standard for both binary and multi-class classification.

**Mean Squared Error (MSE):** For regression. Average of squared differences between predictions and targets.

**Contrastive Loss / Triplet Loss:** For learning embeddings. Pushes similar items closer together and dissimilar items further apart in embedding space. Used in training embedding models (relevant to Phase 7 RAG).

## 2.2 Convolutional Neural Networks (CNNs)

CNNs are specialized for processing grid-like data, primarily images. Even though your focus is LLMs, understanding CNNs builds important intuition about how neural networks learn hierarchical features.

**Core concepts:**
- **Convolution:** A filter (kernel) slides across the input, computing element-wise multiplication and sum at each position. Extracts local features (edges, textures, shapes).
- **Pooling:** Reduces spatial dimensions (max pooling, average pooling). Makes the representation invariant to small translations.
- **Feature hierarchy:** Early layers detect simple features (edges), middle layers combine them into textures and parts, later layers recognize complex objects.

**Architectures to know (high-level):** LeNet, AlexNet (started the deep learning revolution), VGG, ResNet (introduced residual connections — critical concept), Inception.

**Why ResNet matters for you:** Residual connections (skip connections) allow gradients to flow directly through the network, enabling training of very deep networks. This concept is used in Transformers too.

## 2.3 Recurrent Neural Networks (RNNs)

RNNs process sequential data by maintaining a hidden state that acts as "memory" of previous inputs. Important historical context for understanding why Transformers replaced them.

**Vanilla RNN:** Processes one token at a time, updating hidden state. Problems: vanishing gradients make it impossible to learn long-range dependencies.

**LSTM (Long Short-Term Memory):** Adds gates (forget, input, output) to control information flow. Can learn longer dependencies than vanilla RNNs. Was the dominant architecture for NLP before Transformers.

**GRU (Gated Recurrent Unit):** Simplified LSTM with fewer parameters. Similar performance, faster training.

**Why RNNs lost to Transformers:**
- Sequential processing — can't parallelize (token N depends on token N-1)
- Still struggle with very long sequences despite LSTM/GRU improvements
- Transformers process all tokens in parallel during training, dramatically faster

## 2.4 The Transformer Architecture

**This is the most important section in Phase 2.** Every modern LLM (GPT, Llama, Claude, Gemini, Mistral) is a Transformer. Understanding this architecture is non-negotiable.

### The Core Innovation: Self-Attention

Self-attention allows every token in a sequence to attend to (look at) every other token, learning which tokens are most relevant to each other.

```
Input: "The cat sat on the mat"

For the word "sat":
- High attention to "cat" (who sat?)
- High attention to "mat" (where sat?)
- Low attention to "the" (less informative)
```

**How self-attention works mathematically:**

```
1. For each token, compute three vectors:
   Query (Q): "What am I looking for?"
   Key (K):   "What do I contain?"
   Value (V): "What information do I provide?"

2. Compute attention scores:
   Score(i,j) = Q_i · K_j / √d_k    (dot product, scaled)

3. Apply softmax to get attention weights:
   Weights = softmax(Scores)          (probabilities summing to 1)

4. Weighted sum of values:
   Output_i = Σ Weight(i,j) × V_j    (attend to relevant tokens)
```

**Multi-Head Attention:** Instead of a single attention mechanism, run multiple attention heads in parallel, each learning different relationships (syntax, semantics, coreference, etc.). Outputs are concatenated and projected.

### The Full Transformer Block

```
Input
  │
  ▼
[Multi-Head Self-Attention]
  │
  + ←── Residual Connection (add input back)
  │
  ▼
[Layer Normalization]
  │
  ▼
[Feed-Forward Network (FFN)]  ← Two linear layers with activation
  │
  + ←── Residual Connection
  │
  ▼
[Layer Normalization]
  │
  ▼
Output (feed to next Transformer block)
```

A Transformer model stacks many of these blocks. GPT-3 has 96 blocks. Llama 3 70B has 80 blocks.

### Positional Encoding

Self-attention treats the input as a set — it has no inherent notion of order. Positional encodings inject position information.

**Original (sinusoidal):** Fixed mathematical functions. Limited to a maximum sequence length.

**RoPE (Rotary Positional Embeddings):** Used by Llama, Mistral, and most modern LLMs. Encodes position as rotation in the embedding space. Can extrapolate to longer sequences than seen during training.

**ALiBi (Attention with Linear Biases):** Adds a linear bias to attention scores based on distance. Used by Falcon, Bloom.

### Encoder vs Decoder Architectures

**Encoder-only (e.g., BERT):** Processes the entire input bidirectionally. Each token can attend to all other tokens. Used for classification, NER, sentence embeddings. The attention mask is fully visible.

**Decoder-only (e.g., GPT, Llama, Claude):** Autoregressive — each token can only attend to previous tokens (causal mask). Used for text generation. This is the dominant architecture for LLMs.

**Encoder-Decoder (e.g., T5, BART, original Transformer):** Encoder processes input, decoder generates output while attending to encoder output. Used for translation, summarization.

```
Encoder-only (BERT):        Decoder-only (GPT/Llama):
Token 1 can see: [1,2,3,4]  Token 1 can see: [1]
Token 2 can see: [1,2,3,4]  Token 2 can see: [1,2]
Token 3 can see: [1,2,3,4]  Token 3 can see: [1,2,3]
Token 4 can see: [1,2,3,4]  Token 4 can see: [1,2,3,4]
(bidirectional)              (causal/autoregressive)
```

### Tokenization

Before text enters the model, it must be converted to numbers. Tokenizers split text into subword units.

**BPE (Byte Pair Encoding):** Used by GPT models. Starts with individual characters, iteratively merges the most frequent pairs. "unhappiness" might become ["un", "happiness"] or ["unhapp", "iness"].

**SentencePiece:** Used by Llama, T5. Similar to BPE but operates on raw text (no pre-tokenization). Can handle any language without language-specific rules.

**Vocabulary size:** Typical range is 32K to 128K tokens. Larger vocabulary = fewer tokens per text (faster inference, more efficient) but larger embedding matrix (more memory).

**Why tokenization matters for infra:** Token count directly determines memory usage (KV cache), compute time, and API costs. Understanding how text maps to tokens helps you estimate resource requirements.

## 2.5 Training Deep Learning Models

### Hardware: GPUs vs CPUs for Training

**Why GPUs:** Neural network training is embarrassingly parallel — the same operation (matrix multiplication) is applied to thousands of data points simultaneously. GPUs have thousands of cores optimized for this.

```
CPU: 8-64 powerful cores, sequential optimization
GPU: 5,000-16,000 simpler cores, parallel optimization

Matrix multiplication (1024×1024 × 1024×1024):
CPU (Intel Xeon): ~500ms
GPU (A100):       ~0.5ms (1000x faster)
```

**Key GPU specs that matter:**
- **VRAM (GPU Memory):** How large a model and how much data can fit. 16 GB (consumer), 24 GB (prosumer), 40-80 GB (datacenter A100/H100), 141 GB (H200), 192 GB (B200).
- **TFLOPS (Compute):** How many operations per second. Determines training speed.
- **Memory bandwidth:** How fast data can be read from VRAM. Critical for inference (memory-bound).
- **Interconnect (NVLink):** How fast GPUs can communicate. Critical for distributed training.

### Mixed Precision Training

Train with FP16/BF16 instead of FP32. Almost halves memory usage and doubles training speed with negligible accuracy loss.

```
FP32 training: Model + gradients + optimizer state ≈ 16× parameter count (bytes)
BF16 training: Model + gradients + optimizer state ≈ 10× parameter count (bytes)

70B model:
FP32: ~1,120 GB (14 × A100 80GB)
BF16 with mixed precision: ~700 GB (9 × A100 80GB)
```

BF16 is preferred over FP16 for training because it has the same exponent range as FP32 (avoids overflow/underflow issues).

### Key Hyperparameters for Deep Learning

| Hyperparameter | What It Does | Typical Range |
|----------------|-------------|---------------|
| Learning rate | Step size for weight updates | 1e-5 to 1e-2 |
| Batch size | Samples processed together | 16 to 4096 |
| Number of epochs | Passes over full dataset | 3 to 100+ |
| Weight decay | L2 regularization strength | 0.01 to 0.1 |
| Dropout rate | Fraction of neurons dropped | 0.1 to 0.5 |
| Warmup steps | Gradual LR increase at start | 100 to 10,000 |
| Gradient clipping | Max gradient norm | 1.0 to 5.0 |

### Batch Normalization & Layer Normalization

**Batch Normalization:** Normalizes activations across the batch dimension. Stabilizes training, allows higher learning rates. Common in CNNs.

**Layer Normalization:** Normalizes activations across the feature dimension (within each sample). Used in Transformers. Doesn't depend on batch size, making it better for variable-length sequences.

**RMSNorm:** Simplified layer normalization used in Llama and modern LLMs. Removes the mean-centering step, keeping only the variance normalization.

---

# Phase 3: Data Engineering for ML

> **Goal:** Learn how to build and manage the data infrastructure that feeds ML systems. Your data engineering skills are the bridge between raw data and trained models.
>
> **Time estimate:** 3-4 weeks (faster if you already know data engineering)

---

## 3.1 Data Pipelines and ETL for ML

### How ML Data Pipelines Differ from Traditional ETL

Traditional ETL produces clean, structured data for analytics and dashboards. ML data pipelines must additionally handle: feature computation, dataset versioning, train/test splitting, data validation, and feeding data to training jobs in the right format.

```
Traditional ETL:
Raw Data → Clean → Transform → Data Warehouse → BI Dashboards

ML Data Pipeline:
Raw Data → Clean → Feature Engineering → Validation → Split → 
           Version → Training Format (TFRecord/Parquet/Arrow) → 
           Training Job → Model Artifact
```

### Pipeline Orchestrators

**Apache Airflow:**
The most widely used workflow orchestrator. Define pipelines as Directed Acyclic Graphs (DAGs) in Python. Study:
- DAG definition and task dependencies
- Operators: PythonOperator, BashOperator, KubernetesPodOperator
- XComs for passing data between tasks
- Connections and hooks for external services
- Sensors for waiting on external events
- Task retries, SLAs, and alerting
- The Airflow scheduler and executor types (LocalExecutor, CeleryExecutor, KubernetesExecutor)
- Deploying Airflow on Kubernetes (Helm chart, KubernetesExecutor)

**Prefect:**
Modern alternative to Airflow. Pythonic API, better error handling, easier local development. Study:
- Flows and tasks (Python decorators)
- Retries, caching, and concurrency
- Prefect Cloud vs self-hosted server
- When to choose Prefect over Airflow (smaller teams, simpler workflows)

**Dagster:**
Software-defined assets approach — define what data assets exist, Dagster figures out the pipeline. Study:
- Assets, ops, and jobs
- Type checking and data validation
- Integration with dbt, Spark, and ML frameworks
- Dagster Cloud vs self-hosted

**Comparison:**

| Feature | Airflow | Prefect | Dagster |
|---------|---------|---------|---------|
| Philosophy | Task-centric DAGs | Pythonic workflows | Asset-centric |
| Learning curve | Steep | Moderate | Moderate |
| Industry adoption | Dominant | Growing | Growing |
| ML-specific features | Limited (needs plugins) | Good | Excellent |
| Local development | Painful | Easy | Easy |
| Kubernetes integration | KubernetesExecutor | Kubernetes agent | Kubernetes deployment |

### ML-Specific Orchestrators

**Kubeflow Pipelines:** Kubernetes-native ML pipeline orchestrator. Tightly integrated with ML tools (training, serving, model registry). Covered more in Phase 4.

**ZenML:** MLOps framework that abstracts pipeline orchestration. Can run on Airflow, Kubeflow, or locally. Good for smaller teams.

**Metaflow (Netflix):** Designed for data scientists. Each step is a Python function. Handles versioning, dependency management, and cloud execution automatically.

## 3.2 Feature Engineering and Feature Stores

### Feature Engineering for ML

Features are the inputs to your model. Raw data rarely works well directly — you need to transform it into features that help the model learn.

**Types of feature transformations:**
- **Numerical:** Normalization (min-max scaling to [0,1]), standardization (mean=0, std=1), log transforms for skewed distributions, binning continuous variables
- **Categorical:** One-hot encoding, label encoding, target encoding, embedding (for high-cardinality categories)
- **Text:** Tokenization, TF-IDF, word embeddings, sentence embeddings
- **Temporal:** Day of week, hour of day, time since last event, rolling averages, lag features
- **Aggregations:** Count, sum, mean, min, max over groups and time windows
- **Interactions:** Products or ratios of features (price_per_sqft = price / area)

### Feature Stores

A feature store is a centralized system for managing, storing, serving, and discovering ML features. It solves the problem of feature consistency between training and serving.

**The problem it solves:**
```
Without a feature store:
Training: SELECT avg(purchase_amount) FROM orders WHERE user_id = ?
           GROUP BY user_id  (computed in Python, batch)

Serving:  Recompute the same feature in real-time at inference...
          But wait, the SQL is slightly different, the time window is wrong,
          the joins are different... → Training-serving skew!
```

**Feast (open source):**
- **Offline store:** Historical features for training (BigQuery, Redshift, S3 + Parquet)
- **Online store:** Low-latency features for inference (Redis, DynamoDB, PostgreSQL)
- **Feature definitions:** Declared in Python, versioned in Git
- **Materialization:** Process to sync features from offline to online store
- Study: entity definitions, feature views, feature services, on-demand features, point-in-time joins

**Tecton (managed):**
- Built by the team that created Feast at Uber
- Real-time feature computation (streaming features)
- Managed infrastructure
- Study when Tecton is worth the cost vs self-hosting Feast

**Why feature stores matter for infra:**
You'll be deploying and managing the feature store infrastructure — Redis clusters for online serving, data warehouse connections for offline, materialization pipelines, monitoring feature freshness and drift.

## 3.3 Data Versioning

### Why Version Data

Code is versioned with Git. Models are versioned in model registries. But data — the most critical input to ML — is often not versioned at all. This makes ML experiments irreproducible.

**DVC (Data Version Control):**
"Git for data." DVC tracks large files and datasets alongside your Git repository without storing them in Git.

Study:
- `dvc init`, `dvc add`, `dvc push`, `dvc pull`
- Remote storage backends (S3, GCS, Azure, SSH)
- DVC pipelines (`dvc.yaml`) for reproducible ML workflows
- `dvc repro` for reproducing experiments
- Integration with Git (`.dvc` files tracked in Git, actual data stored remotely)
- DVC experiments for tracking hyperparameters and metrics

**LakeFS:**
Git-like operations (branch, commit, merge, diff) on data lakes. Works with S3-compatible storage. Enables data CI/CD — test data transformations on a branch before merging to production.

Study:
- Branching data for experimentation
- Commit and merge workflows
- Integration with Spark, Airflow, and ML frameworks
- Data quality gates before merge

## 3.4 Data Quality and Validation

### Why Data Quality Matters for ML

"Garbage in, garbage out." A model trained on bad data will produce bad predictions, no matter how sophisticated the algorithm.

**Great Expectations:**
Define "expectations" (assertions) about your data and validate automatically.

Study:
- Expectations: `expect_column_values_to_not_be_null`, `expect_column_values_to_be_between`, `expect_column_unique_value_count_to_be_between`
- Data Docs: auto-generated documentation of your data
- Checkpoints: automated validation runs
- Integration with Airflow/Prefect pipelines
- Custom expectations for domain-specific validations

**Pandera:**
Schema-based data validation for pandas DataFrames. Lighter weight than Great Expectations, good for in-pipeline validation.

**Key data quality checks for ML:**
- Schema validation (correct columns, correct types)
- Null/missing value rates
- Value distributions (detect drift from training data)
- Statistical tests (KS test, chi-squared test for distribution changes)
- Referential integrity
- Freshness (is the data up to date?)

## 3.5 Large-Scale Data Processing

### Apache Spark

The standard for processing datasets too large for a single machine. Study:

- Spark architecture: Driver, executors, cluster manager
- RDDs, DataFrames, and Datasets
- Transformations (lazy) vs actions (eager)
- Spark SQL for structured data processing
- PySpark API (Python interface)
- Spark MLlib for distributed ML
- Spark on Kubernetes (spark-operator)
- Tuning: partitioning, caching, broadcast joins, shuffle optimization

### Dask

Python-native distributed computing. Like pandas but distributed across a cluster. Study:

- Dask DataFrames (parallel pandas)
- Dask Arrays (parallel NumPy)
- Dask Delayed (parallelize arbitrary Python functions)
- Dask distributed scheduler
- When to use Dask vs Spark (Dask: Python-native, smaller scale; Spark: JVM, larger scale, more mature)

### Ray Data

Part of the Ray ecosystem (also used for distributed training and serving). Study:

- Ray Datasets for parallel data processing
- Integration with Ray Train and Ray Serve
- When to use Ray Data vs Spark (Ray: ML-focused, Python-native; Spark: general-purpose, more mature for ETL)

## 3.6 Storage Patterns for ML

### Data Lakes

Store raw data in its native format on object storage (S3, GCS, Azure Blob). Schema-on-read rather than schema-on-write.

### Data Lakehouses

Combine the flexibility of data lakes with the data management features of data warehouses. Technologies: Delta Lake, Apache Iceberg, Apache Hudi. Provide ACID transactions, schema enforcement, time travel, and efficient metadata management on top of object storage.

### Object Storage for Training Data

**Formats to know:**
- **Parquet:** Columnar format, excellent compression, fast reads. The standard for tabular ML data.
- **TFRecord:** TensorFlow's binary format. Protocol buffer based. Used with TensorFlow datasets.
- **Arrow (Apache Arrow):** In-memory columnar format. Zero-copy reads. Foundation of many ML data tools.
- **WebDataset:** Tar-based format for large-scale training. Excellent for streaming data to GPU training jobs. Used with PyTorch.
- **Hugging Face Datasets:** Arrow-based dataset library. Standard for NLP datasets. Memory-mapped for large datasets.

---

# Phase 4: ML Training Infrastructure

> **Goal:** Understand how to set up, manage, and optimize the infrastructure for training machine learning models at scale. This is the core of "AI Infrastructure" engineering.
>
> **Time estimate:** 4-6 weeks

---

## 4.1 GPU/TPU Cluster Management

### GPU Node Configuration

**NVIDIA Driver Stack:**
```
Application (PyTorch, TensorFlow)
     │
CUDA Toolkit (CUDA APIs, cuBLAS, cuDNN)
     │
NVIDIA Driver (kernel module)
     │
GPU Hardware (A100, H100, etc.)
```

Study:
- NVIDIA driver installation and version management
- CUDA toolkit versions and compatibility matrices
- cuDNN (CUDA Deep Neural Network library) for optimized neural network operations
- NCCL (NVIDIA Collective Communications Library) for multi-GPU communication
- nvidia-smi for monitoring GPU utilization, memory, temperature
- NVIDIA Container Toolkit (nvidia-docker) for GPU access in containers
- NVIDIA GPU Operator for Kubernetes (automates driver and toolkit installation)

### GPU Scheduling on Kubernetes

**NVIDIA Device Plugin:**
Exposes GPUs as a schedulable resource (`nvidia.com/gpu`) in Kubernetes. Each GPU is allocated to exactly one pod (no sharing by default).

Study:
- Device plugin installation and configuration
- GPU resource requests and limits in pod specs
- Node labeling for GPU types (`gpu-type: a100-80gb`)
- Taints and tolerations for GPU nodes
- GPU topology awareness (which GPUs share NVLink)

**Time-Slicing and MPS (Multi-Process Service):**
Allow multiple pods to share a single GPU. Time-slicing gives each pod turns on the GPU. MPS allows concurrent execution. Useful for development and light workloads, not for heavy training.

**MIG (Multi-Instance GPU):**
A100 and H100 can be physically partitioned into up to 7 independent GPU instances. Each instance has its own memory, L2 cache, and compute cores. Useful for serving multiple small models on a single GPU.

Study:
- MIG configuration and profiles (1g.10gb, 2g.20gb, 3g.40gb, etc.)
- When MIG vs time-slicing vs full GPU allocation
- MIG support in Kubernetes

### GPU Cluster Networking

**InfiniBand:**
The high-performance network fabric used in GPU clusters for distributed training. Much higher bandwidth and lower latency than Ethernet.
- HDR InfiniBand: 200 Gb/s per port
- NDR InfiniBand: 400 Gb/s per port
- RDMA (Remote Direct Memory Access): GPU-to-GPU data transfer bypassing the CPU

**RoCE (RDMA over Converged Ethernet):**
RDMA over standard Ethernet. Lower cost than InfiniBand, higher latency. Acceptable for smaller-scale distributed training.

**EFA (Elastic Fabric Adapter) on AWS:**
AWS's high-performance network interface for GPU instances. Provides RDMA-like performance on cloud infrastructure. Used with p4d, p5 instance types.

**GDR (GPUDirect RDMA):**
Direct data transfer between GPU memory and network without going through CPU. Essential for efficient multi-node training.

## 4.2 Distributed Training

### Why Distributed Training

Modern models are too large and datasets too big to train on a single GPU in a reasonable time.

```
Training time estimation:
Total FLOPs = 6 × Parameters × Tokens (Chinchilla approximation)

Llama 3 70B trained on 15T tokens:
6 × 70B × 15T = 6.3 × 10²² FLOPs

A100 at ~300 TFLOPS effective:
6.3 × 10²² / (300 × 10¹²) = 210,000,000 seconds = 6.65 years on 1 GPU

On 2048 GPUs: 6.65 years / 2048 ≈ 1.2 days
(In practice ~30 days due to communication overhead, failures, etc.)
```

### Data Parallelism

The simplest form of distributed training. Each GPU has a full copy of the model. The training batch is split across GPUs. After each step, gradients are averaged across all GPUs.

```
Data Parallel Training:

GPU 0: Full model copy, processes batch shard 0 → gradients 0 ─┐
GPU 1: Full model copy, processes batch shard 1 → gradients 1 ─┤─ AllReduce → averaged gradients
GPU 2: Full model copy, processes batch shard 2 → gradients 2 ─┤   → update all model copies
GPU 3: Full model copy, processes batch shard 3 → gradients 3 ─┘
```

**PyTorch DDP (DistributedDataParallel):**
PyTorch's built-in data parallelism. Study:
- Process group initialization (NCCL backend for GPUs)
- DistributedSampler for data sharding
- Gradient synchronization (AllReduce with NCCL)
- torch.distributed.launch and torchrun for multi-GPU/multi-node launch

**Limitation:** Each GPU must hold the full model. Doesn't work when the model is too large for a single GPU.

### Model Parallelism

Split the model itself across GPUs. Two main approaches:

**Tensor Parallelism (TP):**
Split individual layers (weight matrices) across GPUs. Each GPU computes a portion of every layer. Requires high-bandwidth interconnect (NVLink) because GPUs must communicate at every layer.

**Pipeline Parallelism (PP):**
Split the model by layers — GPU 0 runs layers 1-20, GPU 1 runs layers 21-40, etc. Less communication than TP (only between stages), but creates pipeline bubbles (idle time).

```
Pipeline Parallelism with microbatching:

GPU 0: [micro1] [micro2] [micro3] [micro4]  ---- idle ----  [grad4] [grad3] [grad2] [grad1]
GPU 1:  ---- [micro1] [micro2] [micro3] [micro4] ---- [grad4] [grad3] [grad2] [grad1] ----
GPU 2:   ---- ---- [micro1] [micro2] [micro3] [micro4] [grad4] [grad3] [grad2] [grad1] ----

Pipeline bubbles = wasted GPU time at beginning and end
```

### FSDP (Fully Sharded Data Parallelism)

PyTorch's native solution that combines aspects of data parallelism and model parallelism. The model parameters, gradients, and optimizer states are sharded (split) across all GPUs. Each GPU only holds 1/N of the complete state.

Study:
- FSDP sharding strategies: FULL_SHARD (shard everything), SHARD_GRAD_OP (shard gradients and optimizer), NO_SHARD (equivalent to DDP)
- Auto-wrapping policies (which modules to shard)
- Mixed precision with FSDP
- Activation checkpointing (gradient checkpointing) to save memory
- FSDP vs DeepSpeed ZeRO comparison

### DeepSpeed

Microsoft's deep learning optimization library. The ZeRO (Zero Redundancy Optimizer) stages are its key innovation.

**ZeRO Stages:**

```
Standard DDP (each GPU holds everything):
GPU Memory = Model Parameters + Gradients + Optimizer State
           = Φ + Φ + 12Φ (for Adam: momentum + variance + fp32 copy)
           = 14Φ per GPU (where Φ = model size in FP16 bytes)

ZeRO Stage 1 (partition optimizer state):
Each GPU: Model + Gradients + 1/N Optimizer State
Memory: Φ + Φ + 12Φ/N

ZeRO Stage 2 (partition optimizer state + gradients):
Each GPU: Model + 1/N Gradients + 1/N Optimizer State
Memory: Φ + Φ/N + 12Φ/N

ZeRO Stage 3 (partition everything):
Each GPU: 1/N Model + 1/N Gradients + 1/N Optimizer State
Memory: Φ/N + Φ/N + 12Φ/N = 14Φ/N
```

Study:
- ZeRO configuration (ds_config.json)
- ZeRO-Offload: offload optimizer/parameters to CPU RAM
- ZeRO-Infinity: offload to NVMe SSD
- DeepSpeed integration with Hugging Face Trainer
- Mixed precision training configuration
- Gradient accumulation for effectively larger batch sizes

### Horovod

Uber's distributed training framework. Uses MPI (Message Passing Interface) for communication. Simpler API than PyTorch DDP but less actively developed. Study it if your organization already uses it.

### 3D Parallelism

Combining all three parallelism strategies for training the largest models:

```
3D Parallelism = Data Parallelism × Tensor Parallelism × Pipeline Parallelism

Example: 1024 GPUs for a 175B model
- 8-way Tensor Parallelism (within a node, connected by NVLink)
- 4-way Pipeline Parallelism (across 4 nodes)
- 32-way Data Parallelism (32 replicas of the TP+PP configuration)
- 8 × 4 × 32 = 1024 GPUs
```

## 4.3 Experiment Tracking

### MLflow

The most widely used open-source experiment tracking platform.

Study:
- **MLflow Tracking:** Log parameters, metrics, and artifacts for each experiment run. Compare runs in the UI.
- **MLflow Projects:** Packaging ML code for reproducible runs. MLproject files, conda/docker environments.
- **MLflow Models:** Standard model packaging format. Flavors (sklearn, pytorch, tensorflow, transformers). Model signatures.
- **MLflow Model Registry:** Version and manage models through stages (Staging, Production, Archived). Approval workflows.
- **Deployment:** MLflow server setup, backend store (PostgreSQL), artifact store (S3/GCS), deployment on Kubernetes.
- **API:** Python, REST, R, and Java APIs for logging and querying experiments.

### Weights & Biases (W&B)

Cloud-based experiment tracking with superior visualization. Study:
- Experiment tracking (wandb.init, wandb.log)
- Hyperparameter sweeps (wandb.sweep)
- Artifacts (dataset and model versioning)
- Tables (interactive data visualization)
- Reports (collaborative experiment documentation)
- W&B Weave (LLM evaluation and observability)

### Neptune

Alternative to W&B with strong metadata tracking and comparison features.

### Comparison

| Feature | MLflow | W&B | Neptune |
|---------|--------|-----|---------|
| Hosting | Self-hosted or managed | Cloud (free tier) | Cloud (free tier) |
| Cost | Free (OSS) | Free for individuals | Free for individuals |
| Visualization | Basic | Excellent | Very good |
| Model Registry | Built-in | Supported | Supported |
| LLM Support | Via plugins | Weave, Traces | Limited |
| Enterprise | Databricks integration | Enterprise tier | Enterprise tier |

## 4.4 Hyperparameter Tuning

### Optuna

Python-based hyperparameter optimization framework. Uses Bayesian optimization (Tree-structured Parzen Estimator by default).

Study:
- Study and trial concepts
- Define-by-run API (suggest_float, suggest_int, suggest_categorical)
- Pruning (early stopping of unpromising trials)
- Multi-objective optimization
- Integration with PyTorch, TensorFlow, XGBoost
- Distributed optimization across multiple workers
- Visualization (optimization history, parameter importance)

### Ray Tune

Distributed hyperparameter tuning from the Ray ecosystem.

Study:
- Search algorithms (Bayesian, HyperOpt, genetic)
- Schedulers (ASHA, PBT — Population Based Training)
- Integration with Ray Train for distributed training
- Fault tolerance (checkpoint and resume)

## 4.5 Training Job Orchestration on Kubernetes

### Kubeflow Training Operator

Kubernetes-native operator for distributed training jobs.

Study:
- PyTorchJob CRD: Define master and worker specs for distributed PyTorch training
- MPIJob CRD: For Horovod-based training
- TFJob CRD: For TensorFlow distributed training
- Gang scheduling: Ensure all pods in a training job are scheduled together (avoid deadlocks)
- Integration with DeepSpeed and FSDP

### Ray on Kubernetes (KubeRay)

Deploy Ray clusters on Kubernetes for distributed training and tuning.

Study:
- RayCluster CRD: Define head and worker nodes
- RayJob CRD: Submit Ray jobs
- Autoscaling Ray clusters based on workload
- GPU scheduling with Ray

### Volcano

Kubernetes-native batch scheduling system. Provides fair-share scheduling, gang scheduling, queue management, and preemption. Essential for managing multi-tenant GPU clusters.

## 4.6 Spot/Preemptible Instance Strategies

Training jobs can run for days or weeks. Using spot instances (AWS), preemptible instances (GCP), or spot VMs (Azure) can save 60-90% on GPU costs.

Study:
- Checkpointing: Save model state periodically so training can resume after interruption
- Elastic training: Automatically adjust to available GPUs (resize data parallelism)
- Spot instance interruption handling: Graceful shutdown, checkpoint, restart on new instance
- Mixed spot + on-demand strategies: Use on-demand for the critical minimum, spot for scaling
- Nebius, Lambda Labs, RunPod, CoreWeave: Alternative GPU cloud providers, often cheaper than big-3

## 4.7 Storage and Networking for Training

### High-Performance Storage

Training jobs read massive datasets. Storage I/O can become the bottleneck.

- **Local NVMe SSDs:** Fastest option. Copy data to local SSD before training. Limited by node storage capacity.
- **Parallel file systems (Lustre, GPFS, BeeGFS):** Distributed file systems optimized for high-throughput parallel I/O. Standard in HPC environments.
- **FSx for Lustre (AWS):** Managed Lustre file system with S3 integration.
- **Filestore (GCP):** Managed NFS for GCP.
- **Object storage direct read:** Read from S3/GCS directly. Simplest but highest latency. Use streaming/prefetching to hide latency (WebDataset, FFCV).

### NCCL (NVIDIA Collective Communications Library)

The communication library for multi-GPU training. Study:
- Collective operations: AllReduce, AllGather, ReduceScatter, Broadcast
- NCCL topology detection (NVLink, PCIe, InfiniBand)
- NCCL environment variables for tuning (NCCL_IB_HCA, NCCL_SOCKET_IFNAME, NCCL_DEBUG)
- Debugging NCCL failures (the bane of distributed training)

---
# Complete Study Guide: DevOps/SRE → AI Infra / MLOps Engineer
## Part 2: Phases 5–9 + Cross-Cutting Topics

---

# Phase 5: Model Packaging, Serving & Inference

> **Goal:** Learn how to take a trained model and make it available as a reliable, scalable, low-latency service. This is where software engineering meets ML.
>
> **Time estimate:** 3–4 weeks

---

## 5.1 Model Serialization Formats

Before you can serve a model, you must understand how to save and load it. Each format has different portability, performance, and ecosystem tradeoffs.

### PyTorch Formats

**torch.save / torch.load (pickle-based):**
The simplest approach. Serializes the entire model (architecture + weights) using Python pickle.
- Fast and easy, but Python/PyTorch-version dependent
- Not portable across frameworks
- Security risk (pickle can execute arbitrary code on load)
- Use only for checkpointing during training, never for serving untrusted models

**TorchScript:**
Compiles PyTorch model into an intermediate representation that can run without a Python interpreter.
- `torch.jit.script`: Compiles based on code analysis (supports control flow)
- `torch.jit.trace`: Records execution trace (faster, but doesn't capture dynamic control flow)
- Portable across Python versions, can run in C++ via LibTorch
- Useful for mobile and embedded deployment

**ONNX (Open Neural Network Exchange):**
Framework-agnostic model format. Convert once, run anywhere — PyTorch, TensorFlow, scikit-learn, XGBoost, all export to ONNX. ONNX Runtime optimizes and executes ONNX models efficiently.
- Study: `torch.onnx.export`, ONNX Runtime inference, operator support and limitations, opset versions
- ONNX Runtime supports CPU, CUDA, TensorRT, OpenVINO, and DirectML execution providers
- Limitations: Dynamic shapes, custom ops, and some model architectures don't export cleanly

**SafeTensors (Hugging Face):**
Safe, fast alternative to pickle for storing model weights only (not architecture).
- No code execution risk (unlike pickle)
- Memory-mapped loading (extremely fast for large models)
- The standard for storing LLM weights in the Hugging Face ecosystem
- Architecture defined separately in `config.json`

### TensorFlow Formats

**SavedModel:**
TensorFlow's standard serialization format. Contains the computation graph and weights.
- Portable across TensorFlow versions
- Includes serving signatures (input/output specs)
- Supported by TensorFlow Serving and TFLite

**TFLite:**
Compressed format for mobile and edge deployment.

### Universal Formats

**GGUF (GPT-Generated Unified Format):**
The format used by llama.cpp for running LLMs on CPU and Apple Silicon. Stores weights with mixed precision (different layers at different bit widths). The format of choice for local LLM inference outside the cloud.

**TensorRT Engine:**
NVIDIA's compiled execution engine. Platform and hardware-specific — must be compiled for the target GPU architecture. Maximum performance on NVIDIA hardware.

### Comparison Matrix

| Format | Portability | Performance | Use Case |
|--------|-------------|-------------|----------|
| PyTorch .pt/.bin | Low (Python only) | Baseline | Training checkpoints |
| SafeTensors | Medium (needs architecture code) | Fast load | HuggingFace model distribution |
| TorchScript | Medium (PyTorch ecosystem) | Good | Edge, mobile, C++ serving |
| ONNX | High (any ONNX runtime) | Good (with ORT) | Cross-framework deployment |
| TensorRT Engine | Very Low (GPU-specific) | Maximum | Production NVIDIA serving |
| GGUF | Medium (llama.cpp ecosystem) | Good on CPU | Local inference, CPU serving |

## 5.2 Model Registries

A model registry is the system of record for trained models — versioning, metadata, lineage, and lifecycle management.

### MLflow Model Registry

Study:
- Registering models from training runs
- Model versioning (each registered model can have multiple versions)
- Model stages: None → Staging → Production → Archived
- Transition workflows and approval
- Model aliases (tag a version as "champion" or "challenger")
- Model descriptions and tags for searchability
- REST API for programmatic access
- Integration with CI/CD pipelines for automated promotion
- Deployment from registry to serving infrastructure

### Hugging Face Hub / Model Registry

Study:
- Pushing models to Hugging Face Hub (public and private)
- Model cards (documentation standard)
- Pulling models for inference (from_pretrained API)
- Private repositories and access tokens
- Organization management
- Using Hub as a model registry for self-hosted deployments

### Vertex AI Model Registry (GCP)

Study:
- Importing models (from GCS, BigQuery, or training jobs)
- Model versions and labels
- Model evaluation tracking
- Endpoint management and traffic splitting
- Integration with Vertex AI Pipelines

### Vendor-Specific Registries

- **AWS SageMaker Model Registry:** Groups, versions, and approval workflows for SageMaker
- **Azure Machine Learning Model Registry:** Model versioning, environments, and deployment

## 5.3 Serving Frameworks

### vLLM (covered in detail in Phase 7.1)

The current industry standard for LLM serving. OpenAI-compatible API, PagedAttention for memory efficiency, continuous batching.

### NVIDIA Triton Inference Server

The most feature-complete model serving framework for production. Supports TensorRT, ONNX, TorchScript, TensorFlow, Python backends, and more.

Study:
- **Model repository structure:** Directory layout with model versions and config
- **Configuration (`config.pbtxt`):** Input/output tensor specs, backend selection, batching configuration, instance groups (multiple model instances per GPU)
- **Dynamic batching:** Server-side batching of requests for throughput
- **Ensemble pipelines:** Chain multiple models (preprocessing → model → postprocessing)
- **gRPC and HTTP APIs:** Client libraries (tritonclient)
- **Backend types:** TensorRT, ONNX Runtime, PyTorch (LibTorch), TensorFlow, Python (custom)
- **Model control mode:** Poll, explicit (on-demand load/unload)
- **Rate limiting and priority scheduling**
- **Metrics endpoint:** Prometheus-compatible metrics (request count, queue time, compute time)
- **GPU and CPU instance groups:** Load same model on multiple GPUs or CPUs

```
triton/
└── model_repository/
    └── llama_70b/
        ├── config.pbtxt      # Model configuration
        ├── 1/                # Model version 1
        │   └── model.plan    # TensorRT engine (or model.onnx, model.pt, etc.)
        └── 2/                # Model version 2
            └── model.plan
```

### TorchServe

PyTorch's native model server. Designed for PyTorch models specifically.

Study:
- Model archiver (torch-model-archiver): Package model, handler, and dependencies into a .mar file
- Handler classes: Default handlers (image_classifier, text_classifier), custom handlers
- Management API: Load/unload models, scale workers
- Inference API: HTTP and gRPC endpoints
- Batch inference
- Custom handlers for preprocessing and postprocessing
- Logging and metrics

### BentoML

Framework for building and deploying ML services with a focus on developer experience.

Study:
- Runners: Wrap ML models as scalable components
- Service definition: Define endpoints with input/output types
- Bentos: Packaged services with all dependencies
- Containerization: `bentml containerize` builds Docker images
- Deployment to Kubernetes, AWS SageMaker, Google Cloud Run
- Input/output validation with Pydantic
- Adaptive batching

### Seldon Core

Kubernetes-native model serving platform focused on enterprise features.

Study:
- SeldonDeployment CRD: Define serving infrastructure in Kubernetes
- Pre-packaged servers: MLflow, Triton, TorchServe, SKLearn
- Custom servers: Python microservice approach
- Inference graphs: Chain models, routers, and combiners
- A/B testing and traffic splitting
- Canary rollouts
- Explanability integration (SHAP, Anchors)
- Outlier detection and drift detection integration

### FastAPI for Custom Serving

Sometimes the right approach is building a custom serving API with FastAPI.

Study:
- Async request handling
- Background tasks for preprocessing
- Lifespan events (load model on startup)
- Pydantic models for request/response validation
- Streaming responses for token streaming
- Prometheus metrics middleware
- Health check endpoints
- CORS handling

## 5.4 A/B Testing and Canary Deployments for Models

Model updates are risky. You need to gradually expose users to new models and compare performance.

**A/B Testing:**
Route a percentage of traffic to the new model, compare metrics.
- Define success metrics (accuracy, latency, user engagement)
- Randomize assignment consistently (same user always sees same model)
- Statistical significance testing
- Shadow mode: Send requests to both models, compare outputs without affecting users

**Canary Deployments:**
Gradually increase traffic to new model: 1% → 5% → 20% → 50% → 100%.
- Automated rollback on metric regression
- Monitor error rates, latency, and model quality metrics

**Tools:**
- Kubernetes Gateway API / Istio for traffic splitting
- Argo Rollouts for automated canary analysis
- Seldon Core for ML-specific A/B testing and shadow deployments
- Feature flags for model routing

## 5.5 Batch Inference

Not all inference needs to be real-time. Many ML workloads are batch: run predictions on millions of records overnight.

Study:
- Batch vs online inference tradeoffs
- Spark MLlib for distributed batch inference
- Ray for parallelized batch inference
- Batched GPU inference with PyTorch DataLoader
- Output storage: predictions back to database, data warehouse, or feature store
- Scheduling with Airflow or Kubernetes CronJobs
- Optimizing batch throughput: maximum batch sizes, mixed precision, model optimization

## 5.6 Inference Optimization

### Quantization

Covered in detail in Phase 7.1. Summary:
- FP16/BF16: Standard, no quality loss
- INT8: ~2x memory reduction, slight quality impact, use bitsandbytes or ONNX Runtime quantization
- INT4: ~4x memory reduction, noticeable quality impact, use GPTQ or AWQ for LLMs

### Pruning

Remove unnecessary weights (set them to zero) or entire neurons/heads.
- **Unstructured pruning:** Zero out individual weights. High compression ratio but poor hardware acceleration (sparse computation is not well-supported on GPUs).
- **Structured pruning:** Remove entire neurons, attention heads, or layers. Lower compression but better hardware acceleration.
- Study: `torch.nn.utils.prune`, magnitude pruning, gradient-based pruning

### Knowledge Distillation

Train a smaller "student" model to mimic a larger "teacher" model.
- Student learns from teacher's soft probability outputs (not just hard labels)
- Much more sample-efficient than training from scratch
- Study: distillation loss (KL divergence between teacher and student outputs), temperature scaling

### ONNX Runtime Optimization

- Graph optimization (constant folding, node fusion)
- Execution providers (CUDA, TensorRT, OpenVINO)
- Quantization (static and dynamic)

---

# Phase 6: MLOps Pipelines & Automation

> **Goal:** Build the automated systems that make ML development reproducible, reliable, and scalable. This is the core of "MLOps" and is where your DevOps background is most directly applicable.
>
> **Time estimate:** 4–5 weeks

---

## 6.1 CI/CD for Machine Learning

Traditional CI/CD tests and deploys code. ML CI/CD must additionally handle data, models, and statistical tests.

### The Four Types of Tests in ML CI/CD

**Code tests:** Unit tests for feature engineering code, data processing functions, and custom model layers. Standard pytest/unittest.

**Data tests:** Validate the training data. Schema checks, distribution checks, data quality assertions with Great Expectations or Pandera.

**Model training tests:** Does the model train without errors? Does loss decrease? Are metrics above a minimum threshold? Run a mini-training job.

**Model quality tests:** Does the model perform better than the current production model? Is it above the minimum acceptable threshold? Run inference on a held-out evaluation set.

### CI/CD Pipeline Structure for ML

```
Developer pushes code
        │
        ▼
  ┌─────────────┐
  │ 1. Code CI  │── Unit tests, linting, type checking
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 2. Data CI  │── Data validation, schema tests, drift detection
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 3. Train    │── Training job (mini or full), experiment tracking
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 4. Evaluate │── Benchmark vs production model, quality gates
  └──────┬──────┘
         │ (pass quality gate)
         ▼
  ┌─────────────┐
  │ 5. Stage    │── Deploy to staging, integration tests
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 6. Deploy   │── Canary → Production promotion
  └─────────────┘
```

### GitHub Actions for ML

Study:
- Defining ML workflows in `.github/workflows/`
- Self-hosted runners with GPU access for training jobs
- Caching dependencies and model artifacts
- Secrets management (cloud credentials, API keys)
- Matrix jobs for testing across Python/PyTorch versions
- Triggering retraining on data changes (via DVC or webhooks)
- Model deployment actions (push to registry, update serving config)

### DVC in CI/CD

Study:
- `dvc pull` in CI to get datasets
- `dvc repro` to reproduce pipelines
- `dvc metrics show` and `dvc metrics diff` for comparing model performance
- CML (Continuous Machine Learning): DVC's companion for ML CI/CD reporting. Comments metrics and plots directly on GitHub/GitLab PRs.

## 6.2 ML Pipeline Orchestrators

### Kubeflow Pipelines

Kubernetes-native ML workflow orchestration. Defines pipelines as Python code compiled to YAML.

Study:
- **Components:** Containerized pipeline steps defined with `@component` decorator
- **Pipelines:** Composed from components with `@pipeline` decorator
- **Artifacts:** Pass datasets, models, and metrics between components
- **Caching:** Skip re-running steps when inputs haven't changed
- **Pipeline parameters:** Configurable values per pipeline run
- **KFP SDK:** Python SDK for defining and submitting pipelines
- **Visualizations:** ROC curves, confusion matrices as pipeline step outputs
- **Recurring runs:** Scheduled pipeline execution
- **Integration with Vertex AI Pipelines:** GCP's managed Kubeflow Pipelines

### ZenML

Framework-agnostic MLOps pipeline tool. Same Python code runs locally, on Airflow, on Kubeflow, or on Vertex AI.

Study:
- Steps and pipelines (Python decorators)
- Materializers (how artifacts are serialized/deserialized)
- Stack: combination of orchestrator + artifact store + model registry
- ZenML Server for team collaboration
- Integration hub: 50+ integrations with ML tools

### Metaflow (Netflix)

Designed to be simple for data scientists while handling cloud execution transparently.

Study:
- Flow and steps (Python class-based)
- `@step` decorator and data passing between steps
- `@resources` decorator for specifying compute (CPU, memory, GPU)
- `@conda` / `@pypi` for dependency management
- Local development and cloud execution (AWS Batch, Kubernetes)
- `@retry` and `@catch` for fault tolerance
- Metaflow Cards for step documentation and visualization
- The `.metaflow` local metadata store and remote metadata service

### Vertex AI Pipelines (GCP)

Managed Kubeflow Pipelines on Google Cloud. Study if your organization is on GCP:
- Pre-built Google Cloud components (BigQuery, AutoML, etc.)
- Integration with Vertex AI experiments and model registry
- Pipeline scheduling and triggering
- Artifact lineage and metadata tracking

## 6.3 GitOps for ML

### What to Version

Everything that determines model behavior must be versioned:
- **Code:** Training scripts, preprocessing, feature engineering (Git)
- **Data:** Dataset version, data processing config (DVC, LakeFS)
- **Config:** Hyperparameters, architecture choices (Git, Hydra, YAML)
- **Model:** Trained weights and metadata (MLflow/W&B registry + artifact store)
- **Environment:** Python package versions, Docker image (requirements.txt, Dockerfile, Git)
- **Infrastructure:** Serving config, Kubernetes manifests (Git + Helm/Kustomize)

### Hydra for Configuration Management

Study:
- Structured configs with Python dataclasses
- Config composition (override from command line or other configs)
- Config groups for different environments/experiments
- Multirun for sweeping over config combinations
- Integration with Optuna for hyperparameter optimization

### Reproducibility

A training run should be reproducible given the same inputs. Study:
- Random seed management (torch.manual_seed, numpy.random.seed)
- Deterministic algorithms in PyTorch
- Recording all config, data version, and code version in experiment tracking
- Docker for environment reproducibility

## 6.4 Automated Testing for ML

### Data Tests

Using Great Expectations or Pandera:
- Schema tests: column names, types, no unexpected columns
- Distribution tests: mean, std, min, max within expected ranges
- Uniqueness tests: primary keys are unique
- Completeness tests: no nulls in required columns
- Consistency tests: derived columns match their derivation

### Model Tests

**Behavioral tests (CheckList methodology):**
- **Minimum functionality:** The model must get simple, unambiguous examples right
- **Invariance tests:** The model output should not change when irrelevant input changes (e.g., name change should not change sentiment)
- **Directional tests:** The model output should change in a predictable direction when input changes (e.g., adding "not" to a positive review should decrease sentiment score)

**Regression tests:**
- Performance on a held-out test set must stay above a threshold
- Comparison against the current production model ("challenger vs champion")

**Infrastructure tests:**
- Model loads successfully
- Inference API returns correct response format
- Response time under SLA for a set of test inputs
- Model handles edge cases (empty input, max length input, unicode)

## 6.5 Infrastructure as Code for ML Workloads

### Terraform for ML Infrastructure

Study:
- GPU instance provisioning (AWS p3/p4/p5, GCP A2/A3, Azure NCv3/NCasT4)
- Networking setup for GPU clusters (VPCs, security groups, placement groups)
- Managed ML services (SageMaker, Vertex AI, Azure ML)
- Storage for datasets and models (S3, GCS, Azure Blob)
- Kubernetes clusters with GPU node pools (EKS, GKE, AKS)
- Secrets management (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault)

### Helm for ML Workloads

Study:
- Helm charts for MLflow, Kubeflow, Airflow, JupyterHub
- Custom values files per environment (dev, staging, prod)
- Helm chart for vLLM serving deployments
- Dependency management (umbrella charts)

### Kustomize

Study:
- Kustomization files for environment overlays
- Patches for GPU resource limits/requests
- ConfigMap and Secret generators
- Integration with GitOps tools (ArgoCD, Flux)

---

# Phase 7: LLM & GenAI Infrastructure

> **Goal:** Become an expert in the infrastructure that powers large language models. This is the highest-demand specialization in 2025–2026.
>
> **Time estimate:** 6–8 weeks

---

## 7.1 LLM Serving & Optimization

*(Full deep-dive guide available separately)*

### Core Inference Concepts

**Autoregressive generation loop:** Token-by-token generation; each token depends on all previous tokens; sequential decode phase with parallel prefill.

**Prefill vs decode phases:** Prefill processes the entire prompt in parallel (compute-bound, determines TTFT); decode generates tokens one at a time (memory-bandwidth-bound, determines ITL).

**KV Cache:** Stores Key and Value vectors for all processed tokens to avoid recomputation. Memory is O(layers × sequence_length × batch_size). The primary memory bottleneck in LLM serving.

**Batching strategies:**
- Static batching: Fixed batch waits for slowest request
- Continuous/dynamic batching: Tokens are scheduled at each decode step; completed requests immediately free slots

**Throughput vs latency tradeoffs:** Larger batches → higher throughput, higher latency. Tune based on workload (interactive chat vs batch processing).

### Serving Frameworks

**vLLM:**
- PagedAttention: Block-based KV cache management eliminating memory fragmentation
- Continuous batching with iteration-level scheduling
- Prefix caching: Share KV cache blocks for common system prompts
- Tensor parallelism across multiple GPUs
- OpenAI-compatible REST API
- Quantization: AWQ, GPTQ, FP8, bitsandbytes
- Docker and Kubernetes deployment
- Configuration: `--gpu-memory-utilization`, `--max-model-len`, `--tensor-parallel-size`, `--enable-prefix-caching`, `--swap-space`
- Benchmarking: `benchmark_throughput.py`, `benchmark_serving.py`

**TGI (Text Generation Inference):**
- Rust-based core for performance
- Flash Attention integration (first-class)
- Speculative decoding (first-class)
- Watermark-based continuous batching
- HuggingFace model ecosystem native
- When to choose: Speculative decoding, HF ecosystem integration, managed via HF Inference Endpoints

**TensorRT-LLM:**
- Compile-time model optimization into TRT engines
- Kernel fusion and layout optimization
- FP8 quantization (native on H100+)
- In-flight batching
- Integration with Triton Inference Server
- When to choose: Maximum throughput on NVIDIA hardware, multi-model serving, enterprise deployments

### Inference Optimization Techniques

**Quantization (deep dive):**
- FP32 → FP16/BF16: Standard baseline, negligible quality loss
- FP8 (E4M3/E5M2): Native on H100, ~2x speedup, near-FP16 quality
- INT8: bitsandbytes (simple), ONNX Runtime (cross-platform), Smooth Quant (activation-aware)
- INT4 via GPTQ: Post-training, requires calibration dataset, good quality
- INT4 via AWQ: Activation-aware weight quantization, better than GPTQ for most models
- GGUF: llama.cpp format, CPU/Apple Silicon, mixed precision per layer

**Speculative decoding:**
- Draft model proposes K tokens; target model verifies in one forward pass
- Acceptance rate depends on draft-target alignment (same family: 70–90%)
- Optimal for low-concurrency, latency-sensitive workloads
- Medusa: Multiple draft heads within a single model (no separate draft model needed)

**KV cache optimizations:**
- Multi-Query Attention (MQA): All heads share one KV pair — aggressive reduction
- Grouped-Query Attention (GQA): Groups of heads share KV pairs — used by Llama 3, Mistral
- Sliding Window Attention: Bound KV cache size to a fixed window
- Prefix caching: Share KV cache across requests with identical prefixes

**Model parallelism for serving:**
- Tensor Parallelism: Split weight matrices across GPUs; requires NVLink; all GPUs on same node
- Pipeline Parallelism: Split layers across GPUs; works across nodes; pipeline bubbles
- When to use each: TP for intra-node, PP for cross-node
- TP degree must divide num_attention_heads evenly

**Structured output optimization:**
- Constrained decoding: Restrict next tokens to valid grammar/schema
- Grammar-guided generation: Use formal grammars (JSON Schema, EBNF)
- Tools: Outlines library, vLLM guided decoding, XGrammar

### Hardware Knowledge

**GPU comparison:**

| GPU | VRAM | Mem BW | FP16 TFLOPS | FP8 TFLOPS | NVLink BW |
|-----|------|--------|-------------|------------|-----------|
| A100 40GB | 40 GB | 1.6 TB/s | 312 | — | 600 GB/s |
| A100 80GB | 80 GB | 2.0 TB/s | 312 | — | 600 GB/s |
| H100 SXM | 80 GB | 3.35 TB/s | 990 | 1,979 | 900 GB/s |
| H200 | 141 GB | 4.8 TB/s | 990 | 1,979 | 900 GB/s |
| B200 | 192 GB | 8.0 TB/s | 2,250 | 4,500 | 1,800 GB/s |
| L40S | 48 GB | 864 GB/s | 366 | 733 | — |
| A10G | 24 GB | 600 GB/s | 125 | — | — |

**Memory bandwidth dominates decode speed.** Key formula:
`Theoretical max tokens/sec ≈ Memory Bandwidth / Model Size in Memory`

**NVLink:** Required for efficient tensor parallelism. Without NVLink, AllReduce over PCIe becomes the bottleneck.

**NVSwitch:** Full-bisection bandwidth GPU fabric. Enables any-to-any GPU communication at full NVLink speed within a node.

**CPU offloading:**
- KV cache offload to CPU RAM (vLLM swap-space)
- Optimizer state offload during training (ZeRO-Offload)
- Full model offload to NVMe (ZeRO-Infinity, llama.cpp)

**Cost analysis:** Right-size GPU to model; quantize to fit more on fewer GPUs; spot instances for stateless serving (60–90% savings); model routing for cost optimization.

---

## 7.2 RAG Architecture

*(Full deep-dive guide available separately)*

### Foundational Concepts

**Why RAG:** LLMs have knowledge cutoffs, hallucinate, and can't access private data. RAG solves all three at query time.

**Basic pipeline:** Embed query → retrieve similar chunks → augment prompt → generate.

**RAG vs fine-tuning:** RAG for knowledge injection and private data; fine-tuning for behavior, style, and format. Often combined.

**Architecture levels:**
- Naive RAG: Basic embed-retrieve-generate
- Advanced RAG: Pre/post retrieval optimizations (reranking, query rewriting, compression)
- Modular RAG: Composable, swappable modules for each pipeline step

### Embeddings

**What they do:** Map text to dense vectors in high-dimensional space; semantically similar texts are geometrically close.

**Popular models:**
- OpenAI text-embedding-3-large (3072 dims, proprietary)
- Cohere embed-v4 (1024 dims, multilingual)
- BGE-M3 (1024 dims, open source, multilingual, 8192 context)
- E5-large-v2 (1024 dims, needs "query:"/"passage:" prefix)
- nomic-embed-text-v1.5 (768 dims, fully open source)

**Evaluation:** MTEB benchmark for general ranking; build a task-specific eval set from your domain.

**Dimensionality tradeoffs:** Higher dims = better quality, more storage, slower search. 768–1024 is the sweet spot for most production systems.

**Self-hosting:** TEI (Text Embeddings Inference) by HuggingFace; supports batching, GPU acceleration, and quantization.

**Multimodal embeddings:** CLIP (image+text shared space); ColPali/ColQwen (page-level image embeddings for PDFs).

### Chunking Strategies

**Fixed-size with overlap:** Simple, fast. Good default for unstructured text. 256–1024 tokens, 10–20% overlap.

**Recursive character splitting:** Uses a hierarchy of separators (paragraph → sentence → word). Better than fixed-size for natural text. LangChain's default.

**Semantic chunking:** Split on meaning boundaries using embedding similarity between consecutive sentences. Coherent chunks, variable size.

**Document-structure-aware:** Respect headers, sections, tables, lists. Best for structured documents (Markdown, HTML, PDFs with clear structure).

**Agentic chunking:** LLM decides chunk boundaries. Best quality, most expensive.

**Parent-document retrieval:** Small chunks for retrieval, return larger parent for context.

**Metadata enrichment per chunk:** source, page_number, section_title, document_date, document_type, language, summary, hypothetical_questions.

### Vector Databases

**Core concepts:**
- Distance metrics: Cosine similarity (text), Euclidean (when magnitude matters), dot product (fast, unnormalized)
- ANN algorithms: HNSW (best recall, memory-heavy), IVF (cluster-based, less memory), PQ/IVFPQ (compressed, lossy), DiskANN (billion-scale, SSD-based)
- HNSW parameters: M (connections, default 16), ef_construction (build quality, default 200), ef_search (query recall, default 100)

**Pinecone:** Fully managed, serverless. Indexes, namespaces, metadata filtering, hybrid search. Zero ops, vendor lock-in.

**Weaviate:** Open source, self-hostable. Rich schema, built-in vectorizers, multi-tenancy, hybrid BM25+vector. Good for feature-rich self-hosted deployments.

**Milvus/Zilliz:** High-performance open source. Multiple index types (IVF_FLAT, IVF_PQ, HNSW, DiskANN), GPU indexing, billion-scale. Most complex to operate.

**pgvector:** PostgreSQL extension. HNSW and IVF-FLAT indexes. Best when < 5M vectors, already on Postgres, need SQL joins alongside vector search.

**Qdrant:** Rust-based, strong performance, named vectors, rich payload filtering. Growing fast.

**Chroma:** Lightweight, embedded. Development and prototyping only.

**Redis Vector Search:** Good when already using Redis, smaller scale.

### Retrieval Strategies

**Hybrid search:** Dense vector + sparse BM25 combined with alpha weighting or Reciprocal Rank Fusion (RRF). Covers both semantic and keyword retrieval.

**Reranking:** Two-stage: retrieve top-50 with vector search → rerank to top-5 with cross-encoder (Cohere Rerank, BGE-reranker-v2, FlashRank). Adds 50–200ms but significant quality improvement.

**Multi-query retrieval:** LLM generates 3–5 query variations; union of results improves recall.

**HyDE:** Generate a hypothetical answer, embed it, search with that embedding instead of the raw query.

**Contextual compression:** LLM extracts only relevant sentences from retrieved chunks.

**Multi-index routing:** Route queries to different indexes based on intent (FAQ vs technical docs vs policy).

**Recursive retrieval:** Iteratively refine retrieval — retrieve, evaluate, re-query if needed.

### Advanced RAG Patterns

**Self-RAG:** Model generates reflection tokens to decide when to retrieve and self-evaluate its outputs. Requires a fine-tuned model.

**CRAG (Corrective RAG):** Evaluate retrieval quality with a classifier; fall back to web search if retrieval confidence is low.

**Graph RAG:** Extract entities and relationships, build a knowledge graph, traverse it for retrieval. Better for complex cross-document reasoning.

**Agentic RAG:** An agent autonomously decides how to retrieve — multi-step, multi-tool, multi-index.

**Multi-modal RAG:** Retrieve and reason over images, tables, and charts alongside text. Options: text extraction, page-image embeddings (ColPali), hybrid.

### RAG Evaluation

**Retrieval metrics:** Context precision (fraction of retrieved chunks that are relevant), context recall (fraction of relevant chunks retrieved), MRR, NDCG@k.

**Generation metrics:** Faithfulness (no hallucination), answer relevancy, answer correctness.

**Frameworks:** RAGAS (faithfulness, answer relevancy, context precision/recall), DeepEval (hallucination, toxicity, conversation evaluation), Promptfoo (A/B testing, regression testing).

---

## 7.3 Fine-Tuning Infrastructure

### When to Fine-Tune

Fine-tune for: consistent output format, domain-specific language and tone, specialized task performance, behavior modification (follow specific instructions reliably).

Use RAG instead for: current information, private data access, knowledge injection, quick updates.

Combine both: Fine-tune for format/style + RAG for knowledge.

### Fine-Tuning Approaches

**LoRA (Low-Rank Adaptation):**
- Freeze base model weights; add trainable low-rank matrices to attention layers
- ΔW = B × A where B ∈ R^(d×r), A ∈ R^(r×k), rank r << d
- Parameters: rank r (typically 8–128; higher = more expressive, more memory), alpha (scaling, typically = rank or 2×rank), target modules (q_proj, v_proj, all linear layers)
- Memory savings: Only r×(d+k) parameters instead of d×k; typically < 1% of base model parameters
- Merging: LoRA weights can be merged back into base model for zero-overhead inference
- Multi-LoRA serving: S-LoRA, Punica — serve base model + hundreds of LoRA adapters on same GPU

**QLoRA (Quantized LoRA):**
- Quantize base model to 4-bit NF4 (NormalFloat4) — reduces base model memory by ~75%
- Train LoRA adapters in BF16 (full precision for the trainable parts)
- Double quantization: Also quantize the quantization constants (saves ~0.5 bits/parameter)
- Paged optimizers: Offload optimizer states to CPU RAM with unified memory (handles GPU OOM gracefully)
- Enables fine-tuning a 70B model on a single 48GB GPU
- Slight quality degradation vs full LoRA due to quantization noise

**PEFT (Parameter-Efficient Fine-Tuning) techniques:**
- Prefix tuning: Prepend trainable prefix tokens to keys and values in attention
- Prompt tuning: Learn soft prompt embeddings (simplest, weakest)
- IA³: Rescale activations with learned vectors — very few parameters
- AdaLoRA: Adaptively allocate rank budget across layers based on importance

**Full fine-tuning:**
- Update all model weights
- Requires FSDP or DeepSpeed ZeRO Stage 3 for large models
- Use when: maximum quality required, sufficient GPU budget, domain shift is large
- Risk: Catastrophic forgetting (mitigate with replay, EWC, or LoRA)

### Fine-Tuning Data

**Data format:**
- Instruction format: `{"instruction": "...", "input": "...", "output": "..."}`
- Chat format: OpenAI messages format with system/user/assistant turns
- Apply the model's chat template (Llama 3 uses a specific template; using the wrong one hurts performance)

**Data quality over quantity:** 1,000 high-quality examples often outperform 100,000 noisy examples.

**Synthetic data generation:** Use GPT-4 or Claude to generate instruction-response pairs. Verify quality before training.

**Data decontamination:** Ensure no test set benchmarks leak into your training data.

**Tools:**
- Argilla: Open-source data annotation and curation platform
- Label Studio: General annotation tool
- Hugging Face datasets: Preprocessing and dataset management

### Fine-Tuning Tooling

**Hugging Face TRL (Transformer Reinforcement Learning):**
- SFTTrainer: Supervised fine-tuning with PEFT integration
- DPOTrainer: Direct Preference Optimization (alignment)
- RewardTrainer: Training reward models
- PPOTrainer: RLHF with PPO
- DataCollatorForCompletionOnlyLM: Mask prompt tokens in loss computation

**Axolotl:**
- YAML-based configuration for complex fine-tuning setups
- Supports LoRA, QLoRA, full fine-tuning
- Handles dataset mixing, chat templates, and multi-GPU automatically
- Preferred by practitioners for "production" fine-tuning runs

**Unsloth:**
- 2–5x faster training via custom CUDA kernels
- 70% less memory through manual gradient computation
- Drop-in replacement for HuggingFace Trainer
- Best for single-GPU fine-tuning where speed matters

**LLaMA-Factory:**
- Web UI for fine-tuning (good for data scientists)
- Supports 100+ models out of the box
- Distributed training support

**Cloud compute for fine-tuning:**
- Lambda Labs, RunPod, CoreWeave, Vast.ai: Cheaper GPU rentals
- Modal, Replicate: Serverless GPU for fine-tuning jobs
- AWS SageMaker, GCP Vertex AI: Managed training with enterprise features

### Alignment Techniques

**RLHF (Reinforcement Learning from Human Feedback):**
1. Supervised fine-tuning on demonstrations
2. Train a reward model on human preference data (which response is better?)
3. Use PPO (Proximal Policy Optimization) to optimize the LLM against the reward model

**DPO (Direct Preference Optimization):**
Skips the reward model. Directly optimizes the policy using human preference pairs (chosen vs rejected responses). Simpler, more stable than PPO. The current standard for alignment.

**Study:** KL divergence constraint (preventing the model from straying too far from the base model), beta parameter in DPO (controls KL penalty strength), choosing reference model.

### Post-Training Evaluation

**LM Evaluation Harness:** Standardized evaluation on benchmarks (MMLU, HellaSwag, ARC, TruthfulQA, etc.)

**Human evaluation:** Pair comparisons, Likert scales, task-specific rubrics.

**Domain-specific eval sets:** Build your own test set from real use cases. The most important evaluation.

**Model merging:**
- TIES merging: Resolves parameter conflicts when merging multiple fine-tuned models
- DARE: Drops random fine-tuned weights before merging to reduce interference
- Model soups: Average weights of multiple fine-tuned models for ensemble effect

---

## 7.4 Prompt Management & Evaluation

### Prompt Engineering Fundamentals

**Zero-shot:** Direct instruction with no examples. Works for simple tasks with capable models.

**Few-shot:** Provide 2–10 examples of desired input-output behavior. Improves reliability dramatically for complex tasks.

**Chain-of-Thought (CoT):** Ask the model to reason step-by-step before answering. Dramatically improves performance on reasoning tasks. "Let's think step by step."

**ReAct:** Interleave reasoning (Thought) with actions (Act) and observations. Foundation for agents.

**System prompts:** Instructions that persist across all turns. Define persona, capabilities, constraints, output format.

**Structured output prompting:** Specify the exact JSON schema in the prompt. Use with constrained decoding for guaranteed valid output.

**Meta-prompting:** Use an LLM to generate or improve prompts. Automatic Prompt Engineer (APE) for automated prompt optimization.

### Prompt Management in Production

**Version control for prompts:**
- Store prompts in Git as `.txt`, `.yaml`, or `.json` files
- Use semantic versioning for major prompt changes
- Tag prompt versions that correspond to production deployments
- Never hardcode prompts in application code — treat them as configuration

**Prompt registries:**
- LangSmith: Prompt Hub for versioned, annotated prompts
- Langfuse: Prompt management with versioning and A/B testing
- Custom: Simple Git + API service for prompt retrieval

**Environment-specific prompts:**
- Different prompts for dev/staging/prod
- A/B test prompt variants in production
- Gradual rollout of prompt changes (same as canary deployments)

**Dynamic prompts:**
- Template variables for runtime injection (`{{user_name}}`, `{{context}}`)
- Conditional prompt assembly based on request properties
- Few-shot example selection based on semantic similarity to the query

### LLM Evaluation

**Task-specific automated metrics:**
- BLEU, ROUGE: N-gram overlap for summarization and translation (poor correlation with human judgment for open-ended tasks)
- METEOR, BERTScore: Semantic similarity-based metrics
- Exact match, F1: For QA with specific correct answers
- Pass@k: For code generation (does at least 1 of k samples pass tests?)

**LLM-as-Judge:**
Use a powerful model (GPT-4, Claude) to evaluate another model's outputs. Score on dimensions like helpfulness, harmlessness, honesty, coherence, factual accuracy. Key concerns: position bias (judge prefers first answer), verbosity bias (judge prefers longer answers), self-enhancement bias (judge prefers its own outputs).

**Evaluation frameworks:**
- RAGAS: Faithfulness, answer relevancy, context precision/recall for RAG
- DeepEval: Rich set of metrics, CI/CD integration, regression testing
- Promptfoo: Multi-model comparison, custom assertions, HTML reports
- Braintrust: Experiment tracking for evals with human feedback integration
- OpenAI Evals: Framework for evaluation datasets and metrics

**Building your golden test set:**
- 50–200 representative real-world queries
- Ground truth answers (human-labeled or from authoritative sources)
- Include edge cases: empty input, very long input, adversarial input, ambiguous queries, out-of-scope queries
- Treat the test set as sacred — never train on it, never optimize against it directly

**Regression testing:**
- Run eval suite on every code/prompt change
- Track metrics over time in a dashboard
- Alert on metric drops above a threshold (e.g., > 5% faithfulness decrease)

---

## 7.5 LLM Observability & Guardrails

### What to Monitor for LLMs

**Infrastructure metrics (your SRE instincts apply):**
- TTFT (Time to First Token): P50, P95, P99
- ITL (Inter-Token Latency): P50, P95
- Request throughput (requests/sec)
- Token throughput (tokens/sec, input and output separately)
- Error rate (4xx, 5xx, model errors)
- GPU utilization and memory utilization
- KV cache utilization (vLLM exposes this)
- Request queue depth and wait time

**ML-specific metrics:**
- Token usage per request (input tokens, output tokens, total)
- Cost per request (tokens × price/token)
- Model quality scores (faithfulness, relevancy, hallucination rate)
- Retrieval quality (for RAG: context precision, recall)
- User feedback signals (thumbs up/down, explicit ratings)
- Refusal rate (how often does the model refuse to answer)
- Output length distribution

### Observability Tools

**LangSmith:**
- Trace every LLM call, chain, and agent run
- See exact prompts and completions for every request
- Latency breakdown per chain step
- Dataset management for evaluation
- Human annotation workflows
- Feedback collection and aggregation
- Online evaluation (automated scoring of production traces)

**Langfuse (open source, self-hostable):**
- LangSmith alternative with full data control
- Tracing with span hierarchy
- Prompt management with versioning
- Metric dashboards
- Human annotation interface
- Self-host on Docker or Kubernetes (PostgreSQL + Redis)

**Arize Phoenix (open source):**
- LLM tracing and evaluation
- Embedding visualization (UMAP/t-SNE of queries and retrieved chunks)
- Data drift detection for embeddings
- RAG-specific metrics

**Helicone:**
- LLM proxy: Insert between your application and the LLM API
- Automatic logging without code changes
- Caching (semantic and exact match)
- Rate limiting per user/org
- Cost tracking and alerting

**OpenTelemetry for LLMs:**
- Extending standard OTel to LLM workloads
- LLM-specific spans and attributes (model name, token counts, prompts)
- Works with existing OTel infrastructure (Grafana, Jaeger)
- OpenTelemetry Semantic Conventions for LLMs (emerging standard)

### Guardrails & Safety

**Input guardrails (validate before sending to LLM):**
- Prompt injection detection: Identify attempts to override system prompt
- Jailbreak detection: Classifier or rule-based detection of adversarial inputs
- PII detection: Flag or redact personal information in user input (Microsoft Presidio, custom NER)
- Content policy enforcement: Block prohibited topics before model call
- Rate limiting: Per user, per IP, per organization

**Output guardrails (validate before returning to user):**
- Toxicity and content filtering: Classify output safety (Perspective API, custom classifiers)
- Hallucination detection: Check if claims are grounded in context (for RAG)
- PII in output: Detect and redact PII generated by the model
- Format validation: Ensure output matches required schema (JSON, specific fields)
- Length and coherence checks: Detect truncated or nonsensical outputs

**Tools:**
- Guardrails AI: Define Rails (output specifications) declaratively; validator library for common checks
- NeMo Guardrails (NVIDIA): Programmable guardrails using Colang language; topical, safety, and dialog rails
- LlamaGuard (Meta): Fine-tuned classifier for safety categories
- Custom classifiers: Train your own using a small labeled dataset

### Semantic Caching and Cost Optimization

**Exact match caching:** Cache LLM responses for identical prompts. High hit rate for common queries.

**Semantic caching:** Cache based on embedding similarity — if a new query is very similar to a cached query, return the cached response. Tools: GPTCache, Redis with vector search, Momento Semantic Cache.

**Benefits:** Reduced latency, reduced cost (no LLM call for cache hits), improved consistency.

**Considerations:** Cache invalidation (when knowledge changes), semantic threshold tuning (too strict = low hit rate, too loose = incorrect responses), monitoring cache hit rate.

**Model routing for cost optimization:**
- Use a small classifier to assess query complexity
- Route simple queries to small cheap models (8B)
- Route complex queries to large models (70B) or API (GPT-4)
- Target: serve 80% of queries on the cheap path

---

## 7.6 Multi-Model Orchestration & Agent Infrastructure

### Orchestration Frameworks

**LangChain:**
- LCEL (LangChain Expression Language): Compose chains with the pipe operator (`|`)
- Runnables: Composable components (prompts, models, output parsers, retrievers)
- Chains: Pre-built common patterns (RetrievalQA, ConversationalRetrievalChain)
- Memory: Conversation buffer, summary, entity memory for stateful conversations
- Tools: Python REPL, web search, custom API tools
- Agents: ReAct, OpenAI functions, custom agent loops
- Callbacks: Logging, streaming, monitoring hooks
- LangGraph: Stateful multi-actor workflows (covered below)

**LangGraph:**
- Graph-based workflow definition: Nodes (functions) + Edges (transitions)
- State: Typed state shared across nodes (messages, context, tool results)
- Conditional edges: Branch based on state
- Cycles: Agents can loop until a condition is met (true agents)
- Checkpointing: Persist agent state between turns (conversation memory)
- Human-in-the-loop: Interrupt graph for human approval before tool use
- Multi-agent: Supervisor graph routing to specialized sub-agents

**LlamaIndex:**
- Data connectors (LlamaHub): 100+ source connectors (PDF, Notion, Slack, databases)
- Index types: Vector, list, tree, keyword, knowledge graph
- Query engines: Built-in RAG pipeline
- Routers: Route queries to appropriate index or query engine
- Sub-question decomposition: Break complex questions into sub-questions
- Agents: ReAct and OpenAI function-calling agents
- Workflows: Async, event-driven workflow engine (newer API)

**Semantic Kernel (Microsoft):**
- .NET first, also supports Python and Java
- Plugin system for extending model capabilities
- Memory and context management
- Planner (automated multi-step planning)
- Good choice for Microsoft/Azure ecosystem

**Haystack (deepset):**
- Pipeline-based architecture for RAG and search
- Components: Document stores, retrievers, readers, generators
- Focus on production NLP pipelines
- Good for search-heavy applications

### Agent Architecture

**ReAct (Reasoning + Acting):**
The foundational agent loop. Model alternates between Thought (reasoning), Action (tool call), Observation (tool result).
```
Thought: I need to find the current population of Tokyo
Action: web_search("Tokyo population 2025")
Observation: Tokyo population is approximately 13.96 million
Thought: I have the answer
Final Answer: Tokyo's population is approximately 13.96 million
```

**Function Calling / Tool Use:**
- Model outputs structured JSON tool calls instead of free text
- More reliable than ReAct parsing
- Supported natively by GPT-4, Claude, Llama 3, Gemini
- Study: tool schemas (name, description, parameters), parallel tool calls, tool result injection

**Planning strategies:**
- Single-step (direct tool call)
- Multi-step (sequential tool calls with reasoning)
- Plan-and-execute (generate full plan first, then execute steps)
- Tree-of-Thought (explore multiple reasoning paths)

**Memory for agents:**
- In-context (conversation history in prompt): Simple but limited by context window
- External short-term (Redis, PostgreSQL): Store conversation summaries
- Long-term memory (vector DB + entity store): Recall past interactions semantically
- Episodic memory: Remember specific events (MemGPT approach)

**Multi-agent systems:**
- Supervisor pattern: One orchestrator agent routes to specialized sub-agents
- Collaborative pattern: Agents debate and verify each other's outputs
- Hierarchical: Manager → sub-managers → worker agents
- Swarm: Decentralized, agents hand off to each other based on context

**Agent evaluation:**
- Trajectory evaluation: Was the sequence of tool calls correct?
- Final answer correctness: Is the output right?
- Efficiency: Did the agent take unnecessary steps?
- Robustness: Does it handle tool failures gracefully?

### Model Context Protocol (MCP)

MCP is the emerging standard (from Anthropic) for connecting LLMs to external tools and data sources.

Study:
- MCP architecture: Client (LLM application) + Server (tool/data provider)
- Three primitives: Resources (read data), Tools (execute actions), Prompts (reusable templates)
- Transport: stdio (local process), HTTP + SSE (remote server)
- Building MCP servers: Python SDK, TypeScript SDK
- Authentication and security for remote MCP servers
- Why MCP matters: Standardizes the integration layer so tools work with any LLM — like USB-C for AI

### LLM Gateway Pattern

A centralized proxy/gateway for all LLM traffic in your organization.

**LiteLLM:**
- Unified API for 100+ LLM providers (OpenAI, Anthropic, Bedrock, Vertex AI, self-hosted)
- Load balancing across multiple deployments of the same model
- Fallback routing (primary fails → backup model)
- Rate limiting per API key
- Cost tracking and budget enforcement
- Prompt caching pass-through
- Deploy as a proxy server

**Portkey:**
- Similar to LiteLLM with additional features
- Semantic caching, guardrails, load balancing
- Managed cloud version available

**Kong AI Gateway:**
- Enterprise API gateway with LLM-specific plugins
- Rate limiting, authentication, logging, semantic caching
- Good for organizations already using Kong

---

# Phase 8: Monitoring, Observability & Reliability for ML

> **Goal:** Apply and extend your SRE expertise to ML-specific operational challenges. This is where your background gives you a significant head start over pure ML engineers.
>
> **Time estimate:** 3–4 weeks

---

## 8.1 ML-Specific Monitoring

### Model Drift Detection

**Data drift (covariate shift):** The distribution of input features changes over time. The model was trained on one distribution and is now seeing a different one.

**Concept drift:** The relationship between inputs and outputs changes over time (the world changed). The model's predictions become wrong even though inputs look similar.

**Prediction drift:** The distribution of model outputs changes over time. Leading indicator of concept drift.

**Label drift:** The distribution of true labels changes (if you have ground truth).

**Statistical tests for drift:**
- KS test (Kolmogorov-Smirnov): Continuous features. Tests if two distributions are different.
- Chi-squared test: Categorical features. Tests independence from expected distribution.
- Population Stability Index (PSI): Compares feature distributions between training and production
- Maximum Mean Discrepancy (MMD): Distribution-level comparison
- Wasserstein distance: Earth mover's distance between distributions

**Tools:**
- Evidently AI: Open-source ML monitoring. Reports for data drift, model performance, data quality. Integrates with Grafana.
- Whylabs: Cloud-based, uses WhyLogs library. Real-time drift monitoring.
- Arize: Enterprise ML observability. Strong embedding drift visualization.
- Nannyml: Confidence-based performance monitoring without labels (CBPE)

### GPU Observability

**DCGM (Data Center GPU Manager):**
- NVIDIA's toolkit for GPU telemetry in data centers
- Metrics: GPU utilization, memory utilization, SM occupancy, memory bandwidth, temperature, power draw, PCIe throughput, NVLink throughput, error counts (ECC, XID errors)
- DCGM Exporter: Prometheus-compatible exporter for Kubernetes
- Study: DCGM health checks, error codes (XID errors and their meanings), integration with Grafana dashboards

**nvidia-smi:**
- `nvidia-smi dmon`: Continuous monitoring of GPU metrics
- `nvidia-smi nvlink -s`: NVLink status and bandwidth
- `nvidia-smi mig -lgi`: MIG instance status
- `nvidia-smi topo -m`: GPU topology (PCIe vs NVLink connectivity)

**Key GPU metrics to alert on:**
- GPU utilization < 60% for extended periods (underutilization, wasted compute)
- Memory utilization > 95% (risk of OOM errors)
- Temperature > 85°C (throttling or hardware risk)
- ECC memory errors (hardware degradation)
- XID errors (GPU fault codes — many indicate hardware failure)
- Power exceeding TDP (thermal design power)

### SLOs and SLIs for Model Serving

**Adapt SRE concepts to ML:**

| SLI Type | Traditional | ML Serving |
|----------|-------------|------------|
| Latency | Request p99 < 200ms | TTFT p95 < 500ms, ITL p95 < 50ms |
| Availability | Error rate < 0.1% | Successful completions / Total requests |
| Throughput | Requests/sec | Tokens/sec (input + output) |
| Quality | N/A | Faithfulness > 0.8, Answer relevancy > 0.7 |
| Freshness | Data age < 1 hour | Model age, knowledge cutoff, RAG index age |

**Defining SLOs for model quality:**
This is harder than infrastructure SLOs. Options:
- Sample-based evaluation: Evaluate 1% of production traffic automatically
- User feedback signals: Track thumbs up/down rates, regeneration requests
- Business metrics: Downstream task completion rates, user session length
- Canary-based: Compare new model quality vs production continuously

## 8.2 Cost Monitoring and Optimization

**What to track:**
- GPU instance cost per hour (on-demand vs spot)
- Cost per 1,000 tokens (input and output separately)
- Cost per request
- Cost per user
- GPU utilization (low utilization = waste)
- Cache hit rate (higher = lower cost)
- Token usage by model (which model is most expensive?)

**Tools:**
- Custom dashboards: Combine GPU billing data with token usage logs
- Cloud cost management: AWS Cost Explorer, GCP Billing, Azure Cost Management
- Infracost: IaC cost estimation
- FinOps practices applied to GPU spend

**Optimization levers:**
- Right-size GPU type to model size
- Quantize models to fit more on fewer GPUs
- Maximize batch size to improve GPU utilization
- Semantic caching to reduce redundant LLM calls
- Model routing (cheap model for simple queries)
- Spot instances for training and stateless inference
- Reserved instances for baseline serving capacity

## 8.3 Incident Response for ML Systems

**Types of ML incidents:**

**Performance degradation:** Model quality drops. Could be data drift, concept drift, upstream data pipeline failure, or a bad model deployment. Harder to detect than crashes because the system is still "running."

**Infrastructure failure:** GPU failure, serving pod crash, KV cache OOM. Similar to traditional infrastructure incidents.

**Data pipeline failure:** Training data is stale, feature store not updated, data quality issues upstream. Can silently corrupt model predictions.

**Model serving failure:** Model loaded incorrectly, wrong model version deployed, out-of-memory errors under load.

**Runbooks for ML incidents:**
- Model quality drop: Check data freshness → check feature store → check model version → compare predictions with expected distribution → rollback if needed
- Serving OOM: Check request length distribution → check concurrent request count → reduce max-model-len or max-num-seqs → add replicas
- High latency: Check GPU utilization → check queue depth → check batch configuration → check if preemption is occurring

## 8.4 Capacity Planning for ML

**Training capacity planning:**
- Estimate compute requirements: FLOPs = 6 × Parameters × Tokens
- Factor in utilization efficiency (40–60% effective GPU utilization for distributed training)
- Plan for retries, checkpointing overhead, data loading overhead
- Model: how many training runs per month? What size models?

**Inference capacity planning:**
- Estimate tokens per request (input + output)
- Estimate requests per second (peak)
- Calculate required throughput: tokens/sec = requests/sec × tokens/request
- Calculate required GPUs: tokens/sec / (max_tokens_per_sec_per_GPU)
- Add headroom: 30–50% over peak for autoscaling buffer

**Autoscaling for inference:**
- Horizontal scaling: Add more serving pods
- Metrics for autoscaling: Request queue depth, GPU utilization, tokens/sec
- Kubernetes HPA with custom metrics (KEDA for queue-based scaling)
- Warm-up time: LLMs take 1–5 minutes to load — plan ahead with predictive scaling

---

# Phase 9: Platform Engineering for ML

> **Goal:** Build the internal platform that abstracts ML infrastructure complexity, enabling data scientists and ML engineers to focus on models rather than infrastructure.
>
> **Time estimate:** Ongoing — this is a senior-level, continuous practice

---

## 9.1 Internal Developer Platforms for ML

### What an ML Platform Provides

An ML platform abstracts the complexity of infrastructure from ML teams. Instead of "provision a GPU cluster, set up storage, configure networking, install dependencies, set up experiment tracking," the data scientist runs one command and gets a working environment.

**Platform primitives to build:**
- **Notebook environments:** JupyterHub with GPU access, pre-installed libraries, auto-scaling
- **Training job submission:** Simple CLI/API to submit training runs to the cluster
- **Experiment tracking:** Centralized MLflow/W&B instance with auth and team isolation
- **Feature store:** Managed Feast instance with data engineering team support
- **Model registry:** Centralized registry with review and approval workflows
- **Serving infrastructure:** Self-service model deployment (submit a model → get an endpoint)
- **Dataset catalog:** Searchable catalog of approved, versioned datasets
- **Cost visibility:** Show teams their GPU spend

### JupyterHub for ML Teams

Study:
- JupyterHub on Kubernetes (KubeSpawner)
- GPU profile selection (choose GPU type and count)
- Pre-built Docker images with ML libraries (PyTorch, JAX, RAPIDS)
- Persistent storage for notebooks and datasets
- Resource quotas per user/team
- Single Sign-On (SSO) integration (LDAP, SAML, OAuth)
- Idle culling to reclaim resources from inactive notebooks

## 9.2 Self-Service Training and Serving Abstractions

### Training Platform

**Goals:** Data scientist submits a training config → platform handles everything else.

**What the platform manages:**
- Selecting appropriate GPU type and count
- Provisioning the Kubernetes job with correct resources
- Setting up distributed training environment
- Mounting datasets from the data catalog
- Connecting to experiment tracking
- Checkpointing and fault tolerance
- Cost attribution

**Tools:** Custom CLI wrapping `kubectl`, Kubeflow Pipelines, Ray Job API, custom web UI.

### Serving Platform

**Goals:** ML engineer submits a model artifact → platform handles deployment, scaling, monitoring.

**What the platform manages:**
- Optimal GPU type selection based on model size
- vLLM/TorchServe configuration
- Kubernetes deployment with health checks
- Auto-scaling configuration
- Monitoring and alerting setup
- A/B testing and canary deployment support

## 9.3 Multi-Tenancy and Resource Isolation

**Namespace isolation:** Each team gets its own Kubernetes namespace with resource quotas.

**GPU resource quotas:** Limit total GPU hours per team per day/month. Enforce with Kubernetes ResourceQuota and LimitRange.

**Priority classes:** Research (low priority, preemptible), production (high priority, guaranteed). Production jobs are never preempted; research jobs use spare capacity.

**Volcano / Yunikorn scheduling:** Advanced batch scheduling for ML workloads.
- Fair-share scheduling between teams
- Gang scheduling for distributed training
- Queue management with priority
- Preemption policies

**Network policies:** Isolate training jobs from each other. Prevent data exfiltration.

## 9.4 Kubernetes Operators for ML

Custom operators extend Kubernetes for ML-specific workloads.

**Study existing operators:**
- Kubeflow Training Operator (PyTorchJob, TFJob, MPIJob)
- Ray Operator (KubeRay: RayCluster, RayJob, RayService)
- Milvus Operator (managed Milvus deployment)
- Strimzi (Kafka operator — for streaming data pipelines)

**Building custom operators:**
- When to build: If existing operators don't meet your needs
- Tools: Operator SDK, Kubebuilder, controller-runtime
- Pattern: Reconciliation loop — desired state (spec) vs actual state → take actions to converge
- Study CRD (Custom Resource Definition) design

## 9.5 Cost Allocation and Chargeback

**Why it matters:** Without cost visibility, teams over-provision and waste GPU resources. Showback/chargeback creates accountability.

**What to track:**
- GPU hours per team, per project, per user
- Storage costs (dataset storage, checkpoint storage)
- Network egress costs
- Ratio of utilized vs allocated GPU time

**Implementation:**
- Kubernetes labels: Label every workload with team, project, environment
- Usage aggregation: Parse GPU metrics by label (DCGM + Prometheus)
- Cost model: GPU cost/hour × utilization hours per label
- Dashboards: Per-team cost dashboards in Grafana
- Alerting: Alert when team exceeds budget threshold

---

# Cross-Cutting Topics

> These topics span all phases. Integrate them throughout your learning, don't treat them as separate.

---

## Cloud-Specific ML Services

**Pick one cloud to go deep on based on your organization. Here are the key services for each:**

### AWS

| Service | What It Does | When to Use |
|---------|-------------|-------------|
| SageMaker | End-to-end ML platform: notebooks, training, serving | When you want managed ML on AWS |
| SageMaker Training | Managed distributed training jobs | GPU training without managing clusters |
| SageMaker Endpoints | Model serving with auto-scaling | Simple model serving on AWS |
| SageMaker Pipelines | ML pipeline orchestration (Kubeflow-like) | AWS-native MLOps |
| SageMaker Feature Store | Managed feature store | If not self-hosting Feast |
| Bedrock | Managed API for foundation models (Claude, Llama, etc.) | When you don't want to self-host LLMs |
| EKS with GPU nodegroups | Self-managed Kubernetes with GPU nodes | When you need full control |
| p3/p4/p5 instances | NVIDIA V100/A100/H100 GPU instances | GPU compute for training and serving |
| FSx for Lustre | High-performance file system for training | When S3 I/O is the bottleneck |
| ECR | Container registry | Store training and serving Docker images |

### GCP

| Service | What It Does |
|---------|-------------|
| Vertex AI | Managed ML platform (training, pipelines, serving, registry) |
| Vertex AI Workbench | Managed JupyterHub |
| Vertex AI Training | Managed distributed training |
| Vertex AI Endpoints | Model serving with autoscaling |
| Vertex AI Pipelines | Managed Kubeflow Pipelines |
| Model Garden | Pre-trained models and fine-tuning |
| GKE with GPU node pools | Managed Kubernetes with GPU nodes |
| A2/A3 instances | NVIDIA A100/H100 GPU instances |
| TPU v4/v5 | Google's custom AI accelerators |
| Filestore | Managed NFS for training data |

### Azure

| Service | What It Does |
|---------|-------------|
| Azure Machine Learning | End-to-end ML platform |
| Azure ML Pipelines | ML workflow orchestration |
| Azure ML Endpoints | Model serving |
| Azure OpenAI Service | Managed access to OpenAI models |
| AKS with GPU node pools | Managed Kubernetes with GPU nodes |
| NCv3/NCasT4/NDv2 instances | GPU instances (V100, T4, A100) |
| Azure Blob Storage | Object storage for datasets and models |

## Security and Compliance for ML

### Data Security in ML Pipelines

- **Data encryption:** At-rest (S3 SSE, GCS CMEK) and in-transit (TLS) for training data and model artifacts
- **Access control:** IAM roles for training jobs (least-privilege), data access audit logging
- **PII handling:** Detect and anonymize PII before using in training data (Microsoft Presidio, cloud DLP services)
- **Data residency:** Ensure training data doesn't cross regional boundaries in violation of regulations

### Model Security

- **Model access control:** Who can download model weights? Who can call the serving endpoint?
- **API authentication:** JWT, API keys, OAuth2 for serving endpoints
- **Model watermarking:** Embedding provenance information in model weights or outputs
- **Adversarial robustness:** Understanding adversarial examples and how to harden models
- **Prompt injection:** Input validation for LLM applications (see Phase 7.5)

### Compliance

- **GDPR:** Right to erasure — if training data is deleted, does the model need to be retrained?
- **HIPAA:** Protected health information (PHI) in training data — special handling requirements
- **SOC 2:** Access controls, audit logging, change management for ML systems
- **Model cards:** Documenting model intended use, performance characteristics, limitations, biases
- **NIST AI RMF:** AI Risk Management Framework — emerging standard for AI governance

## Networking for ML

### VPC and Security Groups for GPU Clusters

- **Placement groups:** AWS cluster placement groups put GPU instances physically close for better network performance
- **Enhanced networking:** AWS ENA, GCP gVNIC — higher bandwidth, lower latency networking
- **VPC design:** Separate subnets for training nodes, serving nodes, data access
- **Security groups:** Restrict inter-node communication to necessary ports (NCCL, SSH, monitoring)
- **NAT gateways:** Allow GPU nodes in private subnets to reach internet for package installs

### Service Mesh for ML Serving

- **Istio/Linkerd:** mTLS between services, traffic management, observability
- **Traffic splitting:** Use Istio VirtualService for A/B testing and canary deployments
- **Circuit breakers:** Prevent cascading failures when model serving is overloaded
- **Retries and timeouts:** Configure appropriately for LLM latency (higher timeouts for long generation)

## Container Optimization for ML

### CUDA Base Images

- **NVIDIA CUDA images:** cuda:12.1-cudnn8-runtime-ubuntu22.04 — choose runtime vs devel, include cuDNN for training
- **PyTorch base images:** pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
- **TensorFlow images:** tensorflow/tensorflow:2.16.1-gpu
- **Multi-stage builds:** Separate build stage (large) from runtime stage (smaller)

### Reducing Image Size

- Use runtime (not devel) CUDA images for serving (devel includes compiler tools ~4 GB)
- Multi-stage builds: compile in large image, copy only binaries to runtime image
- pip install with `--no-cache-dir`
- Combine RUN commands to reduce layers
- Use .dockerignore to exclude large files

### Container Security for ML

- Run as non-root user
- Read-only root filesystem where possible
- Drop unnecessary Linux capabilities
- Use distroless or minimal base images
- Scan images for vulnerabilities (Trivy, Snyk)

---

## Recommended Study Order Summary

```
Week 1-2:   Phase 1 (ML Fundamentals) + Phase 2 start (Neural Nets)
Week 3-4:   Phase 2 complete (Deep Learning, Transformers)
Week 5-6:   Phase 3 (Data Engineering for ML)
Week 7-8:   Phase 4 start (Distributed Training, GPU Clusters)
Week 9-10:  Phase 4 complete + Phase 5 (Model Serving)
Week 11-12: Phase 6 (MLOps Pipelines)
Week 13-16: Phase 7 — LLM/GenAI Infrastructure (prioritize 7.1 and 7.2)
Week 17-18: Phase 8 (Monitoring and Reliability)
Week 19+:   Phase 9 (Platform Engineering) — ongoing, senior-level
Ongoing:    Cross-cutting topics woven throughout
```

## Tools Summary by Category

| Category | Primary Tools | Secondary Tools |
|----------|--------------|-----------------|
| **Orchestration** | Airflow, Kubeflow Pipelines | Prefect, Dagster, ZenML, Metaflow |
| **Experiment Tracking** | MLflow | Weights & Biases, Neptune |
| **Vector Databases** | pgvector, Weaviate | Milvus, Pinecone, Qdrant |
| **LLM Serving** | vLLM | TGI, TensorRT-LLM + Triton |
| **Model Serving** | Triton, TorchServe | BentoML, Seldon, FastAPI |
| **Fine-tuning** | Axolotl, TRL | Unsloth, LLaMA-Factory |
| **LLM Frameworks** | LangChain, LangGraph | LlamaIndex, Semantic Kernel |
| **LLM Observability** | Langfuse | LangSmith, Arize Phoenix |
| **ML Monitoring** | Evidently AI | Whylabs, Arize |
| **GPU Monitoring** | DCGM Exporter, Prometheus | nvidia-smi |
| **Data Versioning** | DVC | LakeFS |
| **Feature Stores** | Feast | Tecton |
| **Data Validation** | Great Expectations | Pandera |
| **Model Registry** | MLflow Registry | HuggingFace Hub |
| **Embeddings Serving** | TEI (HF) | Infinity |
| **Batch Scheduling** | Volcano | Yunikorn |
| **LLM Gateway** | LiteLLM | Portkey |
| **Guardrails** | Guardrails AI | NeMo Guardrails |
| **Eval Frameworks** | RAGAS, DeepEval | Promptfoo, Braintrust |
| **Container Registry** | ECR, GCR, Harbor | Quay |
| **IaC** | Terraform | Pulumi |
| **GitOps** | ArgoCD, Flux | — |
| **Distributed Training** | PyTorch FSDP | DeepSpeed, Horovod |
| **Hyperparameter Tuning** | Optuna | Ray Tune |
| **Training Operators** | Kubeflow Training Operator | KubeRay |
