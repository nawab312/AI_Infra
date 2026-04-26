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
