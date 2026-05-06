# Daily Practice Problem Sheet — #001
## Topic: ML Fundamentals — The Training Loop, Evaluation Metrics & Overfitting
### Phase 1 | Difficulty: Beginner–Intermediate | Time: 90–120 minutes

---

> **How to use this sheet:**
> - Part A → Do on paper or in your head. No Googling allowed.
> - Part B → Code it up locally. Actually run it.
> - Part C → Design/architecture thinking. Write your answer in a `.txt` or comment block.
> - Check your answers only after attempting. Struggle is the point.

---

## Part A — Concept Questions (No Code, No Google)

### Q1. The Accuracy Trap
You build a fraud detection model on a dataset where **1% of transactions are fraud**.
Your model predicts "not fraud" for every single transaction.

- a) What accuracy does it achieve?
- b) Why is this score completely misleading?
- c) Which two metrics should you use instead, and why?
- d) If your model catches 80 out of 100 actual frauds but also raises 40 false alarms — what is the Precision and Recall?

---

### Q2. Overfitting Diagnosis
Your friend trains a model on 10,000 emails for spam detection.

| Dataset | Accuracy |
|---------|----------|
| Training set | 99.2% |
| Validation set | 61.4% |
| Test set | 60.9% |

- a) What is this problem called?
- b) What does this gap tell you about what the model learned?
- c) Name **four** concrete techniques to fix this. Don't just name them — write one line explaining *why* each one works.

---

### Q3. Learning Rate Intuition
Explain in plain English what the learning rate controls in gradient descent.

- a) What happens visually on the loss curve if learning rate is **too high**?
- b) What happens if it is **too low**?
- c) What does a **good** learning rate loss curve look like?
- d) Your training loss is oscillating wildly and sometimes goes UP between epochs. What is most likely wrong?

---

### Q4. Bias vs Variance
Match each scenario to **high bias**, **high variance**, or **well-balanced**:

| Scenario | Answer |
|----------|--------|
| Train acc: 98%, Val acc: 97% | ? |
| Train acc: 72%, Val acc: 70% | ? |
| Train acc: 99%, Val acc: 55% | ? |
| Train acc: 75%, Val acc: 74%, but human-level is 95% | ? |

- Why does a very simple model (e.g., single decision tree with depth=1) have high bias?
- Why does a very complex model (e.g., 500-layer neural net on 100 samples) have high variance?

---

### Q5. Train / Val / Test Split Logic
You have 50,000 labeled samples. Answer the following:

- a) What split ratio would you use and why?
- b) Your colleague normalizes (standardizes) the entire dataset **before** splitting. What is the problem with this? What is this called?
- c) You fine-tune hyperparameters until your validation accuracy is 94%. You run on the test set and get 94.1%. Was the test set "truly held out"? Why or why not?
- d) What is k-fold cross-validation and when would you use it over a fixed validation split?

---

## Part B — Hands-On Coding

> **Setup:** Python 3.10+, scikit-learn, pandas, matplotlib installed.
> Run everything in a `.py` file or Jupyter notebook. Save your plots.

---

### Problem 1 — Build and Evaluate a Classifier End-to-End

**Dataset:** `sklearn.datasets.load_breast_cancer()`
**Task:** Binary classification (malignant vs benign)

**Step-by-step:**

```
1. Load the dataset. Print:
   - Total number of samples
   - Number of features
   - Class distribution (how many malignant vs benign)
   - Feature names

2. Split into train (70%) / val (15%) / test (15%)
   - Use random_state=42 for reproducibility
   - Stratify the split to preserve class ratio

3. Normalize features using StandardScaler
   - IMPORTANT: Fit the scaler ONLY on training data
   - Apply (transform) to val and test separately
   - Why? Write this as a comment in your code.

4. Train three models:
   - LogisticRegression
   - RandomForestClassifier (n_estimators=100)
   - A DummyClassifier with strategy='most_frequent' (this is your baseline)

5. For each model, compute on the VALIDATION set:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - AUC-ROC

6. Print a comparison table of all three models across all five metrics.

7. Plot the ROC curve for all three models on the same figure.
   - Label each curve with its AUC score
   - Add a diagonal baseline (random classifier line)
   - Save the plot as roc_curves.png

8. Print the confusion matrix for your best model on the VALIDATION set.
   - Interpret it: how many false positives? False negatives?
   - Which is more dangerous in cancer diagnosis — a false positive or a false negative? Write your answer as a comment.

9. Finally, evaluate your best model on the TEST set.
   - Report all five metrics.
   - Is it significantly different from validation? What might cause a big gap?
```

---

### Problem 2 — Observe Overfitting in Real Time

**Dataset:** `sklearn.datasets.make_classification(n_samples=500, n_features=20, random_state=42)`

```
1. Train a DecisionTreeClassifier with max_depth ranging from 1 to 25.

2. For each depth value, record:
   - Training accuracy
   - Validation accuracy

3. Plot both curves on the same graph (depth on x-axis, accuracy on y-axis).
   - Label the region where the model underfits
   - Label the region where the model overfits
   - Mark the optimal depth with a vertical dashed line
   - Save as overfitting_curve.png

4. Answer in a comment:
   - At what depth does overfitting clearly begin?
   - What is the validation accuracy at depth=1 vs your optimal depth?
   - What would you set max_depth to in production and why?
```

---

### Problem 3 — The Imbalanced Dataset Problem

**Dataset:** Create it yourself:

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000,
    n_features=10,
    weights=[0.97, 0.03],   # 97% class 0, 3% class 1
    random_state=42
)
```

```
1. Train a LogisticRegression on this dataset (default settings).

2. Print accuracy. Then say out loud — is this a good model?

3. Print the classification report (precision, recall, f1 for each class).
   Notice how different class 0 and class 1 metrics are.

4. Now fix the imbalance using class_weight='balanced' in LogisticRegression.
   Re-train and print the classification report again.

5. Compare the two models:
   - Which has higher accuracy?
   - Which catches more actual fraud (class 1)?
   - Which would you deploy and why?

6. BONUS: Try oversampling the minority class using SMOTE from the imbalanced-learn library.
   Compare results with the class_weight approach.
```

---

## Part C — Design & Thinking

> Write your answers as plain text. No code needed. 3–5 sentences each.

---

### Design Q1 — You are on call (SRE → ML crossover)
Your team deployed an ML model that predicts server failure 30 minutes in advance.
The model was working fine for 3 months. This week, you notice the **recall dropped from 91% to 54%** — it is missing many actual failures now.

- What are three possible causes of this sudden drop?
- How would you investigate? What would you check first?
- Would you roll back the model? What information do you need before deciding?

---

### Design Q2 — Choosing the Right Metric
For each use case below, choose the most appropriate evaluation metric and justify it:

| Use Case | Your Chosen Metric | Why |
|----------|-------------------|-----|
| Detecting cancer from X-rays | ? | ? |
| Spam email filter | ? | ? |
| Recommending movies on Netflix | ? | ? |
| Predicting next-day stock price | ? | ? |
| Autonomous vehicle pedestrian detection | ? | ? |

---

### Design Q3 — Explaining ML to Your SRE Manager
Your manager has a traditional software engineering background. They ask:
*"Why can't we just write rules to detect fraud? Why do we need this ML thing?"*

Write a 5-sentence explanation that covers:
- Why rules break down at scale
- What ML learns that rules cannot
- One concrete example with numbers

---

## Deliverables Checklist

Before moving to the next DPP, make sure you have:

- [ ] Answered all Part A questions on paper
- [ ] `classifier_evaluation.py` or notebook running without errors
- [ ] `roc_curves.png` saved
- [ ] `overfitting_curve.png` saved
- [ ] Imbalanced dataset problem completed with comparison printed
- [ ] Part C design answers written down somewhere
- [ ] Bonus SMOTE attempted (optional but recommended)

---

## Key Concepts Covered in This DPP

| Concept | Where Practiced |
|---------|----------------|
| Accuracy vs Precision vs Recall vs F1 vs AUC | A-Q1, B-Problem1 |
| Overfitting and underfitting | A-Q2, B-Problem2 |
| Learning rate behavior | A-Q3 |
| Bias-variance tradeoff | A-Q4 |
| Train/val/test split and data leakage | A-Q5, B-Problem1 |
| Class imbalance | B-Problem3 |
| Confusion matrix interpretation | B-Problem1 |
| Production model degradation | C-Q1 |
| Choosing metrics for business problems | C-Q2 |

---

## What's Coming Next

**DPP #002** — Topic: Supervised Learning Algorithms Deep Dive (Decision Trees, Random Forests, XGBoost)
You will implement, tune, and compare tree-based models on a real-world dataset.

---

*DPP Series: DevOps/SRE → AI Infra / MLOps Engineer*
*Total DPPs in series: ~90 | Current: 1/90*
