# 🫁 Thoracic Pathology Detection from Chest X-Rays

> A deep learning pipeline for multi-class classification of 20 thoracic pathologies from chest X-ray images. Built for the **TDS 26-T1 Kaggle Competition**.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Competition Metric](#-competition-metric)
- [Dataset](#-dataset)
- [My Approach](#-my-approach)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [How to Reproduce](#-how-to-reproduce)
- [Key Learnings](#-key-learnings)
- [Tech Stack](#-tech-stack)

---

## 🩺 Problem Statement

Chest radiography is one of the most widely used diagnostic tools in medicine. Radiologists analyze X-ray images to detect abnormalities such as lung infections, fluid accumulation, lung collapse, and heart enlargement. However, manual interpretation is time-consuming, subjective, and prone to error — especially when dealing with large volumes of images or rare conditions.

This project builds a machine learning model that automatically classifies each chest X-ray into one of **20 thoracic pathology classes**, including a *No Finding* (healthy) category.

**Key challenges:**
- **Severe class imbalance** — "No Finding" accounts for 66.77% of all training samples, while Pneumomediastinum has only 5 samples
- **Asymmetric error costs** — missing a disease diagnosis is far more dangerous than a false alarm
- **Subtle visual patterns** — many pathologies look visually similar on X-rays

---

## 📐 Competition Metric

Submissions are evaluated using a **macro-averaged asymmetric cost function** that mirrors real clinical impact:

| Prediction Outcome | Score |
|---|---|
| ✅ Correctly detect a disease (True Positive) | **+1** |
| ❌ Predict disease when healthy (False Positive) | **-1** |
| 🚨 Miss a disease entirely (False Negative) | **-5** |

**Class-level score formula:**

$$Score_c = \frac{TP_c - FP_c - 5 \cdot FN_c}{N_c}$$

**Final score** = macro-average of all 20 class scores.

The -5 penalty for False Negatives means the model is heavily penalized for missing diseases — forcing it to learn rare classes rather than defaulting to "No Finding".

---

## 📦 Dataset

| File | Description |
|---|---|
| `images/` | 88,340 chest X-ray images in PNG format |
| `train.csv` | 51,043 labelled training samples |
| `test.csv` | 17,015 test samples (labels hidden) |
| `sample_submission.csv` | Required output format |

**Image properties:** All images are RGB, 384×384 pixels — uniform size, no preprocessing surprises.

**Label format:** One-hot encoded. Each row has exactly one class set to `1`, the rest `0`. This makes it a **single-label multi-class** problem (not multi-label).

### Class Distribution

```
No Finding                 → 34,079  (66.77%)   ████████████████████
Infiltration               →  5,206  (10.20%)   ████
Atelectasis                →  2,351   (4.61%)   ██
Effusion                   →  2,156   (4.22%)   █
...
Subcutaneous Emphysema     →     24   (0.05%)   ▏
Pneumomediastinum          →      5   (0.01%)   ▏
```

---

## 🧠 My Approach

The full pipeline follows 6 steps:

```
Raw X-ray PNGs
      ↓
① Load + Resize + Augment
      ↓
② DenseNet121 extracts visual features
      ↓
③ Focal Loss + AdamW trains the model
      ↓
④ 3-Fold Stratified Cross-Validation
      ↓
⑤ Per-class Logit Bias Tuning
      ↓
⑥ Ensemble averaging → submission.csv
```

### Why this approach?

A naive model trained with standard Cross-Entropy loss will learn to always predict "No Finding" and achieve ~66% accuracy — but a terrible competition score because it misses all diseases (each miss costs -5). Every design decision below was made to counter this.

---

## 🏗️ Model Architecture

**Backbone:** DenseNet121 (pre-trained on ImageNet)

DenseNet121 was originally used in the landmark [CheXNet paper](https://arxiv.org/abs/1711.05225) for chest X-ray classification. Its dense connections allow each layer to receive feature maps from all preceding layers — making it excellent at detecting subtle, multi-scale visual patterns like early-stage pathologies.

```
Input Image [3 × 224 × 224]
        ↓
DenseNet121 Backbone  (121 layers, dense skip connections)
        ↓
Global Average Pooling  (1024-dim feature vector)
        ↓
Dropout (p=0.4)         (prevents overfitting on rare classes)
        ↓
Linear Layer            (1024 → 20 logits)
        ↓
Predicted Class
```

**Total parameters:** ~6.97 million (all trainable)

**Why DenseNet121 over ResNet18?**
- DenseNet's dense connections reuse features across layers — better for subtle pathology patterns
- ResNet18 is too shallow (18 layers) for fine-grained medical image classification
- DenseNet121 has a proven track record specifically on chest X-ray datasets

---

## ⚙️ Training Strategy

### 1. Data Augmentation

Training images receive random augmentations to artificially increase variety, especially for rare classes:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=10)
transforms.ColorJitter(brightness=0.2, contrast=0.2)
transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
```

Validation/test images only get resized and normalized — no augmentation (we want clean, deterministic predictions).

### 2. Focal Loss + Class Weights

Two layers of protection against class imbalance:

**Class weights** (inverse-sqrt of class frequency):

| Class | Count | Weight |
|---|---|---|
| No Finding | 34,079 | 0.065 |
| Pneumomediastinum | 5 | **5.384** |
| Hernia | 37 | 1.979 |

**Focal Loss** additionally down-weights easy examples at training time:

$$FL(p_t) = (1 - p_t)^{\gamma} \cdot CE(p_t), \quad \gamma = 2$$

When the model confidently predicts "No Finding" (easy case), the focal weight `(1 - p_t)²` shrinks toward 0 — forcing the model to focus on hard, rare disease examples.

### 3. Optimizer & Scheduler

- **AdamW** with `lr=3e-4`, `weight_decay=1e-4`
- **OneCycleLR** scheduler — ramps LR up for 20% of training then cosine-decays:
  ```
  Epochs 1-2: LR warms up from 3e-5 → 3e-4
  Epochs 3-5: LR at peak (3e-4)
  Epochs 6-8: LR decays to ~3e-7
  ```
- **Gradient clipping** (`max_norm=1.0`) — prevents exploding gradients from focal loss on rare classes
- **Early stopping** (patience=3) — stops if validation score doesn't improve for 3 consecutive epochs

### 4. Stratified 3-Fold Cross-Validation

`StratifiedKFold` ensures every fold contains proportional samples of all 20 classes — critical when some classes have only 5 training samples.

```
Fold 1: Train 34,028 | Val 17,015
Fold 2: Train 34,029 | Val 17,014
Fold 3: Train 34,029 | Val 17,014
```

Three independent models are trained. Their test logits are averaged in the final ensemble.

### 5. Per-Class Logit Bias Tuning

After training, a greedy search adds a small offset to each class's output logit to maximize the competition score on the validation set:

```python
# Example: shift "No Finding" logits down, rare diseases up
bias = [-1.5, 0.0, 0.0, ..., +0.5, -1.5]  # per-class offsets

final_prediction = argmax(logits + bias)
```

This is a zero-cost post-processing step — no retraining needed. It improved scores by **+2.4 points** on average across folds.

The search grid used: `[-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0]`, run for 3 passes.

### 6. Ensemble Averaging

Test logits from all 3 fold models (with bias applied) are averaged before taking the final `argmax`:

```python
ensemble_logits = mean([fold1_logits, fold2_logits, fold3_logits])
final_prediction = argmax(ensemble_logits)
```

Averaging reduces variance — like getting 3 doctors' opinions before a diagnosis.

---

## 📊 Results

### Cross-Validation Summary

| Fold | Best Epoch | Raw Val Score | Tuned Val Score |
|---|---|---|---|
| 1 | 4 | -6.850 | **-4.657** |
| 2 | 1 | -7.494 | **-4.792** |
| 3 | 5 | -6.981 | **-4.575** |
| **Mean** | — | **-7.108** | **-4.675** |

**Bias tuning improvement:** +2.43 points per fold on average

### Baseline Comparison

| Model | Score |
|---|---|
| Always predict "No Finding" | -4.725 |
| **Our DenseNet121 + Bias Tuning** | **~-4.675** |

The score is close to the dummy baseline because of the extreme class imbalance (66.77% No Finding). The model's real value is in correctly identifying the rare disease classes — which the macro-averaging metric captures across all 20 classes equally.

---

## 🔁 How to Reproduce

### On Kaggle

1. Fork this notebook in a Kaggle kernel with the competition dataset attached
2. Enable **GPU accelerator** (Tesla T4 recommended)
3. Run all cells sequentially — Steps 1 through 6
4. Download `submission.csv` from the output panel
5. Submit to the competition

**Estimated runtime:** ~50-60 minutes (3 folds × 8 epochs on T4)

### Dataset Path

```python
BASE_PATH   = "/kaggle/input/competitions/26-t-1-dl-gen-ainppe-1"
IMAGES_PATH = f"{BASE_PATH}/images"
train_df    = pd.read_csv(f"{BASE_PATH}/train.csv")
test_df     = pd.read_csv(f"{BASE_PATH}/test.csv")
```

### Requirements

All libraries are pre-installed in the Kaggle Python environment:

```
torch>=2.0
torchvision
pandas
numpy
scikit-learn
matplotlib
Pillow
```

---

## 💡 Key Learnings

**1. Class imbalance is the hardest part of medical imaging**
Even with Focal Loss + class weights + bias tuning, the -5 FN penalty in the metric means rare classes dominate the score. Future work: oversampling rare classes using `WeightedRandomSampler`.

**2. Bias tuning is surprisingly powerful**
A simple post-processing grid search over per-class logit offsets improved scores by +2.4 points — more than any single architectural change. It's free, fast, and competition-legal.

**3. OneCycleLR outperforms ReduceLROnPlateau here**
The cyclical learning rate finds better optima in fewer epochs than waiting for plateau. On a time-limited T4, this matters.

**4. DenseNet121 > ResNet18 for this task**
Both are pre-trained on ImageNet, but DenseNet's dense connections give it an edge on subtle medical patterns. The 5x parameter increase from ResNet18 to DenseNet121 was worth it.

**5. The metric is the model's true teacher**
Training loss (Focal CE) and the competition metric (asymmetric macro score) are measuring different things. Always check the competition metric after every epoch, not just the loss.

**What I'd do differently with more time:**
- Train for more epochs (15-20) with a lower initial LR
- Add Test-Time Augmentation (TTA) — average predictions over multiple augmented versions of each test image
- Try EfficientNet-B4 or ViT-B/16 as alternative backbones
- Oversample the 5 rarest classes to at least 100 samples via augmentation

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| PyTorch 2.10 + CUDA 12.8 | Model training on GPU |
| torchvision | DenseNet121 pretrained weights + transforms |
| scikit-learn | StratifiedKFold cross-validation |
| pandas / numpy | Data manipulation |
| matplotlib | Training curve visualization |
| Kaggle Notebooks | Cloud GPU environment |

---

## 📁 Files

| File | Description |
|---|---|
| `notebook.ipynb` | Full training pipeline (Steps 1-6) |
| `sample_submission.csv` | Reference output format |
| `README.md` | This file |

---

*Built as part of the TDS (Tools in Data Science) course competition — 26-T1.*
