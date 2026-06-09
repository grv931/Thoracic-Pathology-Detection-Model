# Automated Detection of Thoracic Pathologies from Chest X-ray Images

A deep learning pipeline to automate the detection of thoracic diseases from chest X-ray images. The task is a **20-class mutually exclusive classification problem** — each image is assigned to exactly one pathology. The system is built around a fine-tuned DenseNet121 backbone trained with a custom asymmetric loss function designed to reflect the clinical cost of missed diagnoses.

---

## 🚀 Tech Stack

| Category | Details |
| :--- | :--- |
| **Framework** | TensorFlow / Keras |
| **Domain** | Computer Vision & Medical Image Analysis |
| **Language** | Python |
| **Core Libraries** | NumPy, Pandas, Pillow (PIL), Matplotlib, Scikit-learn |
| **Backbone** | DenseNet121 (ImageNet pretrained, fine-tuned) |
| **Platform** | Kaggle (GPU: Tesla T4 ×2) |

---

## 📊 Dataset & Pathology Classes

The dataset consists of high-resolution chest X-ray images, each labeled with a single mutually exclusive pathology class:

- **Training Set**: 51,043 labeled chest X-ray images
- **Test Set**: 17,015 unlabeled images for Kaggle evaluation
- **Image Resolution**: Resized to `224 × 224` (RGB) for model input

### 20 Pathology Classes

| # | Class | # | Class |
|---|-------|---|-------|
| 1 | Atelectasis | 11 | Nodule |
| 2 | Cardiomegaly | 12 | Pleural Thickening |
| 3 | Consolidation | 13 | Pneumonia |
| 4 | Edema | 14 | Pneumothorax |
| 5 | Effusion | 15 | Pneumoperitoneum |
| 6 | Emphysema | 16 | Pneumomediastinum |
| 7 | Fibrosis | 17 | Subcutaneous Emphysema |
| 8 | Hernia | 18 | Tortuous Aorta |
| 9 | Infiltration | 19 | Calcification of the Aorta |
| 10 | Mass | 20 | No Finding (normal) |

### Class Distribution (Training Set)

The dataset is severely imbalanced. `No Finding` dominates with ~66% of all samples, while rare classes like `Pneumomediastinum` appear only 5 times:

```
No Finding              34,079   (66.8%)
Infiltration             5,206
Atelectasis              2,351
Effusion                 2,156
Nodule                   1,527
...
Pneumomediastinum            5   (0.01%)
```

---

## ⚡ Key Engineering Challenges & Strategies

### 1. Severe Class Imbalance
Standard models bias toward the `No Finding` majority class, achieving high accuracy while missing nearly all rare diseases.

**Strategies implemented:**
- **Per-class positive weights** computed as `(total_negatives / total_positives)` per class — ranging from `0.5×` for `No Finding` to `~9,898×` for `Pneumomediastinum`
- **Asymmetric Focal Loss** (see below) to aggressively down-weight easy majority-class negatives
- **GlobalAveragePooling2D** instead of Flatten to reduce parameter count and overfitting

### 2. Asymmetric Misclassification Costs
In clinical settings, a missed diagnosis (False Negative) is far more dangerous than a false alarm (False Positive). The competition metric directly encodes this:

| Outcome | Meaning | Score |
| :--- | :--- | :---: |
| True Positive (TP) | Correctly detected disease | `+1` |
| False Positive (FP) | Alarm raised with no disease | `-1` |
| False Negative (FN) | Disease completely missed | `-5` |
| Wrong disease predicted | FN for true class + FP for predicted class | `-6` |

**Loss function designed to match this:** A custom **Asymmetric Loss** is used with:
- `gamma_pos = 1.0` — mild focusing on hard positives (rare diseases)
- `gamma_neg = 4.0` — aggressive suppression of easy negatives (`No Finding`)

This forces the model to prioritise detecting rare diseases over confidently classifying normal scans.

---

## 🏗️ Model Architecture

The final model is built on **DenseNet121** fine-tuned from the `conv5_block15_2_conv` layer onwards:

```
DenseNet121 (ImageNet weights, partial fine-tune)
    └── GlobalAveragePooling2D
    └── Dense(512, relu)
    └── Dropout(0.2)
    └── Dense(128, relu)
    └── Dropout(0.2)
    └── Dense(20, sigmoid)          ← 20-class output
```

- **Optimizer**: AdamW (`lr = 1e-3`, `weight_decay = 1e-4`)
- **Loss**: Custom Asymmetric Loss (with per-class positive weights)
- **Metric**: AUC (Area Under ROC Curve, `multi_label=True`)
- **Epochs**: 12
- **Batch size**: 32

> **Explored but not used**: A VGG16 model (fine-tuned from `block5_conv1`) was defined and downloaded but never trained. A custom shallow CNN (Conv2D → Dense → Softmax) was also sketched out and discarded.

---

## 📈 Evaluation Metric

The competition uses a **macro-averaged asymmetric cost function** — each of the 20 classes contributes equally to the final score regardless of its frequency:

$$Score_c = \frac{TP_c - FP_c - 5 \cdot FN_c}{N_c}$$

$$\text{Final Score} = \frac{1}{20} \sum_{c=1}^{20} Score_c$$

Macro averaging is critical — it prevents gaming the metric by predicting only `No Finding`. Models must perform well on rare classes too.

**Training-set score: `-4.72`** — a strong result under the macro-averaged asymmetric cost metric, where scores in the range of `-4.x` indicate the model is performing well above the worst-case baseline of `-5.0`.

---

## 📁 Dataset File Structure

```
dataset/
├── images/
│   ├── img_0001.png
│   ├── img_0002.png
│   └── ...
├── train.csv               ← image_id + 20 one-hot class columns
├── test.csv                ← image_id only
└── sample_submission.csv   ← submission format reference
```

### Submission Format (`submission.csv`)

Each row must be **one-hot encoded** — exactly one `1`, all others `0`:

```csv
id,Atelectasis,Cardiomegaly,...,Calcification of the Aorta,No Finding
img_001.png,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
img_002.png,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
```

---

## ⚙️ Local Setup

### Prerequisites

```bash
pip install tensorflow numpy pandas pillow matplotlib scikit-learn
```

> GPU support requires a CUDA-compatible environment. The notebook was trained on Kaggle with dual Tesla T4 GPUs.

### Running the Notebook

The notebook is designed to run on Kaggle with the competition dataset mounted at:
```
/kaggle/input/competitions/26-t-1-dl-gen-ainppe-1/
```

Kaggle notebook: `22f3003274-notebook-26t1`
