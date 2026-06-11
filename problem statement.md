# Automated Detection of Thoracic Pathologies from Chest X-ray Images

## 1. Introduction
Chest X-ray imaging is one of the most widely used diagnostic tools in medical practice for detecting diseases related to the thoracic region such as lung infections, fluid accumulation, or structural abnormalities of the heart and lungs. However, manual analysis of X-ray images by radiologists can be time-consuming and prone to variability due to human factors.

In this assignment, the objective is to design and implement an automated deep learning system capable of detecting thoracic diseases from chest X-ray images. The task is formulated as a multi-class classification problem, where each X-ray image must be assigned to one of 20 possible pathology classes.

The system should learn patterns in medical imaging data and predict the most likely pathology class associated with the image.

This problem is particularly challenging because it involves medical data, class imbalance, and asymmetric misclassification costs, meaning that different types of prediction errors have different levels of severity.

---

## 2. Problem Objective
The goal of this assignment is to build a machine learning or deep learning model that can:
1. **Analyze** chest X-ray images.
2. **Identify** the most probable thoracic pathology.
3. **Assign** the image to one of the 20 predefined classes.

This task is formulated as a multi-class classification problem, where each input image belongs to exactly one pathology class.

### Predefined Pathology Classes
The possible classes include various thoracic conditions:
* `Atelectasis`
* `Cardiomegaly`
* `Consolidation`
* `Edema`
* `Effusion`
* `Emphysema`
* `Fibrosis`
* `Hernia`
* `Infiltration`
* `Mass`
* `Nodule`
* `Pleural_Thickening`
* `Pneumonia`
* `Pneumothorax`
* `Pneumoperitoneum`
* `Pneumomediastinum`
* `Subcutaneous Emphysema`
* `Tortuous Aorta`
* `Calcification of the Aorta`
* `No Finding` (normal baseline)

> [!NOTE]
> The `No Finding` class represents cases where no abnormality is present in the X-ray.

---

## 3. Key Challenges in the Dataset

### 3.1 Class Imbalance
Medical datasets often suffer from class imbalance, meaning that some diseases appear very frequently while others appear only rarely.

| Severity / Type | Frequency |
| :--- | :--- |
| **Common conditions** | Many examples |
| **Rare diseases** | Very few examples |

This creates a challenge because machine learning models tend to become biased toward majority classes, especially the `No Finding` class. If the model simply predicts the majority class most of the time, it may achieve high accuracy but perform poorly in detecting rare diseases.

#### Mitigation Strategies:
To address this bias, you can employ special techniques:
* **Class Weighting**: Adjust loss penalties per class.
* **Oversampling**: Balance dataset counts for rare classes.
* **Loss Functions**: Use specialized loss functions like **Focal Loss**.
* **Sampling**: Implement balanced batch sampling.

### 3.2 Asymmetric Misclassification Costs
Unlike many typical machine learning tasks, not all prediction errors are equally harmful in medical diagnosis.
* **False Negatives (FN)**: Missing a disease can have catastrophic consequences for the patient.
* **False Positives (FP)**: Raising a false alarm is less harmful but still undesirable.

Therefore, the scoring system heavily penalizes False Negatives.

### 3.3 Cost Interpretation

| Prediction Outcome | Meaning | Score |
| :--- | :--- | :---: |
| **True Positive (TP)** | Correctly detecting a disease | `+1` |
| **False Positive (FP)** | Predicting disease when it is not present | `-1` |
| **False Negative (FN)** | Missing the disease | `-5` |

This scoring mechanism strongly encourages models to avoid missing diseases (minimizing False Negatives).

---

## 4. Special Cases in Prediction
Because `No Finding` is also treated as a class, certain prediction cases require special interpretation.

* **Case 1: Correctly Predicting No Finding**
  * *Scenario*: Model predicts `No Finding` and the patient truly has no disease.
  * *Score*: `+1`
* **Case 2: Predicting No Finding when Disease Exists**
  * *Scenario*: Model predicts `No Finding` but the patient actually has a disease (False Negative).
  * *Score*: `-5` (Highly penalized because the disease was completely missed).
* **Case 3: Predicting Disease when it Should be No Finding**
  * *Scenario*: Model predicts a disease but the patient has `No Finding` (False Positive).
  * *Score*: `-1` (Unnecessary medical alarm).
* **Case 4: Confusing One Disease with Another**
  * *Scenario*: True disease is `Disease A` but the model predicts `Disease B`.
  * *Penalty*: Counts as both a False Negative for `A` (`-5`) and a False Positive for `B` (`-1`).
  * *Total Score*: `-6` (Serious diagnostic mistake).

---

## 5. Evaluation Metric
The competition uses a **macro-averaged asymmetric cost function**. Because of the penalties, scores will often be negative (e.g., `-4.xxxx` or `-5.xxxx`).

The score for each class $c$ is computed as:

$$Score_c = \frac{TP_c - FP_c - 5 \cdot FN_c}{N_c}$$

Where:
* $TP_c$: True positives for class $c$
* $FP_c$: False positives for class $c$
* $FN_c$: False negatives for class $c$
* $N_c$: Number of true samples for class $c$ ($N_c = TP_c + FN_c$)

### 5.1 Final Competition Score
The final score is obtained by averaging the scores across all 20 classes:

$$Final\ Score = \frac{1}{C} \sum_{c=1}^{C} Score_c$$

Where $C = 20$.

> [!IMPORTANT]
> **Why Macro Averaging is Important**
> Macro averaging ensures that each class contributes equally to the final score regardless of its frequency. This prevents models from maximizing performance simply by predicting the majority class (`No Finding`). Instead, models must perform well on rare diseases, which is critical in medical diagnosis.

---

## 6. Dataset Description
The dataset contains chest X-ray images labeled with thoracic pathologies across 20 classes.

### Dataset Splits

#### Training Dataset
| Property | Value |
| :--- | :--- |
| **Number of images** | 51,043 |
| **Labels available** | Yes |
| **Purpose** | Model training and validation |

*Each training image has a corresponding label indicating the presence of exactly one pathology.*

#### Test Dataset
| Property | Value |
| :--- | :--- |
| **Number of images** | 17,015 |
| **Labels available** | No |
| **Purpose** | Kaggle evaluation |

### Label Format
Labels are provided in a one-hot encoded format. Each image belongs to **exactly one class** (mutually exclusive).

**Example CSV Format:**
```csv
id,Atelectasis,Cardiomegaly,............,Calcification of the Aorta,No Finding
09285150fc1d4e1db86891eb1a561841.png,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
09285150fc1d4e1db86891eb1a561841.png,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
09285150fc1d4e1db86891eb1a561841.png,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
```

### File Structure
The dataset is organized as follows:
```text
dataset/
├── images/
│   ├── img_0001.png
│   ├── img_0002.png
│   └── ...
├── train.csv
└── sample_submission.csv
```

* **`train.csv`** contains the `image id` and 20 binary columns corresponding to the pathology classes.

---


## 9. Submission Format
Students must submit a CSV file named exactly `submission.csv`. The format must match `sample_submission.csv` provided in Kaggle.

> [!IMPORTANT]
> **Key constraints:**
> * Each row must contain exactly one `1` (one-hot encoded).
> * All other class columns must contain `0`.

---

## 🛠️ Model Development Strategies
To build a high-performing model, consider the following strategies:

### 1. Handling Class Imbalance
* Use **Weighted Loss Functions** to increase the cost of misclassifying minority classes.
* Perform **Oversampling** on rare classes.
* Apply **Data Augmentation** techniques.
* Optimize using **Focal Loss** to focus on hard-to-classify samples.

### 2. Optimizing for the Cost Metric
Since False Negatives are heavily penalized ($-5$), the model training and prediction thresholds should focus on:
* **Maximizing True Positives** (sensitivity/recall for pathologies).
* **Minimizing False Negatives** (avoiding missing active conditions).
