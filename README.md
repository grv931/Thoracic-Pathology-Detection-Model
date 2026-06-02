# Automated Detection of Thoracic Pathologies from Chest X-ray Images

This repository contains a deep learning framework developed to automate the detection of thoracic diseases from chest X-ray imaging. Manual interpretation of these medical images by radiologists can be highly time-consuming and susceptible to human variability[cite: 2]. This project builds an automated diagnostic system that accurately identifies patterns in medical imaging and classifies each image into one of 20 predefined pathology classes[cite: 2].

---

## 🚀 Tech Stack

- **Framework**: PyTorch
- **Domain**: Computer Vision & Medical Image Analysis
- **Languages**: Python, Bash
- **Libraries**: OpenCV, NumPy, Pandas, Scikit-learn

---

## 📊 Dataset & Pathology Classes

The dataset consists of thousands of high-resolution medical images categorized into a single mutually exclusive pathology class per image[cite: 2]:

- **Total Training Dataset**: 51,043 labeled chest X-ray images[cite: 2].
- **Total Test Dataset**: 17,015 unlabeled images utilized for evaluation[cite: 2].
- **Pathology Classes (20 total)**: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax, Pneumoperitoneum, Pneumomediastinum, Subcutaneous Emphysema, Tortuous Aorta, Calcification of the Aorta, and No Finding (normal baseline)[cite: 2].

---

## ⚡ Key Engineering Challenges & Strategies

### 1. Severe Class Imbalance
Medical data naturally exhibits high class imbalance, as common conditions and normal findings appear far more frequently than rare abnormalities[cite: 2]. Standard networks tend to develop a severe bias toward predicting the majority class ("No Finding")[cite: 2]. 
*   **Mitigation Strategy**: The model incorporates specialized handling techniques, including focal loss architectures, class weighting adjustments, data augmentation, and balanced batch sampling to prevent majority-class bias[cite: 2].

### 2. Asymmetric Misclassification Costs
In healthcare applications, diagnostic errors are not equal[cite: 2]. Missing an active disease (a False Negative) yields catastrophic clinical outcomes compared to raising an unnecessary alarm (a False Positive)[cite: 2]. 
To align the model mathematically with clinical priorities, the scoring mechanism operates on a heavily penalized cost matrix[cite: 2]:
*   **True Positive (TP)**: Correctly detecting a pathology maps to a reward of `+1`[cite: 2].
*   **False Positive (FP)**: Erroneously declaring a pathology maps to a penalty of `-1`[cite: 2].
*   **False Negative (FN)**: Completely missing an active pathology maps to a severe penalty of `-5`[cite: 2].
*   **Misclassification Penalty**: Misclassifying Disease A as Disease B incurs a simultaneous False Negative for A and False Positive for B, resulting in an absolute penalty of `-6` points[cite: 2].

---

## 📈 Evaluation Metric

The network optimization functions target a macro-averaged asymmetric cost function designed to ensure each of the 20 classes impacts the score equally[cite: 2]. This guarantees that the architecture maintains high diagnostic sensitivity for rare diseases instead of artificially inflating accuracy on normal images[cite: 2].

---

## ⚙️ Local Setup & Configuration

### Prerequisites
- Python 3.x
- PyTorch (CUDA supported)
- Common data science dependencies (`pip install numpy pandas opencv-python`)

