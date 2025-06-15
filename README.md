# 🩻 AI-Powered Chest X-ray Diagnosis with ResNet-50 and Grad-CAM

This project builds an AI-based medical diagnosis tool that uses **ResNet-50** to classify chest X-ray images and applies **Grad-CAM** for visual interpretability. It is built using the NIH ChestX-ray14 dataset and focuses on detecting pneumonia (or binary classification tasks).

---

## 📌 Features

- ✅ Trained CNN model (ResNet-50) on labeled chest X-ray images
- ✅ Grad-CAM visualization for medical explainability
- ✅ Data preprocessing and augmentation
- ✅ Evaluation metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
- ✅ Modular and easy-to-extend codebase

---

## 🧠 Model Architecture

- Base model: `ResNet-50` (pretrained on ImageNet)
- Final layer modified for binary classification (`Normal` vs `Pneumonia`)
- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam`

---

## 📊 Sample Grad-CAM Output

> Below: Heatmap showing focus region of the model on a Pneumonia-labeled X-ray.

![Grad-CAM Example](https://github.com/abdulmannaan502/AI-Powered-Chest-X-ray-Diagnosis/blob/main/Images/2.png)

---

## 🧪 Results

| Metric      | Score (%) |
|-------------|-----------|
| Accuracy    | 85.5      |
| Precision   | 86.1      |
| Recall      | 85.2      |
| F1-Score    | 84.1      |


---

## 🚀 Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/chest-xray-diagnosis.git
cd chest-xray-diagnosis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train the model or run inference:**
```bash
python train.py           # To train
python predict.py         # To predict

```

4. **Run Grad-CAM:**
```bash
python gradcam/generate.py --image_path sample.jpg
```
---

## 📦 Dataset
We used the NIH ChestX-ray14 dataset for training and evaluation.
Due to its size and license restrictions, the dataset is not included in this repository.
















