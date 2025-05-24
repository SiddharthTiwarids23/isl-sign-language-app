# 🤟 Indian Sign Language Recognition (ISL) using Deep Learning

This project builds a deep learning-based image classification model to recognize Indian Sign Language (ISL) gestures (A–Z, 1–9). It uses **ResNet50** with transfer learning and is deployed as a real-time prediction app using **Streamlit**.

### 🔗 Live Demo:
👉 [Try the App](https://isl-sign-language-app-5shtilmfxfb3dtrgjzkdz9.streamlit.app/)

---

## 📌 Project Highlights

- ✅ 35 ISL gestures (A-Z, 1–9)
- ✅ ResNet50 model with fine-tuning
- ✅ Achieved 100% accuracy on test set
- ✅ Data augmentation to improve generalization
- ✅ Evaluated using a confusion matrix and random predictions
- ✅ Streamlit web app deployment

---

## 🧠 Model Architecture

- **Base Model**: ResNet50 (ImageNet pretrained)
- **Classifier Head**:
  - Global Average Pooling
  - Dropout (0.5)
  - Dense Softmax (35 output nodes)
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Learning Rate Schedule**: 1e-3 → 1e-5 during fine-tuning

---

## 📂 Dataset Overview

- Balanced dataset with ≈1200 images/class
- Folder-structured (`/A`, `/B`, ..., `/9`)
- Preprocessed to 224x224 and normalized
- Augmentation: random flips, zoom, rotation

---

## 📈 Performance

| Metric        | Value       |
|---------------|-------------|
| Test Accuracy | 100%        |
| Classes       | 35          |
| Model Type    | ResNet50    |

Confusion matrix and sample predictions confirm flawless classification under controlled conditions.

---

## 🚀 How to Run

### ▶️ On Streamlit Cloud (No setup required)
Visit the [live app](https://isl-sign-language-app-5shtilmfxfb3dtrgjzkdz9.streamlit.app/)

### 🖥️ Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/isl-sign-language-app.git
   cd isl-sign-language-app
