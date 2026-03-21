# 🔐 Hybrid Fusion Network Intrusion Detection System (UNSW-NB15)

## 👤 Author
**Godbless Keku**

---

## 📌 Project Overview

Cybersecurity threats are becoming increasingly complex, making traditional intrusion detection systems insufficient for detecting modern and evolving attacks.

This project presents a **Hybrid Fusion Framework** that combines:

- **Supervised Learning (Random Forest)** → detects known attack patterns  
- **Unsupervised Learning (Dense Autoencoder)** → detects unknown anomalies  

The system integrates both approaches to improve detection performance and robustness.

---

## ⚙️ Pipeline
Data → Preprocessing → Model → Training → Evaluation → Deployment

---

## 📊 Dataset

- **Dataset:** UNSW-NB15  
- **Type:** Network intrusion detection dataset  
- **Features:** ~49 network traffic features  
- **Target:**
  - `0` → Normal Traffic  
  - `1` → Attack Traffic  

---

## 🧹 Data Preprocessing

- Removed duplicate records  
- Dropped non-informative features:
  - `id`
  - `attack_cat` (to avoid leakage)
- Handled missing values using imputation  
- Applied:
  - **Standard scaling** (numerical features)  
  - **Encoding** (categorical features)  
- Used a consistent preprocessing pipeline across:
  - training  
  - validation  
  - test  

---

## 🤖 Models Implemented

### Supervised Models
- Logistic Regression  
- Random Forest  
- CNN (1D)

### Unsupervised Model
- Dense Autoencoder (Anomaly Detection)

---

## 🔥 Proposed Hybrid Fusion Framework

The final prediction is computed using:

Final Score = α × Supervised Probability + (1 − α) × Anomaly Score

Where:

- **Supervised Probability** → from Random Forest  
- **Anomaly Score** → from Autoencoder  
- **α** → fusion weight (tuned using validation data)  

---

### 🎯 Purpose of Hybrid Model

- Detect **known attacks** (supervised learning)  
- Detect **unknown anomalies** (unsupervised learning)  
- Improve **recall and robustness**

---

## 📈 Results

| Model | Accuracy | Recall | F1 | ROC-AUC |
|------|--------|--------|-----|--------|
| Logistic Regression | ~0.87 | ~0.84 | ~0.90 | ~0.97 |
| Random Forest | 0.901 | 0.866 | 0.923 | **0.983** |
| CNN (1D) | 0.88 | 0.84 | 0.907 | 0.982 |
| Dense Autoencoder | 0.67 | 0.73 | 0.75 | 0.76 |
| **Hybrid Fusion** | **0.906** | **0.875** | **0.927** | 0.982 |

---

## 📊 Key Insights

- Random Forest achieved the highest ROC-AUC  
- Hybrid Fusion achieved the best:
  - Accuracy  
  - Recall  
  - F1-score  

👉 Hybrid model reduces **missed attacks**, which is critical in cybersecurity  

---

## 🧠 Why Hybrid Works

- Random Forest → strong for known patterns  
- Autoencoder → detects unusual behavior  
- Fusion → combines both signals  

---

## 📊 Evaluation

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

---

## 📷 Results Visualization

### ROC Curve
![ROC Curve](/images/roc_curve.png)

### Confusion Matrix
![Confusion Matrix](/images/confusion_matrix.png)

---


## 🚀 How to Run the Project

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model
python train.py

### 3. Evaluate the model
python evaluate.py

### 4. Run Inference
python inference.py --input data/UNSW_NB15_testing-set.csv --output predictions.csv

### 5. Run the Demo app
streamlit run demo.py

### 6. Open in browser
http://127.0.0.1:5000


## 📚 References

UNSW-NB15 Dataset

Scikit-learn Documentation

PyTorch Documentation

## 📷 Demo 
![Dashboard](/images/dashboard.png)
This project includes a **streamlit web application** that simulates the intrusion detection system:

### Features:
- Upload CSV file 
- Preview dataset
---
### Generates:
- upervised_score
- anomaly_score
- fusion_score
- prediction
- Model details
- you can download predictions result in CSV
