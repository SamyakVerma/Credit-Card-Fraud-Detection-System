# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project that detects fraudulent credit card transactions using 3 different models

Dataset: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📌 Problem Statement

Only 0.17% of transactions in the dataset are fraudulent — making this a
classic **extreme class imbalance** problem. A naive model predicting
"not fraud" every time achieves 99% accuracy but is completely useless.
This project tackles that head-on.

---

## 📊 Dataset

| Property       | Details                           |
|----------------|-------------------------------    |
| Source         | Kaggle / ULB Machine Learning     |
| Rows           | 284,807 transactions              |
| Fraud cases    | 492 (0.17%)                       |
| Features       | 28 PCA components + Time + Amount |
| Target         | Class (0 = Legit, 1 = Fraud)      |

---

## ⚙️ Pipeline

Raw Data → EDA → Sampling (30K) → Scaling → Train/Test Split
→ SMOTE → Model Training → Evaluation → Comparison

## 🤖 Models Trained

| Model               | Type                        |
|---------------------|-----------------------------|
| Logistic Regression | Baseline classifier         |
| LightGBM            | Gradient boosting           |
| Autoencoder         | Deep learning / anomaly detection |

---

## 📈 Graphs & Visualizations

- Class distribution (before & after SMOTE)
- Transaction Amount & Time distributions
- Confusion matrices (all 3 models)
- ROC curves (all 3 models overlaid)
- Precision-Recall curves (all 3 models overlaid)
- Autoencoder reconstruction error distribution
- Top 15 Feature Importances (LightGBM)
- Final model comparison table

---

## 📋 Evaluation Metrics

| Model               | Precision | Recall | F1   | ROC-AUC | PR-AUC |
|---------------------|-----------|--------|------|---------|--------|
| Logistic Regression | —         | —      | —    | —       | —      |
| LightGBM            | —         | —      | —    | —       | —      |
| Autoencoder         | —         | —      | —    | —       | —      |

> Fill in your actual scores after running the notebook.

---

## 🛠️ Tech Stack

- Python 3
- Pandas & NumPy
- Scikit-learn
- LightGBM
- TensorFlow / Keras
- Imbalanced-learn (SMOTE)
- Matplotlib & Seaborn
- Google Colab

---

## 🚀 How to Run

1. Clone the repo
```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

2. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   and place it in the project folder as `archive.zip`

3. Open `fraud_detection.ipynb` in Google Colab or Jupyter

4. Run all cells top to bottom

---

## 💡 Key Learnings

- **Accuracy is misleading** for imbalanced datasets — always check Recall & PR-AUC
- **SMOTE** helps balance classes, but must only be applied to training data
- **LightGBM** outperformed all other models in speed and metrics
- **Autoencoders** are a powerful unsupervised approach — no fraud labels needed at train time
- **PR-AUC** is more informative than ROC-AUC when classes are heavily imbalanced

---

## 📁 Project Structure

credit-card-fraud-detection/
├── fraud_detection.ipynb   # Main notebook (all 19 cells)
├── README.md               # This file
└── requirements.txt        # Dependencies

## 📦 Requirements

pandas
numpy
scikit-learn
lightgb
tensorflow
imbalanced-learn
matplotlib
seaborn

