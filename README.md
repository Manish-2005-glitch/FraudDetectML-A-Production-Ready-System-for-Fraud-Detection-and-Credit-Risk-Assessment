# FraudDetectML  
## A Production-Ready System for Fraud Detection & Credit Risk Assessment 🚀

**FraudDetectML** is an end-to-end, production-oriented Machine Learning system designed to detect fraudulent transactions and assess credit risk using real-world financial data.  
This project goes beyond notebooks and focuses on **deployable ML pipelines**, **model explainability**, and **API-based inference**, aligning with industry-level ML engineering standards.

---

## 🔍 Problem Statement

Financial institutions face significant losses due to fraudulent transactions and high-risk credit approvals.  
Traditional rule-based systems fail to adapt to evolving fraud patterns and complex customer behaviors.

**Objective:**  
Build a scalable, explainable, and production-ready ML system that:
- Detects fraudulent activity
- Assesses credit risk accurately
- Supports real-time inference via APIs
- Monitors performance and model drift

---

## 🧠 Solution Overview

This system uses supervised machine learning models trained on transactional data to:
- Predict the probability of fraud or credit default
- Optimize decision thresholds using Precision-Recall tradeoffs
- Explain predictions using SHAP values
- Detect data and prediction drift over time
- Serve predictions through a REST API

---

## ⭐ Key Features

- ✅ End-to-end ML pipeline (training → evaluation → deployment)
- ✅ Fraud detection & credit risk classification
- ✅ Threshold optimization using Precision-Recall curves
- ✅ Explainable AI using SHAP
- ✅ Model drift detection utilities
- ✅ REST API for real-time predictions
- ✅ Modular, production-ready project structure

---

## 🗂️ Project Structure

fraud-detection-ml-system/
│
├── backend/
│   ├── main.py
│   ├── inference.py
│   ├── model_loader.py
│   ├── schemas.py
│   ├── model/
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── threshold.txt
│
├── app.py                 # Streamlit
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

bash
git clone https://github.com/Manish-2005-glitch/FraudDetectML-A-Production-Ready-System-for-Fraud-Detection-and-Credit-Risk-Assessment.git
cd FraudDetectML-A-Production-Ready-System-for-Fraud-Detection-and-Credit-Risk-Assessment
Install Dependencies
pip install -r requirements.txt

##🚦 Model Training
Train the fraud detection / credit risk model:

bash
python training.py
📊 Model Evaluation

Generate Precision-Recall curves and performance metrics:
python precision_recall.py
This helps in selecting the optimal threshold for imbalanced fraud data.

🔍 Explainability (SHAP)

Understand why the model makes certain predictions:
python shap_analysis.py

📈 Drift Detection

Monitor whether incoming data differs from training data:
python drift.py

🚀 Run the API Server

Start the backend API for real-time inference:
python backend/app.py
The API exposes endpoints to:

Accept transaction / credit input
Return fraud probability & classification
Apply trained thresholds automatically

📡 Example API Request
POST /predict
Content-Type: application/json

{
  "transaction_amount": 1200,
  "account_age_days": 365,
  "num_previous_transactions": 58,
  "avg_transaction_value": 430
}

Response:

{
  "fraud_probability": 0.87,
  "is_fraud": true
}

🧩 ML Engineering Highlights

- ✅Imbalanced classification handling
- ✅Threshold tuning for business-critical metrics
- ✅Explainable AI for regulatory transparency
- ✅Modular design for scalability
- ✅Production-ready inference via API
- ✅Drift monitoring for long-term reliability

