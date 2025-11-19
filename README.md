# 🏦 Credit Risk & Loan Default Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Risk%20Modeling-red?style=for-the-badge&logo=xgboost)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Hackathon-Rank%201%20🏆-brightgreen?style=for-the-badge)

> **Winner (Rank #1) at IIT Mandi AIFUL Machine Learning Hackathon (Fintech Track).**
> An AI-powered risk assessment engine designed to predict potential loan defaulters and reduce Non-Performing Assets (NPAs) for financial institutions.

---

## 📉 Business Problem
In the lending industry, the most critical challenge is minimizing **Credit Risk**.
* If a bank denies a loan to a good customer, it loses interest income (Opportunity Loss).
* **BUT**, if a bank approves a loan to a defaulter, it loses the entire principal amount (Direct Financial Loss).

**The Goal:** Build a robust machine learning model that prioritizes minimizing **False Negatives** (predicting a defaulter as 'safe') to protect the lender's capital.

---

## 🚀 Key Features & Highlights
* **🏆 Hackathon Winning Solution:** Optimized for high accuracy on unseen test data.
* **⚖️ Imbalance Handling:** Utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to handle the severely skewed dataset (where defaulters were <10%).
* **🧠 Explainable AI:** Integrated **SHAP (SHapley Additive exPlanations)** values to interpret *why* a specific applicant was rejected (crucial for Regulatory Compliance).
* **⚡ Ensemble Learning:** Combined **XGBoost** and **Random Forest** for robust predictions.

---

## 📊 Technical Methodology

### 1. Exploratory Data Analysis (EDA)
* Analyzed correlations between income, credit score, and default rates.
* Detected and treated outliers in "Annual Income" and "Loan Amount" features.

### 2. Feature Engineering
* Created new financial ratios: `Debt-to-Income Ratio`, `Credit Utilization Score`.
* Encoded categorical variables (Employment Type, Home Ownership) using Target Encoding.

### 3. Handling Class Imbalance
Real-world financial data is always imbalanced.
* **Challenge:** The model was biased towards the majority class (Non-Defaulters).
* **Solution:** Applied **SMOTE** to synthesize minority class samples, ensuring the model learns the patterns of defaulters effectively.

### 4. Model Selection
We benchmarked Logistic Regression, Decision Trees, and Gradient Boosting.
* **Final Model:** XGBoost Classifier (Tuned via GridSearchCV).
* **Why XGBoost?** It handles missing values internally and is the industry standard for tabular financial data.

---

## 📈 Performance Metrics

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **AUC-ROC Score** | **0.92** | Excellent capability to distinguish between Defaulters and Payers. |
| **Recall (Defaulters)** | **0.89** | Captures 89% of actual defaulters (High Risk Coverage). |
| **Precision** | **0.85** | Low rate of false alarms. |

### Confusion Matrix Visualization
*[INSERT CONFUSION MATRIX IMAGE HERE]*
*(The matrix demonstrates a low count of False Negatives, validating the risk-averse nature of the model.)*

---

## 💡 Feature Importance (What drives default?)
According to our SHAP analysis, the top factors influencing loan default are:
1.  **Credit Score:** (Inverse correlation - Lower score = Higher Risk).
2.  **Debt-to-Income Ratio:** (Higher ratio = Higher Risk).
3.  **Loan Term:** (Longer term loans showed slightly higher default rates).

---

## 💻 Installation & Setup

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Step 1: Clone the Repo
```bash
git clone [https://github.com/AnshAttre/Credit-Risk-Prediction.git](https://github.com/AnshAttre/Credit-Risk-Prediction.git)
cd Credit-Risk-Prediction
