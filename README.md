# ML Assignment 2 – Bank Deposit Prediction

## Problem Statement
The objective of this project is to build and compare multiple machine learning
classification models to predict whether a bank customer will subscribe to a
term deposit based on demographic, financial, and campaign-related attributes.
The trained models are deployed using a Streamlit web application that allows
interactive evaluation and comparison.

---

## Dataset Description
The Bank Marketing dataset is a real-world dataset related to direct marketing
campaigns conducted by a banking institution. It contains customer information
such as age, job, marital status, education, balance, loan details, contact
information, campaign history, and previous outcomes. The target variable
`deposit` indicates whether a customer subscribed to a term deposit (`yes` or
`no`). The dataset includes both numerical and categorical features and satisfies
the minimum instance and feature requirements specified in the assignment.

---

## Models Used and Evaluation Metrics

The following machine learning models were implemented and evaluated on the same
dataset:

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Naive Bayes Classifier  
- Random Forest (Ensemble Model)  
- XGBoost (Ensemble Model)  

### Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8258 | 0.9072 | 0.8270 | 0.7996 | 0.8131 | 0.6504 |
| Decision Tree | 0.7971 | 0.7962 | 0.7900 | 0.7788 | 0.7844 | 0.5929 |
| KNN | 0.7783 | 0.8416 | 0.7998 | 0.7098 | 0.7521 | 0.5561 |
| Naive Bayes | 0.7331 | 0.8227 | 0.7992 | 0.5832 | 0.6743 | 0.4738 |
| Random Forest | 0.8571 | 0.9187 | 0.8267 | 0.8837 | 0.8543 | 0.7160 |
| XGBoost | 0.8643 | 0.9257 | 0.8410 | 0.8800 | 0.8600 | 0.7292 |

---

## Model Performance Observations

| Model | Observation |
|-----|------------|
| Logistic Regression | Provides a strong baseline performance with good interpretability but may not capture complex non-linear relationships. |
| Decision Tree | Capable of modeling non-linear patterns but can overfit the training data. |
| KNN | Performance is sensitive to feature scaling and choice of neighbors. |
| Naive Bayes | Computationally efficient but assumes feature independence, which may limit performance. |
| Random Forest | Shows strong performance by reducing variance through ensemble learning. |
| XGBoost | Achieves the best overall performance due to gradient boosting and effective handling of complex feature interactions. |

---

## Streamlit Application
The Streamlit web application allows users to:
- Upload a CSV dataset
- Select a machine learning model
- View evaluation metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- Visualize confusion matrix and classification report

---

## Deployment
The application is deployed using Streamlit Community Cloud.

**Live App Link:** https://mlassignment-k6o5wbah8oexiqtvdgqkws.streamlit.app

---

## GitHub Repository
**Repository Link:**  
https://github.com/2025ab05242-mahathi/ML_Assignment

---

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py