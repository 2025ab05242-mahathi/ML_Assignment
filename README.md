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
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| KNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |

*(Fill the values from your notebook results table)*

---

## Model Performance Observations

| Model | Observation |
|-----|------------|
| Logistic Regression | Provides a strong baseline performance with good interpretability but may not capture complex relationships. |
| Decision Tree | Capable of capturing non-linear patterns but may overfit the training data. |
| KNN | Performance depends on distance metrics and feature scaling and can be sensitive to noise. |
| Naive Bayes | Fast and efficient but relies on strong independence assumptions between features. |
| Random Forest | Demonstrates improved performance by reducing overfitting through ensemble learning. |
| XGBoost | Achieves strong overall performance due to gradient boosting and effective handling of complex feature interactions. |

---

## Streamlit Application
The Streamlit web application allows users to:
- Upload a CSV test dataset
- Select a machine learning model
- View evaluation metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- Visualize confusion matrix and classification report

---

## Deployment
The application is deployed using Streamlit Community Cloud and can be accessed
via the following link:

**Live App Link:** *(Add your Streamlit app URL here)*

---

## GitHub Repository
**Repository Link:** *(Add your GitHub repo link here)*

---

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py