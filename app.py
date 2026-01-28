import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="ML Assignment 2",
    layout="wide"
)

st.title("Machine Learning Classification – Bank Deposit Prediction")
st.write("Upload a test dataset and evaluate different ML models.")
#Load Saved Models and Scaler
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": pickle.load(open("model/logistic_regression.pkl", "rb")),
        "Decision Tree": pickle.load(open("model/decision_tree.pkl", "rb")),
        "KNN": pickle.load(open("model/knn.pkl", "rb")),
        "Naive Bayes": pickle.load(open("model/naive_bayes.pkl", "rb")),
        "Random Forest": pickle.load(open("model/random_forest.pkl", "rb")),
        "XGBoost": pickle.load(open("model/xgboost.pkl", "rb"))
    }
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    return models, scaler

models, scaler = load_models()
#Dataset Upload
uploaded_file = st.file_uploader(
    "Upload CSV file (test data only)",
    type=["csv"]
)

model_name = st.selectbox(
    "Select a Machine Learning Model",
    list(models.keys())
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Encode target
    df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop('deposit', axis=1)
    y = df_encoded['deposit']

    model = models[model_name]

    # Prediction logic
    if model_name in ["Logistic Regression", "KNN"]:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

    # ---------------- Metrics ----------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
    col1.metric("AUC", round(roc_auc_score(y, y_prob), 3))

    col2.metric("Precision", round(precision_score(y, y_pred), 3))
    col2.metric("Recall", round(recall_score(y, y_pred), 3))

    col3.metric("F1 Score", round(f1_score(y, y_pred), 3))
    col3.metric("MCC", round(matthews_corrcoef(y, y_pred), 3))

    # ---------------- Confusion Matrix ----------------
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------- Classification Report ----------------
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))