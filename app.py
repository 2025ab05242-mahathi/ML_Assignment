import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------- Page Config ----------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Classification – Bank Deposit Prediction")

st.write("Upload the Bank Marketing dataset to train and evaluate multiple ML models.")

# ---------------- Upload Dataset ----------------
uploaded_file = st.file_uploader("Upload Bank Marketing CSV", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ---------------- Preprocessing ----------------
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

X = df.drop('deposit', axis=1)
y = df['deposit']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Models ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
}

model_name = st.selectbox("Select Machine Learning Model", list(models.keys()))
model = models[model_name]

# ---------------- Training & Prediction ----------------
if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

# ---------------- Metrics ----------------
st.subheader("Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
col1.metric("AUC", round(roc_auc_score(y_test, y_prob), 3))

col2.metric("Precision", round(precision_score(y_test, y_pred), 3))
col2.metric("Recall", round(recall_score(y_test, y_pred), 3))

col3.metric("F1 Score", round(f1_score(y_test, y_pred), 3))
col3.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 3))

# ---------------- Confusion Matrix ----------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ---------------- Classification Report ----------------
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))