import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# Load and preprocess dataset
# -----------------------------
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
col_names = ["Patient_ID", "Target"] + [f"attr_{i}" for i in range(1, 31)]
dataset = pd.read_csv(data_url, header=None, names=col_names)

# Convert target to numeric (M=1, B=0)
dataset['Target'] = dataset['Target'].map({'M': 1, 'B': 0})
dataset = dataset.drop("Patient_ID", axis=1)

features = dataset.drop("Target", axis=1)
labels = dataset["Target"]

# Scale features for better convergence
scaler_obj = StandardScaler()
features_scaled = scaler_obj.fit_transform(features)

X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
    features_scaled, labels, test_size=0.2, random_state=42
)

# -----------------------------
# Define models
# -----------------------------
classifier_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs"),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Breast Cancer Classifier Dashboard")

# Ensure model directory exists
os.makedirs("saved_models", exist_ok=True)

# File uploader for test data
uploaded_csv = st.file_uploader("Upload Test CSV (optional)", type=["csv"])

if uploaded_csv is not None:
    test_data = pd.read_csv(uploaded_csv)

    # Ensure column names match training schema
    expected_columns = [f"attr_{i}" for i in range(1, 31)] + ["Target"]
    test_data.columns = test_data.columns.str.strip()

    if not set(expected_columns).issubset(set(test_data.columns)):
        st.error("Uploaded file columns do not match training schema.")
        st.stop()

    # Handle Target column safely
    if test_data['Target'].dtype == object:
        test_data['Target'] = (
            test_data['Target']
            .astype(str).str.strip().str.upper()
            .map({'M': 1, 'B': 0})
        )
    else:
        test_data['Target'] = test_data['Target'].astype(int)

    # Drop rows with missing target values
    test_data = test_data.dropna(subset=['Target'])

    # Apply the same scaler used during training
    test_features = scaler_obj.transform(test_data.drop("Target", axis=1))
    test_labels = test_data["Target"].astype(int)
else:
    test_features, test_labels = X_test_set, y_test_set

# -----------------------------
# Provide full RAW test CSV download
# -----------------------------
# Inverse transform to get raw values back
raw_test_df = pd.DataFrame(
    scaler_obj.inverse_transform(X_test_set),
    columns=[f"attr_{i}" for i in range(1, 31)]
)
raw_test_df["Target"] = y_test_set.values

csv_output = raw_test_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Full Raw Test CSV",
    data=csv_output,
    file_name="raw_test_data.csv",
    mime="text/csv"
)

# -----------------------------
# Model selection and execution
# -----------------------------
chosen_model = st.selectbox("Choose Model", list(classifier_models.keys()))

if st.button("Run Model"):
    selected_model = classifier_models[chosen_model]
    selected_model.fit(X_train_set, y_train_set)
    predictions = selected_model.predict(test_features)

    # Save trained model and scaler
    joblib.dump(selected_model, f"saved_models/{chosen_model.replace(' ', '_')}.pkl")
    joblib.dump(scaler_obj, "saved_models/scaler.pkl")

    # Display metrics
    st.write("Accuracy:", accuracy_score(test_labels, predictions))
    st.write("Precision:", precision_score(test_labels, predictions))
    st.write("Recall:", recall_score(test_labels, predictions))
    st.write("F1 Score:", f1_score(test_labels, predictions))
    st.write("MCC:", matthews_corrcoef(test_labels, predictions))

    # AUC Score (binary classification)
    if hasattr(selected_model, "predict_proba"):
        prob_scores = selected_model.predict_proba(test_features)[:, 1]
        st.write("AUC:", roc_auc_score(test_labels, prob_scores))

    # Confusion Matrix
    conf_matrix = confusion_matrix(test_labels, predictions)
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
