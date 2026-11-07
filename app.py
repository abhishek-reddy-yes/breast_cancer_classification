"""Web front end for the breast cancer classifier.

Run from this folder:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data import load_data, split_data

MODEL_PATH = Path(__file__).parent / "models" / "best_model.joblib"
LABELS = {0: "Malignant", 1: "Benign"}
GROUPS = {"Mean values": "mean ", "Standard error": " error", "Worst values": "worst "}

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        st.error("No saved model found. Run `python train.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_data():
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    return X, X_test, y_test


model = get_model()
X, X_test, y_test = get_data()

st.title("🩺 Breast Cancer Classifier")
st.write(
    "Enter the 30 cell-nucleus measurements and the model predicts whether the "
    "tumor is **malignant** or **benign**. Values start at the dataset average."
)
st.caption("For learning purposes only, not a medical diagnosis.")

IMPORTANCE_PLOT = Path(__file__).parent / "reports" / "feature_importance.png"
if IMPORTANCE_PLOT.exists():
    with st.expander("Which measurements matter most to the model?"):
        st.image(str(IMPORTANCE_PLOT))

# Sidebar: fill the form from a real test-set sample
st.sidebar.header("Try a real sample")
sample_idx = st.sidebar.selectbox(
    "Test-set sample",
    options=[None, *range(len(X_test))],
    format_func=lambda i: "Dataset average" if i is None
    else f"Sample {i} (actual: {LABELS[y_test.iloc[i]]})",
)
start = X.mean() if sample_idx is None else X_test.iloc[sample_idx]

# Input form, one tab per feature group
values = {}
with st.form("features"):
    for tab, (group, key) in zip(st.tabs(list(GROUPS)), GROUPS.items()):
        cols_in_group = [c for c in X.columns if key in c]
        with tab:
            grid = st.columns(3)
            for i, col in enumerate(cols_in_group):
                values[col] = grid[i % 3].number_input(
                    col.capitalize(),
                    min_value=0.0,
                    value=float(start[col]),
                    format="%.4f",
                    key=f"{col}-{sample_idx}",
                )
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    row = pd.DataFrame([values])[X.columns]
    pred = int(model.predict(row)[0])
    p_benign = float(model.predict_proba(row)[0, 1])
    confidence = p_benign if pred == 1 else 1 - p_benign

    if pred == 1:
        st.success(f"Prediction: **{LABELS[pred]}**")
    else:
        st.error(f"Prediction: **{LABELS[pred]}**")
    st.metric("Model confidence", f"{confidence:.1%}")
    st.progress(p_benign, text=f"Probability benign: {p_benign:.1%}")
    if sample_idx is not None:
        st.info(f"Actual label for this sample: **{LABELS[y_test.iloc[sample_idx]]}**")
