"""Load the saved model and predict on a few test-set samples.

Run after train.py:  python predict.py
"""
from pathlib import Path

import joblib

from src.data import load_data, split_data

MODEL_PATH = Path(__file__).parent / "models" / "best_model.joblib"
LABELS = {0: "malignant", 1: "benign"}


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Run `python train.py` first.")
    model = joblib.load(MODEL_PATH)

    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    sample, truth = X_test.head(10), y_test.head(10)

    preds = model.predict(sample)
    probs = model.predict_proba(sample)[:, 1]
    for p, prob, t in zip(preds, probs, truth):
        mark = "ok" if p == t else "WRONG"
        print(f"predicted={LABELS[p]:<10} P(benign)={prob:.3f} actual={LABELS[t]:<10} {mark}")


if __name__ == "__main__":
    main()
