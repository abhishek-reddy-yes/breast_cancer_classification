"""Evaluation helpers: metrics and plots."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to files, no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)


def test_metrics(model, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def report(model, X_test, y_test) -> str:
    return classification_report(
        y_test, model.predict(X_test), target_names=["malignant", "benign"]
    )


def save_plots(model, X_test, y_test, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["malignant", "benign"], cmap="Blues"
    )
    plt.title(f"Confusion matrix: {name}")
    plt.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC curve: {name}")
    plt.savefig(out_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()
