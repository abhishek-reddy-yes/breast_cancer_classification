"""Evaluation helpers: metrics and plots."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to files, no display needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
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


def feature_importance(model, X_test, y_test, random_state: int = 42) -> pd.Series:
    """Permutation importance: how much ROC AUC drops when a feature is shuffled."""
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=random_state
    )
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)


def save_importance_plot(importance: pd.Series, out_dir: Path, name: str, top: int = 15) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    top_features = importance.head(top).iloc[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(top_features.index, top_features.values, color="#3b6ea5")
    plt.xlabel("Drop in ROC AUC when shuffled")
    plt.title(f"Top {top} features: {name}")
    plt.savefig(out_dir / "feature_importance.png", bbox_inches="tight")
    plt.close()
