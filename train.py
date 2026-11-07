"""End-to-end pipeline: load data -> compare models with CV -> tune best -> evaluate -> save.

Run from this folder:  python train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.data import describe, load_data, split_data
from src.evaluate import (
    feature_importance,
    report,
    save_importance_plot,
    save_plots,
    test_metrics,
)
from src.models import PARAM_GRIDS, get_models

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SEED = 42


def main() -> None:
    print("== Data ==")
    X, y = load_data()
    describe(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=SEED)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    print("== 5-fold cross-validation on training set (ROC AUC) ==")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = get_models(SEED)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = (scores.mean(), scores.std())
        print(f"{name:<20} {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_results, key=lambda n: cv_results[n][0])
    print(f"\nBest model: {best_name}\n")

    print(f"== Hyperparameter tuning: {best_name} (GridSearchCV, ROC AUC) ==")
    search = GridSearchCV(
        models[best_name], PARAM_GRIDS[best_name], cv=cv, scoring="roc_auc", n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"CV ROC AUC: default {cv_results[best_name][0]:.4f} -> tuned {search.best_score_:.4f}\n")

    print("== Test set evaluation (all models, refit on full training set) ==")
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append({"model": name, "cv_roc_auc": cv_results[name][0],
                     **test_metrics(model, X_test, y_test)})
    rows.append({"model": f"{best_name}_tuned", "cv_roc_auc": search.best_score_,
                 **test_metrics(search.best_estimator_, X_test, y_test)})
    results = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    print(results.to_string(index=False, float_format="%.4f"))

    # Keep the tuned version only if it actually beat the default in cross-validation.
    if search.best_score_ > cv_results[best_name][0]:
        best, best_label = search.best_estimator_, f"{best_name} (tuned)"
    else:
        best, best_label = models[best_name], best_name
    print(f"\nSaved model: {best_label}")
    print(f"\nClassification report ({best_label}):")
    print(report(best, X_test, y_test))

    print("== Top 10 features (permutation importance on test set) ==")
    importance = feature_importance(best, X_test, y_test, random_state=SEED)
    print(importance.head(10).to_string(float_format="%.4f"), "\n")

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "results.csv", index=False)
    importance.rename("importance").to_csv(REPORTS_DIR / "feature_importance.csv")
    save_plots(best, X_test, y_test, REPORTS_DIR, best_label)
    save_importance_plot(importance, REPORTS_DIR, best_label)
    joblib.dump(best, MODELS_DIR / "best_model.joblib")
    print(f"Saved model to {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved results and plots to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
