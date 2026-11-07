# Breast Cancer Classification

Binary classification of tumors as **malignant** or **benign** from 30 numeric
cell-nucleus features (Wisconsin Diagnostic Breast Cancer dataset, 569 samples,
bundled with scikit-learn — no download needed).

## What it covers

| Step | Where |
|------|-------|
| Data loading, inspection, stratified split | `src/data.py` |
| Preprocessing (scaling) inside pipelines to avoid leakage | `src/models.py` |
| Model comparison with 5-fold CV: Logistic Regression, KNN, SVM, Random Forest | `train.py` |
| Test-set metrics (accuracy, F1, ROC AUC), confusion matrix, ROC curve | `src/evaluate.py` |
| Saving and reloading the best model | `train.py`, `predict.py` |

## Setup

```bash
# Ubuntu/Debian, once: sudo apt install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd breast_cancer_classification
python train.py     # trains, compares, evaluates, saves models/ and reports/
python predict.py   # loads saved model and predicts on sample rows
```

Outputs:
- `models/best_model.joblib` — the best pipeline by CV ROC AUC
- `reports/results.csv` — CV and test metrics for every model
- `reports/confusion_matrix.png`, `reports/roc_curve.png`

## Next ideas

- Hyperparameter tuning with `GridSearchCV`
- Feature importance / permutation importance
- Try gradient boosting (`HistGradientBoostingClassifier`)
