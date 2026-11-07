"""Candidate models, each wrapped in a pipeline with its preprocessing."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_models(random_state: int = 42) -> dict[str, Pipeline]:
    # Scaling lives inside the pipeline so it is fit on training folds only
    # (no data leakage during cross-validation).
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=random_state)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            # Calibrated wrapper gives predict_proba (replaces deprecated probability=True).
            ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=random_state), ensemble=False)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=random_state)),
        ]),
    }
