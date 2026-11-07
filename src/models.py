"""Candidate models, each wrapped in a pipeline with its preprocessing."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
        "gradient_boosting": Pipeline([
            ("clf", HistGradientBoostingClassifier(random_state=random_state)),
        ]),
    }


# Small hyperparameter grids, used to tune whichever model wins cross-validation.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "knn": {"clf__n_neighbors": [3, 5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"]},
    "svm": {
        "clf__estimator__C": [0.1, 1, 10, 100],
        "clf__estimator__gamma": ["scale", 0.001, 0.01, 0.1],
    },
    "random_forest": {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 5, 10]},
    "gradient_boosting": {"clf__learning_rate": [0.03, 0.1, 0.3], "clf__max_depth": [None, 3, 5]},
}
