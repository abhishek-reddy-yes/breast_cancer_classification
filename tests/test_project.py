"""Quick checks for data, models, saved model and the web app.

Run from the project folder:  pytest
"""
from pathlib import Path

import joblib
import pytest
from sklearn.metrics import roc_auc_score

from src import evaluate
from src.data import load_data, split_data
from src.models import PARAM_GRIDS, get_models

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"


@pytest.fixture(scope="module")
def split():
    X, y = load_data()
    return split_data(X, y)


def test_data_shape_and_labels():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert X.isna().sum().sum() == 0


def test_split_is_stratified(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) + len(X_test) == 569
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_every_model_has_a_tuning_grid():
    assert set(get_models()) == set(PARAM_GRIDS)


@pytest.mark.parametrize("name", list(get_models()))
def test_models_train_and_score_well(split, name):
    X_train, X_test, y_train, y_test = split
    model = get_models()[name].fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, probs) > 0.95


def test_metrics_and_importance(split):
    X_train, X_test, y_train, y_test = split
    model = get_models()["logistic_regression"].fit(X_train, y_train)
    metrics = evaluate.test_metrics(model, X_test, y_test)
    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
    importance = evaluate.feature_importance(model, X_test, y_test)
    assert len(importance) == 30


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run train.py first")
def test_saved_model_predicts(split):
    _, X_test, _, y_test = split
    model = joblib.load(MODEL_PATH)
    assert (model.predict(X_test) == y_test).mean() > 0.95


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run train.py first")
def test_web_app_predicts():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.number_input) == 30
    app.button[0].click().run()
    assert not app.exception
    assert len(app.success) + len(app.error) == 1
