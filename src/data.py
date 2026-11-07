"""Data loading and preprocessing."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Wisconsin breast cancer dataset (bundled with scikit-learn).

    Target: 0 = malignant, 1 = benign.
    """
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split so both sets keep the same class balance."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def describe(X: pd.DataFrame, y: pd.Series) -> None:
    print(f"Samples: {X.shape[0]}  Features: {X.shape[1]}")
    print(f"Missing values: {int(X.isna().sum().sum())}")
    counts = y.value_counts().rename({0: "malignant", 1: "benign"})
    print("Class balance:", counts.to_dict())
