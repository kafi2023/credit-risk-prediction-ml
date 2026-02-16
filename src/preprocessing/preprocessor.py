"""
Preprocessing pipeline for the German Credit Dataset.

Builds a scikit-learn ColumnTransformer + Pipeline that:
  • Scales numerical features (median imputation → StandardScaler)
  • Encodes categorical features (constant imputation → OneHotEncoder)
  • Provides train/test split with stratification
  • Supports SMOTE for class-imbalance handling
  • Is fully serializable with joblib for reuse at prediction time
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.data_loader import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    get_feature_target_split,
    load_german_credit,
)

# ---------------------------------------------------------------------------
# Project paths (relative to repo root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
MODELS_DIR = _PROJECT_ROOT / "models" / "saved_models"


# ---------------------------------------------------------------------------
# Build the sklearn preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer for the German Credit features.

    Numerical  → median impute → StandardScaler
    Categorical → constant impute ("missing") → OneHotEncoder
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Split + preprocess convenience function
# ---------------------------------------------------------------------------

def prepare_data(
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = False,
    save: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    """Load → split → fit preprocessor on train → transform both sets.

    Parameters
    ----------
    test_size : float
        Fraction of data held out for testing (stratified).
    random_state : int
        Reproducibility seed.
    use_smote : bool
        If True, apply SMOTE to the *training* set after preprocessing
        to handle the 70/30 class imbalance.
    save : bool
        Persist processed arrays and the fitted preprocessor to disk.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_german_credit()
    X, y = get_feature_target_split(df)

    # Stratified split — keeps 70/30 ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # Fit on train only → transform both (no data leakage)
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Optional: SMOTE on training set only
    if use_smote:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=random_state)
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)

    if save:
        _save_artifacts(X_train_processed, X_test_processed, y_train, y_test, preprocessor)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_artifacts(X_train, X_test, y_train, y_test, preprocessor) -> None:
    """Save processed arrays and the fitted preprocessor to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", np.array(y_train))
    np.save(PROCESSED_DIR / "y_test.npy", np.array(y_test))

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")


def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load previously saved processed data from disk."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def load_preprocessor() -> ColumnTransformer:
    """Load the fitted preprocessor from disk."""
    return joblib.load(MODELS_DIR / "preprocessor.joblib")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names from a fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Preparing data …")
    X_tr, X_te, y_tr, y_te, prep = prepare_data(use_smote=False, save=True)

    feature_names = get_feature_names(prep)
    print(f"\n✅  Done!")
    print(f"  X_train : {X_tr.shape}")
    print(f"  X_test  : {X_te.shape}")
    print(f"  y_train : {y_tr.shape}  (Bad={int(np.sum(y_tr))}, Good={int(len(y_tr) - np.sum(y_tr))})")
    print(f"  y_test  : {y_te.shape}  (Bad={int(np.sum(y_te))}, Good={int(len(y_te) - np.sum(y_te))})")
    print(f"  Features: {len(feature_names)}")
    print(f"  Saved to: {PROCESSED_DIR}")
    print(f"  Preprocessor: {MODELS_DIR / 'preprocessor.joblib'}")
