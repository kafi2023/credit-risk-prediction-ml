"""
Model evaluation utilities.

Provides reusable functions for computing classification metrics,
printing comparison tables, and plotting confusion matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Evaluate a trained model and return a dict of metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }
    return metrics


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Run stratified k-fold CV and return mean ± std of key metrics."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1,
    )

    summary = {"model": model_name}
    for metric in scoring:
        key = f"test_{metric}"
        summary[f"{metric}_mean"] = results[key].mean()
        summary[f"{metric}_std"] = results[key].std()

    return summary


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def comparison_table(results: List[Dict[str, float]]) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of evaluate_model results."""
    df = pd.DataFrame(results).set_index("model")
    return df.round(4)


def cv_comparison_table(cv_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a pretty CV comparison table with mean ± std formatting."""
    rows = []
    for r in cv_results:
        row = {"Model": r["model"]}
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mean = r[f"{m}_mean"]
            std = r[f"{m}_std"]
            row[m.upper()] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_classification_report(model, X_test, y_test, model_name="Model"):
    """Print sklearn classification report."""
    y_pred = model.predict(X_test)
    print(f"\n{'='*60}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["Good (0)", "Bad (1)"]))


def get_confusion_matrix(model, X_test, y_test):
    """Return confusion matrix as a 2×2 numpy array."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)
