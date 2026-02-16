"""
SHAP explainability module.

Provides functions to explain individual predictions and compute
global feature importance using TreeSHAP (RF, XGBoost) and
LinearSHAP (Logistic Regression).
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for server use
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------

def create_explainer(model, X_background: Optional[np.ndarray] = None) -> shap.Explainer:
    """Create the appropriate SHAP explainer for the model type.

    • TreeExplainer  → RandomForest, XGBoost
    • LinearExplainer → LogisticRegression
    • KernelExplainer → fallback for any model (slower)
    """
    model_type = type(model).__name__

    if model_type == "RandomForestClassifier":
        return shap.TreeExplainer(model)
    elif model_type == "XGBClassifier":
        # Use TreeExplainer when SHAP/XGBoost versions are compatible;
        # fall back to KernelExplainer with k-means summarised background.
        try:
            return shap.TreeExplainer(model)
        except (ValueError, TypeError):
            if X_background is not None:
                bg = shap.kmeans(X_background, min(10, X_background.shape[0]))
                return shap.KernelExplainer(model.predict_proba, bg)
            raise ValueError(
                "XGBoost TreeExplainer failed and no X_background provided "
                "for fallback. Pass X_background (training data) to fix."
            )
    elif model_type == "LogisticRegression":
        if X_background is not None:
            return shap.LinearExplainer(model, X_background)
        return shap.LinearExplainer(model, np.zeros((1, model.n_features_in_)))
    else:
        # Generic fallback
        if X_background is None:
            raise ValueError("X_background required for KernelExplainer")
        return shap.KernelExplainer(model.predict_proba, X_background)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_shap_values(explainer, X: np.ndarray):
    """Return (shap_vals_1d, base_value_scalar) for class-1, handling all APIs.

    SHAP 0.49+ TreeExplainer returns shape (n_samples, n_features, n_classes).
    Older versions returned a list of arrays. LinearExplainer returns 2-D.
    shap.Explainer returns Explanation objects.
    """
    # New-style Explainer (shap.Explainer) — no shap_values method
    if isinstance(explainer, shap.Explainer) and not hasattr(explainer, "shap_values"):
        explanation = explainer(X)
        vals = np.array(explanation.values)
        bv = np.array(explanation.base_values)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
            bv = bv[:, 1] if bv.ndim == 2 else bv
        vals = vals.squeeze()
        if vals.ndim > 1:
            vals = vals[0]
        bv_scalar = float(np.atleast_1d(bv).ravel()[0])
        return vals, bv_scalar

    # Old-style (TreeExplainer / LinearExplainer)
    shap_values = explainer.shap_values(X)
    sv = np.array(shap_values)

    # 3-D: (n_samples, n_features, n_classes) — SHAP 0.49 TreeExplainer
    if sv.ndim == 3:
        shap_vals = sv[:, :, 1].squeeze()   # class-1
        ev = np.atleast_1d(explainer.expected_value)
        base_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
    # list of 2 arrays: older SHAP TreeExplainer
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals = np.array(shap_values[1]).squeeze()
        base_value = float(np.atleast_1d(explainer.expected_value)[1])
    # 2-D: (n_samples, n_features) — LinearExplainer
    else:
        shap_vals = sv.squeeze()
        ev = np.atleast_1d(explainer.expected_value)
        base_value = float(ev[1]) if len(ev) > 1 else float(ev[0])

    if shap_vals.ndim > 1:
        shap_vals = shap_vals[0]

    return shap_vals, base_value


def _compute_shap_matrix(explainer, X: np.ndarray):
    """Return 2-D shap matrix (n_samples, n_features) for class-1."""
    if isinstance(explainer, shap.Explainer) and not hasattr(explainer, "shap_values"):
        explanation = explainer(X)
        vals = np.array(explanation.values)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return vals

    shap_values = explainer.shap_values(X)
    sv = np.array(shap_values)

    if sv.ndim == 3:
        return sv[:, :, 1]
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        return np.array(shap_values[1])
    return sv if sv.ndim == 2 else sv.reshape(X.shape[0], -1)


# ---------------------------------------------------------------------------
# Individual prediction explanation
# ---------------------------------------------------------------------------

def explain_prediction(
    model,
    X_instance: np.ndarray,
    feature_names: List[str],
    X_background: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Explain a single prediction and return structured data.

    Parameters
    ----------
    model : fitted sklearn/xgboost estimator
    X_instance : 1-D or 2-D array (single row)
    feature_names : list of feature names matching columns
    X_background : optional background data for LinearExplainer

    Returns
    -------
    dict with keys: shap_values, base_value, prediction, probability,
                    feature_names, feature_contributions
    """
    X_instance = np.atleast_2d(X_instance)
    explainer = create_explainer(model, X_background)

    # Compute SHAP values — handle both old and new SHAP APIs
    shap_vals, base_value = _compute_shap_values(explainer, X_instance)

    # Prediction
    prob = float(model.predict_proba(X_instance)[0, 1])
    pred = int(model.predict(X_instance)[0])

    # Top contributing features (sorted by |SHAP|)
    contributions = sorted(
        zip(feature_names, shap_vals.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "shap_values": shap_vals.tolist(),
        "base_value": float(base_value),
        "prediction": pred,
        "probability": prob,
        "risk_level": _risk_level(prob),
        "feature_names": feature_names,
        "feature_contributions": [
            {"feature": f, "contribution": round(v, 4)} for f, v in contributions
        ],
        "top_positive": [
            {"feature": f, "contribution": round(v, 4)}
            for f, v in contributions if v > 0
        ][:10],
        "top_negative": [
            {"feature": f, "contribution": round(v, 4)}
            for f, v in contributions if v < 0
        ][:10],
    }


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"


# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------

def get_global_importance(
    model,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 200,
    X_background: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute mean |SHAP| importance for all features.

    Returns a DataFrame sorted by importance (descending).
    """
    if X.shape[0] > max_samples:
        idx = np.random.default_rng(42).choice(X.shape[0], max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = create_explainer(model, X_background)
    shap_vals = _compute_shap_matrix(explainer, X_sample)

    importance = np.abs(shap_vals).mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Visualisation helpers  (save to file, return bytes for web)
# ---------------------------------------------------------------------------

def plot_waterfall(
    model,
    X_instance: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    X_background: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Generate a SHAP waterfall plot for one instance."""
    X_instance = np.atleast_2d(X_instance)
    explainer = create_explainer(model, X_background)
    vals, base = _compute_shap_values(explainer, X_instance)

    explanation = shap.Explanation(
        values=vals,
        base_values=base,
        data=X_instance[0],
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    plt.close()
    return None


def plot_summary(
    model,
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    max_samples: int = 200,
    X_background: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Generate a SHAP beeswarm / summary plot."""
    if X.shape[0] > max_samples:
        idx = np.random.default_rng(42).choice(X.shape[0], max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = create_explainer(model, X_background)
    shap_vals = _compute_shap_matrix(explainer, X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    plt.close()
    return None


def plot_bar_importance(
    model,
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    max_samples: int = 200,
    X_background: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Generate a SHAP bar plot of mean |SHAP| values."""
    if X.shape[0] > max_samples:
        idx = np.random.default_rng(42).choice(X.shape[0], max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = create_explainer(model, X_background)
    shap_vals = _compute_shap_matrix(explainer, X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    plt.close()
    return None
