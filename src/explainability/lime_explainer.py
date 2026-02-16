"""
LIME explainability module.

Provides a secondary, model-agnostic explanation method using
Local Interpretable Model-agnostic Explanations (LIME).
"""

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# LIME Explainer wrapper
# ---------------------------------------------------------------------------

class LimeExplainerWrapper:
    """Thin wrapper around lime.lime_tabular.LimeTabularExplainer."""

    def __init__(
        self,
        X_train: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        num_samples: int = 5000,
    ):
        self.feature_names = feature_names
        self.class_names = class_names or ["Good (0)", "Bad (1)"]
        self.num_samples = num_samples
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=self.class_names,
            categorical_features=categorical_features,
            mode="classification",
            discretize_continuous=True,
        )

    def explain_prediction(
        self,
        model,
        X_instance: np.ndarray,
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """Explain a single prediction and return structured data.

        Parameters
        ----------
        model : fitted estimator with predict_proba
        X_instance : 1-D array (single sample)
        num_features : number of top features to include

        Returns
        -------
        dict with keys: prediction, probability, risk_level,
                        feature_contributions, intercept, score
        """
        X_instance = np.atleast_1d(X_instance).ravel()

        explanation = self.explainer.explain_instance(
            X_instance,
            model.predict_proba,
            num_features=num_features,
            num_samples=self.num_samples,
        )

        prob = float(model.predict_proba(X_instance.reshape(1, -1))[0, 1])
        pred = int(model.predict(X_instance.reshape(1, -1))[0])

        # Extract feature contributions for class 1 (Bad)
        contrib_list = explanation.as_list(label=1)
        contributions = [
            {"feature": feat, "contribution": round(weight, 4)}
            for feat, weight in contrib_list
        ]

        return {
            "prediction": pred,
            "probability": prob,
            "risk_level": _risk_level(prob),
            "intercept": float(explanation.intercept[1]),
            "score": float(explanation.score),
            "feature_contributions": contributions,
        }

    def plot_explanation(
        self,
        model,
        X_instance: np.ndarray,
        save_path: Optional[str] = None,
        num_features: int = 10,
    ) -> Optional[str]:
        """Generate a LIME explanation plot."""
        X_instance = np.atleast_1d(X_instance).ravel()

        explanation = self.explainer.explain_instance(
            X_instance,
            model.predict_proba,
            num_features=num_features,
            num_samples=self.num_samples,
        )

        fig = explanation.as_pyplot_figure(label=1)
        fig.set_size_inches(10, 6)
        plt.title("LIME Explanation — Class 1 (Bad Credit Risk)")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        plt.close(fig)
        return None


# ---------------------------------------------------------------------------
# Convenience function (matches shap_explainer API style)
# ---------------------------------------------------------------------------

def create_lime_explainer(
    X_train: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    categorical_features: Optional[List[int]] = None,
    num_samples: int = 5000,
) -> LimeExplainerWrapper:
    """Factory that mirrors the SHAP module's create_explainer."""
    return LimeExplainerWrapper(
        X_train=X_train,
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=categorical_features,
        num_samples=num_samples,
    )


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"
