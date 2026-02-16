"""
Predictor module — end-to-end credit risk prediction with explanations.

Loads the trained model and preprocessor, accepts raw user input,
validates, transforms, predicts, and explains.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Any, Dict, Optional

from src.preprocessing.preprocessor import (
    load_preprocessor,
    load_processed_data,
    get_feature_names,
)
from src.training.train_models import load_model
from src.prediction.input_validator import validate_input, get_input_schema
from src.explainability.shap_explainer import explain_prediction as shap_explain

from config import SAVED_MODELS_DIR, ExplainabilityConfig


class CreditRiskPredictor:
    """High-level predictor that combines preprocessing, prediction
    and SHAP explanation in one call."""

    _instance = None        # singleton for web app

    def __init__(self, model_name: str = "random_forest"):
        self.model_name = model_name
        self.model = load_model(model_name)
        self.preprocessor = load_preprocessor()
        self.feature_names = get_feature_names(self.preprocessor)

        # Background data for XGBoost / LinearExplainer SHAP fallback
        X_train, *_ = load_processed_data()
        rng = np.random.default_rng(42)
        n = min(ExplainabilityConfig.SHAP_SAMPLES, X_train.shape[0])
        self._X_background = X_train[rng.choice(X_train.shape[0], n, replace=False)]

    # ------------------------------------------------------------------
    # singleton accessor (avoids reloading on every request)
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, model_name: str = "random_forest") -> "CreditRiskPredictor":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        raw_input: Dict[str, Any],
        explain: bool = True,
    ) -> Dict[str, Any]:
        """Run the full pipeline: validate → transform → predict → explain.

        Parameters
        ----------
        raw_input : dict   JSON-like dict with the 20 feature values.
        explain :  bool    If True, include SHAP explanation.

        Returns
        -------
        dict  with keys: prediction, probability, risk_level,
              explanation (optional), input_summary.
        """
        # 1. Validate
        df, errors = validate_input(raw_input)
        if errors:
            return {"status": "error", "errors": errors}

        # 2. Transform via the fitted preprocessor
        X_transformed = self.preprocessor.transform(df)

        # 3. Predict
        prob = float(self.model.predict_proba(X_transformed)[0, 1])
        pred = int(self.model.predict(X_transformed)[0])

        result: Dict[str, Any] = {
            "status": "success",
            "prediction": pred,
            "prediction_label": "Bad" if pred == 1 else "Good",
            "probability": round(prob, 4),
            "risk_level": _risk_level(prob),
            "model_used": self.model_name,
        }

        # 4. Explain
        if explain:
            explanation = shap_explain(
                self.model,
                X_transformed[0],
                self.feature_names,
                X_background=self._X_background,
            )
            result["explanation"] = {
                "base_value": explanation["base_value"],
                "top_positive": explanation["top_positive"][:ExplainabilityConfig.TOP_FEATURES],
                "top_negative": explanation["top_negative"][:ExplainabilityConfig.TOP_FEATURES],
            }

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def available_models():
        """List model slugs that have been saved."""
        model_dir = Path(SAVED_MODELS_DIR)
        exclude = {"preprocessor"}
        return [p.stem for p in model_dir.glob("*.joblib") if p.stem not in exclude]

    @staticmethod
    def input_schema():
        """Return the JSON input schema for the frontend."""
        return get_input_schema()


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"
