"""
Model explainability module (SHAP/LIME).
Implements SHAP and LIME for model interpretability.
"""

from src.explainability.shap_explainer import (
    create_explainer,
    explain_prediction,
    get_global_importance,
    plot_waterfall,
    plot_summary,
    plot_bar_importance,
)
from src.explainability.lime_explainer import (
    create_lime_explainer,
    LimeExplainerWrapper,
)

__all__ = [
    "create_explainer",
    "explain_prediction",
    "get_global_importance",
    "plot_waterfall",
    "plot_summary",
    "plot_bar_importance",
    "create_lime_explainer",
    "LimeExplainerWrapper",
]
