"""
Unit tests for SHAP and LIME explainability modules.
"""
import pytest
import warnings
import numpy as np

from src.preprocessing.preprocessor import load_processed_data, load_preprocessor, get_feature_names
from src.training.train_models import load_model
from src.explainability.shap_explainer import (
    create_explainer,
    explain_prediction,
    get_global_importance,
)
from src.explainability.lime_explainer import create_lime_explainer


# Shared fixtures
@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, y_test = load_processed_data()
    preprocessor = load_preprocessor()
    feature_names = get_feature_names(preprocessor)
    return X_train, X_test, y_train, y_test, feature_names


@pytest.fixture(scope="module")
def rf_model():
    return load_model("random_forest")


@pytest.fixture(scope="module")
def lr_model():
    return load_model("logistic_regression")


# ---------------------------------------------------------------------------
# SHAP tests
# ---------------------------------------------------------------------------

class TestSHAPExplainer:

    def test_create_explainer_rf(self, rf_model):
        explainer = create_explainer(rf_model)
        assert explainer is not None

    def test_create_explainer_lr(self, lr_model, data):
        X_train = data[0]
        explainer = create_explainer(lr_model, X_train)
        assert explainer is not None

    def test_explain_prediction_keys(self, rf_model, data):
        X_train, X_test, _, _, feature_names = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explain_prediction(rf_model, X_test[0], feature_names)
        assert "prediction" in result
        assert "probability" in result
        assert "risk_level" in result
        assert "shap_values" in result
        assert "feature_contributions" in result

    def test_explain_prediction_probability_range(self, rf_model, data):
        _, X_test, _, _, feature_names = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explain_prediction(rf_model, X_test[0], feature_names)
        assert 0.0 <= result["probability"] <= 1.0

    def test_explain_prediction_risk_levels(self, rf_model, data):
        _, X_test, _, _, feature_names = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explain_prediction(rf_model, X_test[0], feature_names)
        assert result["risk_level"] in ("Low", "Medium", "High")

    def test_shap_values_length(self, rf_model, data):
        _, X_test, _, _, feature_names = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explain_prediction(rf_model, X_test[0], feature_names)
        assert len(result["shap_values"]) == len(feature_names)

    def test_feature_contributions_sorted(self, rf_model, data):
        _, X_test, _, _, feature_names = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explain_prediction(rf_model, X_test[0], feature_names)
        contribs = result["feature_contributions"]
        abs_vals = [abs(c["contribution"]) for c in contribs]
        assert abs_vals == sorted(abs_vals, reverse=True)


class TestGlobalImportance:

    def test_returns_dataframe(self, rf_model, data):
        _, X_test, _, _, feature_names = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = get_global_importance(rf_model, X_test[:50], feature_names, max_samples=50)
        assert hasattr(imp, "columns")
        assert "feature" in imp.columns
        assert "importance" in imp.columns

    def test_sorted_descending(self, rf_model, data):
        _, X_test, _, _, feature_names = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = get_global_importance(rf_model, X_test[:50], feature_names, max_samples=50)
        vals = imp["importance"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_all_features_present(self, rf_model, data):
        _, X_test, _, _, feature_names = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = get_global_importance(rf_model, X_test[:50], feature_names, max_samples=50)
        assert len(imp) == len(feature_names)


# ---------------------------------------------------------------------------
# LIME tests
# ---------------------------------------------------------------------------

class TestLIMEExplainer:

    def test_create_lime_explainer(self, data):
        X_train, _, _, _, feature_names = data
        explainer = create_lime_explainer(X_train, feature_names)
        assert explainer is not None

    def test_lime_explain_prediction_keys(self, rf_model, data):
        X_train, X_test, _, _, feature_names = data
        explainer = create_lime_explainer(X_train, feature_names, num_samples=500)
        result = explainer.explain_prediction(rf_model, X_test[0], num_features=5)
        assert "prediction" in result
        assert "probability" in result
        assert "risk_level" in result
        assert "feature_contributions" in result

    def test_lime_probability_range(self, rf_model, data):
        X_train, X_test, _, _, feature_names = data
        explainer = create_lime_explainer(X_train, feature_names, num_samples=500)
        result = explainer.explain_prediction(rf_model, X_test[0], num_features=5)
        assert 0.0 <= result["probability"] <= 1.0

    def test_lime_contributions_count(self, rf_model, data):
        X_train, X_test, _, _, feature_names = data
        explainer = create_lime_explainer(X_train, feature_names, num_samples=500)
        result = explainer.explain_prediction(rf_model, X_test[0], num_features=5)
        assert len(result["feature_contributions"]) == 5
