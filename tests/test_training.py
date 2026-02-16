"""
Unit tests for model training and evaluation.
"""
import pytest
import numpy as np
from pathlib import Path

from src.preprocessing.preprocessor import prepare_data, load_processed_data
from src.training.train_models import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    train_all_models,
    load_model,
)
from src.training.evaluate import (
    evaluate_model,
    cross_validate_model,
    get_confusion_matrix,
)


# Fixture: load pre-processed data once per test session
@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, y_test = load_processed_data()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

class TestModelBuilders:

    def test_logistic_regression_type(self):
        from sklearn.linear_model import LogisticRegression
        model = build_logistic_regression()
        assert isinstance(model, LogisticRegression)

    def test_random_forest_type(self):
        from sklearn.ensemble import RandomForestClassifier
        model = build_random_forest()
        assert isinstance(model, RandomForestClassifier)

    def test_xgboost_type(self):
        from xgboost import XGBClassifier
        model = build_xgboost()
        assert isinstance(model, XGBClassifier)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:

    def test_train_all_models_returns_dict(self, data):
        X_train, _, y_train, _ = data
        models = train_all_models(X_train, y_train, save=False)
        assert isinstance(models, dict)
        assert len(models) == 3

    def test_models_can_predict(self, data):
        X_train, X_test, y_train, _ = data
        models = train_all_models(X_train, y_train, save=False)
        for name, model in models.items():
            preds = model.predict(X_test)
            assert preds.shape == (X_test.shape[0],), f"{name} predict shape wrong"

    def test_models_predict_proba(self, data):
        X_train, X_test, y_train, _ = data
        models = train_all_models(X_train, y_train, save=False)
        for name, model in models.items():
            proba = model.predict_proba(X_test)
            assert proba.shape == (X_test.shape[0], 2), f"{name} proba shape wrong"
            assert np.allclose(proba.sum(axis=1), 1.0), f"{name} proba doesn't sum to 1"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:

    def test_evaluate_model_keys(self, data):
        X_train, X_test, y_train, y_test = data
        model = build_logistic_regression()
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_metrics_in_valid_range(self, data):
        X_train, X_test, y_train, y_test = data
        model = build_logistic_regression()
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        for key, val in metrics.items():
            if isinstance(val, str):
                continue  # skip 'model' key
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_roc_auc_above_random(self, data):
        X_train, X_test, y_train, y_test = data
        model = build_random_forest()
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["roc_auc"] > 0.5, "Model should beat random"

    def test_confusion_matrix(self, data):
        X_train, X_test, y_train, y_test = data
        model = build_logistic_regression()
        model.fit(X_train, y_train)
        cm = get_confusion_matrix(model, X_test, y_test)
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_test)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:

    def test_cv_returns_dict(self, data):
        X_train, _, y_train, _ = data
        result = cross_validate_model(
            build_logistic_regression(), X_train, y_train, cv_folds=3
        )
        assert isinstance(result, dict)
        assert "accuracy_mean" in result
        assert "roc_auc_mean" in result

    def test_cv_scores_reasonable(self, data):
        X_train, _, y_train, _ = data
        result = cross_validate_model(
            build_logistic_regression(), X_train, y_train, cv_folds=3
        )
        assert 0.0 < result["roc_auc_mean"] < 1.0
        assert result["roc_auc_std"] >= 0.0


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

class TestLoadModel:

    def test_load_random_forest(self):
        model = load_model("random_forest")
        assert hasattr(model, "predict")

    def test_load_logistic_regression(self):
        model = load_model("logistic_regression")
        assert hasattr(model, "predict")

    def test_load_xgboost(self):
        model = load_model("xgboost")
        assert hasattr(model, "predict")

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model")
