"""
Additional tests to increase code coverage.
"""
import pytest
import warnings
import numpy as np

from src.preprocessing.preprocessor import load_processed_data, load_preprocessor, get_feature_names
from src.training.train_models import load_model
from src.explainability.shap_explainer import (
    plot_waterfall,
    plot_summary,
    plot_bar_importance,
    _compute_shap_values,
    _compute_shap_matrix,
    create_explainer,
)
from src.explainability.lime_explainer import LimeExplainerWrapper, _risk_level
from src.training.evaluate import (
    comparison_table,
    cv_comparison_table,
    print_classification_report,
    evaluate_model,
    cross_validate_model,
)
from src.training.train_models import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    train_all_models,
)


@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, y_test = load_processed_data()
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def feature_names():
    return get_feature_names(load_preprocessor())


# ---------------------------------------------------------------------------
# SHAP plot functions
# ---------------------------------------------------------------------------

class TestSHAPPlots:

    def test_plot_summary_saves_file(self, data, feature_names, tmp_path):
        X_train, X_test, _, _ = data
        rf = load_model("random_forest")
        path = str(tmp_path / "summary.png")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_summary(rf, X_test[:30], feature_names, save_path=path, max_samples=30)
        assert result == path

    def test_plot_bar_importance_saves_file(self, data, feature_names, tmp_path):
        X_train, X_test, _, _ = data
        rf = load_model("random_forest")
        path = str(tmp_path / "bar.png")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_bar_importance(rf, X_test[:30], feature_names, save_path=path, max_samples=30)
        assert result == path

    def test_plot_waterfall_saves_file(self, data, feature_names, tmp_path):
        X_train, X_test, _, _ = data
        rf = load_model("random_forest")
        path = str(tmp_path / "waterfall.png")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_waterfall(rf, X_test[0], feature_names, save_path=path)
        assert result == path

    def test_plot_summary_no_save(self, data, feature_names):
        rf = load_model("random_forest")
        _, X_test, _, _ = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_summary(rf, X_test[:20], feature_names, max_samples=20)
        assert result is None

    def test_plot_bar_no_save(self, data, feature_names):
        rf = load_model("random_forest")
        _, X_test, _, _ = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_bar_importance(rf, X_test[:20], feature_names, max_samples=20)
        assert result is None

    def test_plot_waterfall_no_save(self, data, feature_names):
        rf = load_model("random_forest")
        _, X_test, _, _ = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_waterfall(rf, X_test[0], feature_names)
        assert result is None


# ---------------------------------------------------------------------------
# SHAP XGBoost fallback (KernelExplainer)
# ---------------------------------------------------------------------------

class TestXGBoostFallback:

    def test_xgb_explain_with_background(self, data, feature_names):
        from src.explainability.shap_explainer import explain_prediction
        X_train, X_test, _, _ = data
        xgb = load_model("xgboost")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explain_prediction(xgb, X_test[0], feature_names, X_background=X_train[:20])
        assert result["prediction"] in (0, 1)

    def test_xgb_global_importance_with_background(self, data, feature_names):
        from src.explainability.shap_explainer import get_global_importance
        X_train, X_test, _, _ = data
        xgb = load_model("xgboost")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = get_global_importance(xgb, X_test[:10], feature_names, max_samples=10, X_background=X_train[:10])
        assert len(imp) == len(feature_names)


# ---------------------------------------------------------------------------
# LIME extra coverage
# ---------------------------------------------------------------------------

class TestLIMEExtra:

    def test_lime_plot(self, data, feature_names, tmp_path):
        X_train, X_test, _, _ = data
        rf = load_model("random_forest")
        explainer = LimeExplainerWrapper(X_train, feature_names, num_samples=200)
        path = str(tmp_path / "lime.png")
        result = explainer.plot_explanation(rf, X_test[0], save_path=path, num_features=5)
        assert result == path

    def test_lime_plot_no_save(self, data, feature_names):
        X_train, X_test, _, _ = data
        rf = load_model("random_forest")
        explainer = LimeExplainerWrapper(X_train, feature_names, num_samples=200)
        result = explainer.plot_explanation(rf, X_test[0], num_features=5)
        assert result is None

    def test_risk_level_low(self):
        assert _risk_level(0.1) == "Low"

    def test_risk_level_medium(self):
        assert _risk_level(0.4) == "Medium"

    def test_risk_level_high(self):
        assert _risk_level(0.8) == "High"


# ---------------------------------------------------------------------------
# Evaluation extra coverage
# ---------------------------------------------------------------------------

class TestEvaluationExtra:

    def test_comparison_table(self, data):
        X_train, X_test, y_train, y_test = data
        lr = build_logistic_regression()
        lr.fit(X_train, y_train)
        metrics = [evaluate_model(lr, X_test, y_test, "LR")]
        table = comparison_table(metrics)
        assert "LR" in table.index

    def test_print_classification_report(self, data, capsys):
        X_train, X_test, y_train, y_test = data
        lr = build_logistic_regression()
        lr.fit(X_train, y_train)
        print_classification_report(lr, X_test, y_test, "LR")
        captured = capsys.readouterr()
        assert "LR" in captured.out

    def test_cv_comparison_table(self, data):
        X_train, _, y_train, _ = data
        result = cross_validate_model(build_logistic_regression(), X_train, y_train, cv_folds=2, model_name="LR")
        table = cv_comparison_table([result])
        assert "LR" in table.index


# ---------------------------------------------------------------------------
# Training CLI / train_all save + load
# ---------------------------------------------------------------------------

class TestTrainingExtra:

    def test_train_all_models_with_save(self, data):
        """Just verify train_all_models runs with save=True."""
        X_train, _, y_train, _ = data
        models = train_all_models(X_train, y_train, save=True)
        assert len(models) == 3
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            assert name in models
