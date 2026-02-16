"""
Unit tests for prediction module and input validation.
"""
import pytest
import warnings
import numpy as np

from src.prediction.input_validator import validate_input, get_input_schema
from src.prediction.predictor import CreditRiskPredictor


# ---------------------------------------------------------------------------
# Valid sample input
# ---------------------------------------------------------------------------

VALID_INPUT = {
    "duration_months": 24,
    "credit_amount": 5000,
    "installment_rate": 3,
    "residence_years": 2,
    "age": 35,
    "num_existing_credits": 1,
    "num_dependents": 1,
    "checking_account": "< 0 DM",
    "credit_history": "existing credits paid till now",
    "purpose": "car (new)",
    "savings_account": "< 100 DM",
    "employment_years": "1-4 years",
    "personal_status_sex": "male : single",
    "other_debtors": "none",
    "property": "real estate",
    "other_installments": "none",
    "housing": "own",
    "job": "skilled employee / official",
    "telephone": "yes, registered",
    "foreign_worker": "yes",
}


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestValidateInput:

    def test_valid_input(self):
        df, errors = validate_input(VALID_INPUT)
        assert errors == []
        assert df.shape == (1, 20)

    def test_missing_field(self):
        bad = {k: v for k, v in VALID_INPUT.items() if k != "age"}
        _, errors = validate_input(bad)
        assert any("age" in e for e in errors)

    def test_invalid_numerical_type(self):
        bad = VALID_INPUT.copy()
        bad["age"] = "not a number"
        _, errors = validate_input(bad)
        assert any("age" in e for e in errors)

    def test_numerical_out_of_range(self):
        bad = VALID_INPUT.copy()
        bad["age"] = 5  # below 18
        _, errors = validate_input(bad)
        assert any("age" in e for e in errors)

    def test_invalid_categorical(self):
        bad = VALID_INPUT.copy()
        bad["checking_account"] = "invalid_value"
        _, errors = validate_input(bad)
        assert any("checking_account" in e for e in errors)

    def test_multiple_errors(self):
        bad = {"duration_months": "abc"}  # missing almost everything
        _, errors = validate_input(bad)
        assert len(errors) >= 10


class TestGetInputSchema:

    def test_returns_dict(self):
        schema = get_input_schema()
        assert "fields" in schema

    def test_field_count(self):
        schema = get_input_schema()
        assert len(schema["fields"]) == 20

    def test_numerical_fields(self):
        schema = get_input_schema()
        num_fields = [f for f in schema["fields"] if f["type"] == "number"]
        assert len(num_fields) == 7

    def test_select_fields(self):
        schema = get_input_schema()
        sel_fields = [f for f in schema["fields"] if f["type"] == "select"]
        assert len(sel_fields) == 13

    def test_select_has_options(self):
        schema = get_input_schema()
        for f in schema["fields"]:
            if f["type"] == "select":
                assert len(f["options"]) > 0, f"{f['name']} has no options"


# ---------------------------------------------------------------------------
# Predictor tests
# ---------------------------------------------------------------------------

class TestCreditRiskPredictor:

    @pytest.fixture(scope="class")
    def predictor(self):
        warnings.filterwarnings("ignore")
        return CreditRiskPredictor("random_forest")

    def test_predict_success(self, predictor):
        result = predictor.predict(VALID_INPUT, explain=False)
        assert result["status"] == "success"
        assert result["prediction"] in (0, 1)

    def test_predict_with_explanation(self, predictor):
        result = predictor.predict(VALID_INPUT, explain=True)
        assert "explanation" in result
        assert "top_positive" in result["explanation"]
        assert "top_negative" in result["explanation"]

    def test_predict_probability_range(self, predictor):
        result = predictor.predict(VALID_INPUT, explain=False)
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_risk_level(self, predictor):
        result = predictor.predict(VALID_INPUT, explain=False)
        assert result["risk_level"] in ("Low", "Medium", "High")

    def test_predict_validation_error(self, predictor):
        result = predictor.predict({"duration_months": "abc"})
        assert result["status"] == "error"
        assert len(result["errors"]) > 0

    def test_available_models(self):
        models = CreditRiskPredictor.available_models()
        assert "random_forest" in models
        assert "preprocessor" not in models

    def test_input_schema(self):
        schema = CreditRiskPredictor.input_schema()
        assert len(schema["fields"]) == 20

    def test_singleton(self):
        p1 = CreditRiskPredictor.get_instance("random_forest")
        p2 = CreditRiskPredictor.get_instance("random_forest")
        assert p1 is p2
