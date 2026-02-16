"""
Tests for Flask web application endpoints.
"""
import pytest
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from web.app import app

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


@pytest.fixture
def client():
    """Create test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_schema_endpoint(client):
    response = client.get("/schema")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["fields"]) == 20


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "random_forest" in data["models"]


def test_predict_valid(client):
    warnings.filterwarnings("ignore")
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["prediction"] in (0, 1)
    assert "explanation" in data


def test_predict_validation_error(client):
    response = client.post("/predict", json={"duration_months": "abc"})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert len(data["errors"]) > 0


def test_predict_empty_body(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    # Should return 400 or 500 with error
    assert response.status_code in (400, 500)
