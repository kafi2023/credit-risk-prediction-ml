"""
Integration tests for the Credit Risk Prediction ML pipeline.
Tests the full flow from data ingestion to model prediction and explanation
via the Flask API.
"""
import pytest
import json
from web.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_health(client):
    """Test the health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_api_schema(client):
    """Test the schema endpoint."""
    response = client.get('/schema')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'fields' in data
    assert len(data['fields']) == 20

def test_api_models(client):
    """Test the models list endpoint."""
    response = client.get('/models')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'models' in data
    assert isinstance(data['models'], list)

def test_end_to_end_prediction(client):
    """Test the full prediction flow via the API with a valid payload."""
    payload = {
        "duration_months": 24,
        "credit_amount": 2500,
        "installment_rate": 4,
        "residence_years": 4,
        "age": 35,
        "num_existing_credits": 1,
        "num_dependents": 1,
        "checking_account": "0-200 DM",
        "credit_history": "existing credits paid till now",
        "purpose": "radio/television",
        "savings_account": "< 100 DM",
        "employment_years": "1-4 years",
        "personal_status_sex": "male : single",
        "other_debtors": "none",
        "property": "real estate",
        "other_installments": "none",
        "housing": "own",
        "job": "skilled employee / official",
        "telephone": "none",
        "foreign_worker": "yes"
    }

    # Test random forest
    payload["model"] = "random_forest"
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert data["status"] == "success"
    assert "prediction" in data
    assert "prediction_label" in data
    assert data["prediction_label"] in ["Good", "Bad"]
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
    assert "risk_level" in data
    assert "explanation" in data
    
    explanation = data["explanation"]
    assert "base_value" in explanation
    assert "top_positive" in explanation
    assert "top_negative" in explanation

    # Test logistic regression
    payload["model"] = "logistic_regression"
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    
    # Test xgboost
    payload["model"] = "xgboost"
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200

def test_invalid_prediction_payload(client):
    """Test validation errors for invalid input data."""
    payload = {
        "duration_months": -10,  # Invalid range
        "credit_amount": 2500,
        # Missing other fields
    }

    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["status"] == "error"
    assert "errors" in data
    assert isinstance(data["errors"], list)
    assert len(data["errors"]) > 0