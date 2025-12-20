"""
Tests for Flask web application
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from web.app import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Test home page loads"""
    response = client.get('/')
    assert response.status_code == 200

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_predict_endpoint(client):
    """Test prediction endpoint (placeholder)"""
    response = client.post('/predict', 
                          json={'test': 'data'},
                          content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
