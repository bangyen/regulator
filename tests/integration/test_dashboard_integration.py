"""Integration tests for Flask dashboard."""

import pytest

from dashboard.main import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_dashboard_index(client):
    """Test dashboard index route."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Regulator" in response.data


def test_api_data_no_logs(client):
    """Test API data endpoint when no logs exist."""
    response = client.get("/api/data")
    assert response.status_code in [200, 404]


def test_api_experiments(client):
    """Test API experiments endpoint."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    assert response.content_type == "application/json"
