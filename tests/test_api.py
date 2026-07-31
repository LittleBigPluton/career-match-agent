from fastapi.testclient import TestClient

from career_match_agent.api.main import app

client = TestClient(app)

def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "career-match-agent", "version": "0.1.0"}

def test_preferences_endpoint_returns_normalised_input() -> None:
    request_payload = {"roles": [" Machine Learning Engineer ", "machine learning engineer", "Data Scientist"],
                       "locations": [" Berlin ", "berlin"], "work_modes": ["hybrid"], "maximum_results": 10}

    response = client.post("/preferences", json=request_payload)
    assert response.status_code == 200
    response_payload = response.json()
    assert response_payload["roles"] == ["Machine Learning Engineer", "Data Scientist"]
    assert response_payload["locations"] == ["Berlin"]
    assert response_payload["work_modes"] == ["hybrid"]
    assert response_payload["maximum_results"] == 10

def test_preferences_endpoint_rejects_invalid_work_mode() -> None:
    request_payload = {"roles": ["Machine Learning Engineer"], "work_modes": ["sometimes_remote"]}
    response = client.post("/preferences", json=request_payload)
    assert response.status_code == 422
