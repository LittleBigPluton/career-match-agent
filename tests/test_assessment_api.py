from fastapi.testclient import TestClient

from career_match_agent.api.main import app
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIRECTORY = PROJECT_ROOT / "data" / "sample" / "hiring_agent_report.txt"
SAMPLE_TEXT_REPORT = SAMPLE_DATA_DIRECTORY.read_text(encoding="utf-8")
SAMPLE_REPORT_BYTES = SAMPLE_TEXT_REPORT.encode("utf-8")

client = TestClient(app)


def test_hiring_agent_parse_endpoint() -> None:
    response = client.post("/assessments/hiring-agent/parse", files={"report": ("hiring_agent_report.txt", SAMPLE_TEXT_REPORT.encode(), "text/plain")},
                           data={"role_name": "software_engineering_intern"})
    assert response.status_code == 200
    response_payload = response.json()
    assert response_payload["candidate_name"] == "B. Bunny"
    assert response_payload["role_name"] == "software_engineering_intern"
    assert response_payload["reported_overall_score"] == -2
    assert response_payload["computed_overall_score"] == -2
    assert response_payload["category_total"] == 3
    assert response_payload["bonus_points"]["total"] == 1
    assert response_payload["deductions"]["total"] == 6
    assert len(response_payload["categories"]) == 4

def test_hiring_agent_context_endpoint_round_trip() -> None:
    parse_response = client.post("/assessments/hiring-agent/parse", files={"report": ("hiring_agent_report.txt", SAMPLE_TEXT_REPORT.encode(), "text/plain")})
    assert parse_response.status_code == 200
    context_response = client.post("/assessments/hiring-agent/context", json={"profile": {"skills": ["Python", "PyTorch"]},"assessment": parse_response.json()})
    assert context_response.status_code == 200
    response_payload = context_response.json()
    assert response_payload["profile"]["skills"] == ["Python", "PyTorch"]
    assert len(response_payload["assessments"]) == 1
    assert len(response_payload["evidence_signals"]) > 4

def test_hiring_agent_endpoint_rejects_wrong_extension() -> None:
    response = client.post("/assessments/hiring-agent/parse", files={"report": ("evaluation.pdf", b"not a supported report", "application/pdf")})
    assert response.status_code == 415
    assert ".txt, .log and .json" in response.json()["detail"]

def test_hiring_agent_endpoint_rejects_invalid_report() -> None:
    response = client.post("/assessments/hiring-agent/parse", files={"report": ("hiring_agent_report.txt", b"This is not a hiring-agent report.", "text/plain",)})
    assert response.status_code == 422
