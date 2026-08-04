from fastapi.testclient import TestClient

from career_match_agent.api.main import app
from career_match_agent.services.job_normalizer import create_job_fingerprint


client = TestClient(app)


def create_job_payload() -> dict[str, object]:
    title = "Junior Machine Learning Engineer"
    company = "Example AI GmbH"
    location = "Berlin"
    return {"source_id": "mock:1",
            "provider": "mock",
            "external_id": "1",
            "title": title,
            "company": company,
            "description": ("Develop Python and PyTorch models.\n Fluent English is required.\n We offer hybrid working."),
            "location": location,
            "remote": False,
            "visa_sponsorship": None,
            "employment_types": ["full_time"],
            "raw_employment_types": ["Full Time"],
            "tags": ["Python", "Machine Learning"],
            "url": "https://example.com/jobs/1",
            "posted_at": None,
            "fingerprint": create_job_fingerprint(title=title, company=company, location=location)}


def test_matching_filter_endpoint() -> None:
    response = client.post(
        "/matching/filter",
        json={"profile": {"skills": ["Python", "PyTorch"], "languages": [{"language": "English", "proficiency": "C1"}]},
            "preferences": {"roles": ["Machine Learning Engineer"], "locations": ["Berlin"], "work_modes": ["hybrid"], "employment_types": ["full_time"],
                            "seniority_levels": ["entry_level", "junior"], "required_keywords": ["Python"], "excluded_keywords": ["Principal"],
                            "preferred_languages": ["English"], "maximum_results": 20},
            "jobs": [create_job_payload()]})
    assert response.status_code == 200

    response_payload = response.json()
    assert response_payload["statistics"]["accepted_count"] == 1
    assert response_payload["statistics"]["rejected_count"]== 0

    decision = response_payload["accepted_jobs"][0]
    assert decision["accepted"] is True
    assert decision["matched_roles"] == ["Machine Learning Engineer"]
    assert decision["detected_seniority"] == "junior"
    assert decision["detected_work_modes"] == ["hybrid"]


def test_matching_endpoint_returns_rejection_reasons() -> None:
    job_payload = create_job_payload()
    job_payload["title"] = ("Senior Machine Learning Engineer")
    response = client.post("/matching/filter",
        json={"profile": {"skills": ["Python"], "languages": [{"language": "English", "proficiency": "C1"}]},
              "preferences": {"roles": ["Machine Learning Engineer"],"locations": ["Berlin"],"work_modes": ["hybrid"],
                            "employment_types": ["full_time"], "seniority_levels": ["junior"]},
              "jobs": [job_payload]})
    assert response.status_code == 200

    response_payload = response.json()
    assert response_payload["statistics"]["rejected_count"]== 1

    reason_codes = {reason["code"] for reason in response_payload["rejected_jobs"][0]["rejection_reasons"]}
    assert "seniority_mismatch" in reason_codes


def test_matching_endpoint_rejects_invalid_input() -> None:
    response = client.post("/matching/filter", json={"profile": {}, "preferences": {"roles": []}, "jobs": []})
    assert response.status_code == 422
