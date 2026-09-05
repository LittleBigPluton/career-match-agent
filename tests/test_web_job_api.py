import json

from fastapi.testclient import TestClient

from career_match_agent.api.main import app


client = TestClient(app)

def create_html() -> bytes:
    payload = {"@context": "https://schema.org",
               "@type": "JobPosting",
               "identifier": {"value": "job-123"},
               "title": "Machine Learning Engineer",
               "description": ("<p>Build Python ML systems.</p>"),
               "hiringOrganization": {"name": "Example AI GmbH"},
               "jobLocation": {"address": {"addressLocality": "Berlin", "addressCountry": "DE"}}, "employmentType": "FULL_TIME"}

    return (f"""
                <html>
                    <script type="application/ld+json">
                        {json.dumps(payload)}
                    </script>
                </html>
             """).encode()

def test_parse_web_job_endpoint() -> None:
    response = client.post("/jobs/web/parse", files={"file": ("linkedin_job.html", create_html(), "text/html")},
                           data={"source_url": ("https://www.linkedin.com/jobs/view/1234567890/")})
    assert response.status_code == 200

    payload = response.json()
    assert payload["metadata"]["source"] == ("linkedin")
    assert (payload["metadata"]["extraction_strategy"] == "json_ld")
    assert payload["job"]["title"] == ("Machine Learning Engineer")
    assert payload["job"]["provider"] == ("linkedin")

def test_parse_web_job_rejects_unknown_site() -> None:
    response = client.post("/jobs/web/parse", files={"file": ("job.html", create_html(), "text/html")}, data={"source_url": ("https://example.com/job/1")})
    assert response.status_code == 422

def test_parse_web_job_rejects_wrong_file_type() -> None:
    response = client.post("/jobs/web/parse", files={"file": ("job.pdf", b"not html", "application/pdf")}, data={"source_url": ("https://www.linkedin.com/jobs/view/123/")})
    assert response.status_code == 415
