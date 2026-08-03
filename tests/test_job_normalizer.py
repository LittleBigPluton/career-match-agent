from career_match_agent.models.candidate import EmploymentType
from career_match_agent.models.job import JobPosting
from career_match_agent.services.job_normalizer import (
    create_job_fingerprint,
    deduplicate_jobs,
    html_to_plain_text,
    normalize_employment_types
)


def make_job(*, source_id: str, fingerprint: str) -> JobPosting:
    return JobPosting(source_id=source_id,
                      provider="mock",
                      external_id=source_id,
                      title="Machine Learning Engineer",
                      company="Example GmbH",
                      description="Develop machine-learning models.",
                      location="Berlin",
                      remote=False,
                      url="https://example.com/jobs/1",
                      fingerprint=fingerprint)

def test_html_to_plain_text_removes_markup() -> None:
    description = """
                  &lt;h2&gt;Responsibilities&lt;/h2&gt;
                  &lt;ul&gt;
                    &lt;li&gt;Build ML models&lt;/li&gt;
                    &lt;li&gt;Deploy APIs &amp;amp; services&lt;/li&gt;
                  &lt;/ul&gt;
                  """
    plain_text = html_to_plain_text(description)
    assert "<h2>" not in plain_text
    assert "Responsibilities" in plain_text
    assert "Build ML models" in plain_text
    assert "Deploy APIs & services" in plain_text


def test_normalize_employment_types() -> None:
    employment_types = normalize_employment_types(["Full Time", "Teilzeit", "Internship", "unknown"])
    assert employment_types == [EmploymentType.FULL_TIME, EmploymentType.PART_TIME, EmploymentType.INTERNSHIP]

def test_job_fingerprint_is_case_insensitive() -> None:
    first_fingerprint = create_job_fingerprint(title="Machine Learning Engineer", company="Example GmbH", location="Berlin")
    second_fingerprint = create_job_fingerprint(title="machine learning engineer", company="EXAMPLE GMBH", location="berlin")
    assert first_fingerprint == second_fingerprint

def test_deduplicate_jobs_uses_fingerprint() -> None:
    fingerprint = create_job_fingerprint(title="Machine Learning Engineer", company="Example GmbH", location="Berlin")
    result = deduplicate_jobs([make_job(source_id="provider-a:1", fingerprint=fingerprint), make_job(source_id="provider-b:99", fingerprint=fingerprint)])
    assert len(result.jobs) == 1
    assert result.duplicate_count == 1
