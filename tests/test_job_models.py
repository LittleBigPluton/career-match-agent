import pytest
from pydantic import ValidationError

from career_match_agent.models.job import JobSearchQuery


def test_job_search_query_cleans_values() -> None:
    query = JobSearchQuery(
        keywords=[" Machine Learning Engineer ", "machine learning engineer", "Data Scientist"],
        locations=["Berlin", "berlin", "Munich"])

    assert query.keywords == ["Machine Learning Engineer", "Data Scientist"]
    assert query.locations == ["Berlin", "Munich"]

def test_job_search_query_requires_keyword() -> None:
    with pytest.raises(ValidationError, match="non-empty keyword"):
        JobSearchQuery(keywords=["", "   "])

def test_job_search_query_limits_pages() -> None:
    with pytest.raises(ValidationError):
        JobSearchQuery(keywords=["Data Scientist"], max_pages=6)
