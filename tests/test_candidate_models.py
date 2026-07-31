import pytest
from pydantic import ValidationError

from career_match_agent.models.candidate import JobPreferences, WorkMode


def test_job_preferences_accept_valid_input() -> None:
    preferences = JobPreferences(roles=["Machine Learning Engineer", "Data Scientist"], locations=["Berlin", "Munich"], work_modes=[WorkMode.HYBRID], maximum_results=10)
    assert preferences.roles == ["Machine Learning Engineer", "Data Scientist"]
    assert preferences.locations == ["Berlin", "Munich"]
    assert preferences.work_modes == [WorkMode.HYBRID]
    assert preferences.maximum_results == 10

def test_job_preferences_clean_duplicate_values() -> None:
    preferences = JobPreferences(roles=[" Machine Learning Engineer ", "machine learning engineer", "", "Data Scientist",], locations=[" Berlin ", "berlin", "Munich"])
    assert preferences.roles == ["Machine Learning Engineer", "Data Scientist"]
    assert preferences.locations == ["Berlin", "Munich"]

def test_job_preferences_require_at_least_one_role() -> None:
    with pytest.raises(ValidationError):
        JobPreferences(roles=[])

def test_maximum_results_must_not_exceed_limit() -> None:
    with pytest.raises(ValidationError):
        JobPreferences(roles=["Machine Learning Engineer"], maximum_results=101)
