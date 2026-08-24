import json

from career_match_agent.models.agent import AgentSearchPlan
from career_match_agent.models.candidate import (
    EmploymentType,
    JobPreferences,
    WorkMode
)
from career_match_agent.models.matching import JobFilterPolicy
from career_match_agent.services.search_planner import (
    build_job_search_query,
    normalize_broadened_plan,
    parse_search_plan_response
)
from career_match_agent.models.job import JobSearchMatchScope

def test_parse_search_plan_response() -> None:
    response_content = json.dumps({"keywords": ["Machine Learning Engineer", "ML Engineer"], "max_pages": 1, "maximum_results": 100,
                                    "rationale": ("Start with direct target-role titles.")})

    plan = parse_search_plan_response(response_content)
    assert plan.keywords == ["Machine Learning Engineer", "ML Engineer"]
    assert plan.max_pages == 1

def test_broadened_plan_preserves_previous_keywords() -> None:
    previous_plan = AgentSearchPlan(keywords=["Machine Learning Engineer"], max_pages=1, maximum_results=50, rationale="Initial precise search.")
    new_plan = AgentSearchPlan(keywords=["AI Engineer", "ML Engineer"], max_pages=1, maximum_results=40, rationale="Broaden role-title vocabulary.")
    broadened_plan = normalize_broadened_plan(new_plan, previous_plan=previous_plan)
    assert broadened_plan.keywords == ["Machine Learning Engineer", "AI Engineer", "ML Engineer"]
    assert broadened_plan.max_pages == 2
    assert broadened_plan.maximum_results == 50

def test_retrieval_query_preserves_hard_preferences() -> None:
    preferences = JobPreferences(roles=["Machine Learning Engineer"], locations=["Berlin"], work_modes=[WorkMode.ON_SITE], employment_types=[EmploymentType.FULL_TIME])
    plan = AgentSearchPlan(keywords=["AI Engineer"], max_pages=2, maximum_results=80, rationale=("Broaden title retrieval."))
    query = build_job_search_query(plan=plan, preferences=preferences, policy=JobFilterPolicy(), visa_sponsorship=True)
    normalized_keywords = {keyword.casefold() for keyword in query.keywords}
    assert "ai engineer" in normalized_keywords
    assert ("machine learning engineer" in normalized_keywords)
    assert query.locations == ["Berlin"]
    assert query.remote_only is False
    assert query.visa_sponsorship is True
    assert query.employment_types == [EmploymentType.FULL_TIME]
    assert query.max_pages == 2
    assert query.maximum_results == 80
    assert (query.match_scope == JobSearchMatchScope.BROAD)
