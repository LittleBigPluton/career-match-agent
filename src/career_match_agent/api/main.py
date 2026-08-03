from fastapi import FastAPI

from career_match_agent.api.routes.documents import router as documents_router
from career_match_agent.api.routes.profiles import router as profiles_router
from career_match_agent.models.candidate import JobPreferences
from career_match_agent.api.routes.assessments import router as assessments_router
from career_match_agent.api.routes.jobs import router as jobs_router
from career_match_agent.api.routes.matching import router as matching_router

app = FastAPI(title="CareerMatch Agent API", description="Explainable CV-based job search, ranking and recommendation API.",version="0.1.0")
app.include_router(assessments_router)
app.include_router(documents_router)
app.include_router(jobs_router)
app.include_router(matching_router)
app.include_router(profiles_router)

@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the current health status of the API."""
    return {"status": "healthy", "service": "career-match-agent", "version": "0.1.0"}

@app.post("/preferences", response_model=JobPreferences)
def validate_preferences(preferences: JobPreferences) -> JobPreferences:
    """Validate and normalise candidate job preferences."""
    return preferences
