from fastapi import FastAPI
from career_match_agent.models.candidate import JobPreferences

app = FastAPI(title="CareerMatch Agent API", description=("Explainable CV-based job search, ranking and recommendation API."), version="0.1.0")

@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the current health status of the API."""
    return {"status": "healthy", "service": "career-match-agent", "version": "0.1.0"}

@app.post("/preferences", response_model=JobPreferences)
def validate_preferences(preferences: JobPreferences) -> JobPreferences:
    """Validate and normalise candidate job preferences."""
    return preferences
