from enum import StrEnum

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


# Defines the accepted work-location arrangements.
# StrEnum allows enum values to behave like normal strings.
class WorkMode(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ON_SITE = "on_site"

# Defines the accepted employment contract types.
class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    INTERNSHIP = "internship"
    CONTRACT = "contract"

# Defines the accepted experience or seniority levels.
class SeniorityLevel(StrEnum):
    INTERNSHIP = "internship"
    ENTRY_LEVEL = "entry_level"
    JUNIOR = "junior"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"

# Stores the candidate's job-search preferences.
class JobPreferences(BaseModel):
    # At least one target role must be provided.
    roles: list[str] = Field(min_length=1, description="Preferred job titles or role families.")

    # An empty list means that no location restriction was provided.
    locations: list[str] = Field(default_factory=list, description="Preferred cities, regions or countries.")

    # Uses hybrid and on-site as the default accepted work modes.
    work_modes: list[WorkMode] = Field(default_factory=lambda: [WorkMode.HYBRID, WorkMode.ON_SITE])

    # Searches for full-time positions by default.
    employment_types: list[EmploymentType] = Field(default_factory=lambda: [EmploymentType.FULL_TIME])

    # Targets entry-level and junior jobs by default.
    seniority_levels: list[SeniorityLevel] = Field(default_factory=lambda: [SeniorityLevel.ENTRY_LEVEL, SeniorityLevel.JUNIOR])

    # Keywords that should appear in suitable job descriptions.
    required_keywords: list[str] = Field(default_factory=list)

    # Keywords that should cause a job to be excluded.
    excluded_keywords: list[str] = Field(default_factory=list)

    # Preferred languages for the job or workplace.
    preferred_languages: list[str] = Field(default_factory=list)

    # Limits the number of returned jobs to a value between 1 and 100.
    maximum_results: int = Field(default=20, ge=1, le=100)

    # Applies this validator to each listed string-list field.
    @field_validator("roles", "locations", "required_keywords", "excluded_keywords", "preferred_languages")

    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        """
        Clean a list of strings.

        Removes surrounding whitespace, ignores blank values,
        removes case-insensitive duplicates and preserves input order.
        """
        cleaned_values: list[str] = []
        seen_values: set[str] = set()
        for value in values:
            # Remove whitespace from the beginning and end.
            cleaned_value = value.strip()
            # Ignore empty strings such as "", " " or "\n".
            if not cleaned_value:
                continue

            # casefold() allows case-insensitive duplicate comparison.
            comparison_value = cleaned_value.casefold()
            # Keep only the first occurrence of each value.
            if comparison_value not in seen_values:
                seen_values.add(comparison_value)
                cleaned_values.append(cleaned_value)

        return cleaned_values

# Stores information extracted from the candidate's CV.
class CandidateProfile(BaseModel):
    # Optional candidate name. Defaults to None when unavailable.
    full_name: str | None = None

    # Optional summary extracted or generated from the CV.
    professional_summary: str | None = None

    # Candidate's technical and professional skills.
    skills: list[str] = Field(default_factory=list)

    # Current and previous job titles.
    job_titles: list[str] = Field(default_factory=list)

    # Optional experience duration; negative values are rejected.
    years_of_experience: float | None = Field(default=None, ge=0)

    # Education records, degrees or qualifications.
    education: list[str] = Field(default_factory=list)

    # Languages spoken by the candidate.
    languages: list[str] = Field(default_factory=list)

    # Required nested object containing job-search preferences.
    preferences: JobPreferences
