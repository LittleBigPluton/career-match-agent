from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[str, Field(min_length=1)]
EvidenceText = Annotated[str, Field(min_length=1, max_length=300)]


def clean_string_list(values: list[str]) -> list[str]:
    """Strip values and remove case-insensitive duplicates."""
    cleaned_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        cleaned_value = value.strip()
        if not cleaned_value:
            continue

        comparison_value = cleaned_value.casefold()
        if comparison_value in seen_values:
            continue

        seen_values.add(comparison_value)
        cleaned_values.append(cleaned_value)
    return cleaned_values


class WorkMode(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ON_SITE = "on_site"

class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    INTERNSHIP = "internship"
    CONTRACT = "contract"

class SeniorityLevel(StrEnum):
    INTERNSHIP = "internship"
    ENTRY_LEVEL = "entry_level"
    JUNIOR = "junior"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"

class JobPreferences(BaseModel):
    """Candidate-supplied job-search preferences."""
    model_config = ConfigDict(extra="forbid")
    roles: list[str] = Field(min_length=1, description="Preferred job titles or role families.")
    locations: list[str] = Field(default_factory=list, description="Preferred cities, regions or countries.")
    work_modes: list[WorkMode] = Field(default_factory=lambda: [WorkMode.HYBRID, WorkMode.ON_SITE])
    employment_types: list[EmploymentType] = Field(default_factory=lambda: [EmploymentType.FULL_TIME])
    seniority_levels: list[SeniorityLevel] = Field(default_factory=lambda: [SeniorityLevel.ENTRY_LEVEL, SeniorityLevel.JUNIOR])
    required_keywords: list[str] = Field(default_factory=list)
    excluded_keywords: list[str] = Field(default_factory=list)
    preferred_languages: list[str] = Field(default_factory=list)
    maximum_results: int = Field(default=20, ge=1, le=100)

    @field_validator("roles", "locations", "required_keywords", "excluded_keywords", "preferred_languages")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)

class ProfileModel(BaseModel):
    """Base configuration for LLM-extracted profile objects."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class SkillEvidence(ProfileModel):
    """CV evidence supporting an extracted skill."""
    skill: NonEmptyText
    evidence: list[EvidenceText] = Field(default_factory=list)

class ExperienceEntry(ProfileModel):
    """A work or internship experience found in the CV."""
    job_title: NonEmptyText | None = None
    organization: NonEmptyText | None = None
    location: NonEmptyText | None = None
    start_date: NonEmptyText | None = None
    end_date: NonEmptyText | None = None
    is_current: bool = False
    highlights: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    evidence: list[EvidenceText] = Field(default_factory=list)

    @field_validator("highlights", "technologies")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)

    @model_validator(mode="after")
    def require_experience_identity(self) -> Self:
        if self.job_title is None and self.organization is None:
            raise ValueError("An experience must contain a job title or organization.")

        return self

class ProjectEntry(ProfileModel):
    """A technical, academic or personal project found in the CV."""
    name: NonEmptyText
    summary: NonEmptyText | None = None
    technologies: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    evidence: list[EvidenceText] = Field(default_factory=list)

    @field_validator("technologies", "highlights")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)

class EducationEntry(ProfileModel):
    """An education entry found in the CV."""
    degree: NonEmptyText | None = None
    field_of_study: NonEmptyText | None = None
    institution: NonEmptyText | None = None
    location: NonEmptyText | None = None
    start_date: NonEmptyText | None = None
    end_date: NonEmptyText | None = None
    details: list[str] = Field(default_factory=list)
    evidence: list[EvidenceText] = Field(default_factory=list)

    @field_validator("details")
    @classmethod
    def clean_details(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)

    @model_validator(mode="after")
    def require_education_identity(self) -> Self:
        if self.degree is None and self.institution is None:
            raise ValueError("An education entry must contain a degree or institution.")

        return self

class LanguageEntry(ProfileModel):
    """A language and its stated proficiency."""
    language: NonEmptyText
    proficiency: NonEmptyText | None = None

class CandidateProfile(ProfileModel):
    """Structured facts extracted from a candidate CV."""
    full_name: NonEmptyText | None = None
    location: NonEmptyText | None = None
    professional_summary: Annotated[str, Field(min_length=1, max_length=1000)] | None = None
    skills: list[str] = Field(default_factory=list)
    skill_evidence: list[SkillEvidence] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    languages: list[LanguageEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    years_of_experience: float | None = Field(default=None, ge=0)

    @field_validator("skills", "certifications")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)

    @model_validator(mode="after")
    def reject_empty_profile(self) -> Self:
        contains_information = any([self.full_name,
                                    self.location,
                                    self.professional_summary,
                                    self.skills,
                                    self.experience,
                                    self.projects,
                                    self.education,
                                    self.languages,
                                    self.certifications])

        if not contains_information:
            raise ValueError("The extracted candidate profile contains no information.")

        return self

class CandidateContext(ProfileModel):
    """Candidate facts combined with job-search preferences."""
    profile: CandidateProfile
    preferences: JobPreferences
