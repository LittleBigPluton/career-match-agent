from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from career_match_agent.models.candidate import CandidateProfile


def clean_strings(values: list[str]) -> list[str]:
    """Strip empty strings and remove case-insensitive duplicates."""
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


class HiringAgentModel(BaseModel):
    """Base configuration for hiring-agent integration models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class HiringAgentReportFormat(StrEnum):
    TEXT = "text"
    JSON = "json"

class HiringAgentCategoryResult(HiringAgentModel):
    """One dynamically defined hiring-agent scoring category."""
    key: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    label: str = Field(min_length=1)
    score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    evidence: str = Field(min_length=1)
    capped_score: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def calculate_capped_score(self) -> Self:
        self.capped_score = min(self.score, self.max_score)
        return self


class HiringAgentBonus(HiringAgentModel):
    """Bonus points reported by hiring-agent."""
    total: float = Field(default=0, ge=0)
    breakdown: str = ""

class HiringAgentDeductions(HiringAgentModel):
    """Deductions reported by hiring-agent."""
    total: float = Field(default=0, ge=0)
    reasons: str = ""

class HiringAgentAssessment(HiringAgentModel):
    """Normalized evaluation imported from hiring-agent."""
    candidate_name: str | None = None
    role_name: str | None = None
    reported_overall_score: float
    base_max_score: float = Field(gt=0)
    categories: list[HiringAgentCategoryResult] = Field(min_length=1)
    bonus_points: HiringAgentBonus = Field(default_factory=HiringAgentBonus)
    deductions: HiringAgentDeductions = Field(default_factory=HiringAgentDeductions)
    key_strengths: list[str] = Field(default_factory=list)
    areas_for_improvement: list[str] = Field(default_factory=list)
    source_format: HiringAgentReportFormat
    source_filename: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    category_total: float = 0
    computed_overall_score: float = 0
    score_difference: float = 0
    warnings: list[str] = Field(default_factory=list)

    @field_validator("key_strengths", "areas_for_improvement", "warnings")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        return clean_strings(values)

    @model_validator(mode="after")
    def calculate_and_validate_totals(self) -> Self:
        category_keys = [category.key for category in self.categories]
        if len(category_keys) != len(set(category_keys)):
            raise ValueError("Hiring-agent category keys must be unique.")

        category_total = sum(category.capped_score for category in self.categories)
        calculated_base_max = sum(category.max_score for category in self.categories)
        computed_overall = (category_total + self.bonus_points.total - self.deductions.total)
        self.category_total = round(category_total, 2)
        self.computed_overall_score = round(computed_overall, 2)
        self.score_difference = round(self.reported_overall_score - computed_overall, 2)
        generated_warnings = list(self.warnings)

        if abs(self.base_max_score - calculated_base_max) > 0.01:
            generated_warnings.append("The reported base maximum differs from the sum of category maximums.")

        if abs(self.score_difference) > 0.1:
            generated_warnings.append("The reported overall score differs from the score recomputed from categories, bonuses and deductions.")

        for category in self.categories:
            if category.score > category.max_score:
                generated_warnings.append(f"The source score for '{category.label}' exceeded its maximum and was capped.")

        self.warnings = clean_strings(generated_warnings)
        return self


class EvidenceSignalType(StrEnum):
    CATEGORY = "category"
    STRENGTH = "strength"
    IMPROVEMENT = "improvement"
    BONUS = "bonus"
    DEDUCTION = "deduction"

class EvidencePolarity(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    DEVELOPMENT = "development"

class CandidateEvidenceSignal(HiringAgentModel):
    """One reusable evidence signal derived from an assessment."""
    signal_type: EvidenceSignalType
    title: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    polarity: EvidencePolarity
    source: str = "hiring-agent"
    category_key: str | None = None
    source_score: float | None = None
    source_max_score: float | None = None

class CandidateEvidenceContextRequest(HiringAgentModel):
    """Request for combining a candidate profile with an assessment."""
    profile: CandidateProfile
    assessment: HiringAgentAssessment


class CandidateEvidenceContext(HiringAgentModel):
    """Candidate facts and external assessment evidence."""
    profile: CandidateProfile
    assessments: list[HiringAgentAssessment]
    evidence_signals: list[CandidateEvidenceSignal]
