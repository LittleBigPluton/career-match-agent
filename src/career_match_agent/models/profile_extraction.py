from pydantic import BaseModel, Field

from career_match_agent.models.candidate import CandidateProfile
from career_match_agent.models.document import PdfDocumentMetadata


class ProfileExtractionMetadata(BaseModel):
    """Information about the LLM extraction process."""
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)


class CandidateProfileExtractionResponse(BaseModel):
    """Candidate profile generated from an uploaded CV."""
    document: PdfDocumentMetadata
    profile: CandidateProfile
    extraction: ProfileExtractionMetadata
