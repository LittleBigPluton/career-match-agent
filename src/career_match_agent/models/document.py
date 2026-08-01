from pydantic import BaseModel, Field


class PdfExtractionResponse(BaseModel):
    """Metadata and text extracted from an uploaded PDF."""
    filename: str
    content_type: str | None = None
    size_bytes: int = Field(ge=1)
    page_count: int = Field(ge=1)
    character_count: int = Field(ge=1)
    word_count: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=64) # Not an encryption method and should not be treated as one.

class PdfExtractionResponse(PdfDocumentMetadata):
    """Metadata and plain text extracted from an uploaded PDF."""
    text: str = Field(min_length=1)

    def to_metadata(self) -> PdfDocumentMetadata:
        """Return the document information without extracted text."""
        return PdfDocumentMetadata.model_validate(self.model_dump(exclude={"text"}))
