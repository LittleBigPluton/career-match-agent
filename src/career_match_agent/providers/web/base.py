from dataclasses import dataclass
from typing import Protocol

from career_match_agent.models.job import JobPosting
from career_match_agent.models.web_job import (
    WebJobExtractionStrategy,
    WebJobSource
)


class WebJobParserError(ValueError):
    """Base error for web-job parsing."""

class UnsupportedWebJobSourceError(WebJobParserError):
    """Raised when no parser supports a supplied URL."""

class InvalidWebJobDocumentError(WebJobParserError):
    """Raised when the supplied HTML cannot be parsed."""

@dataclass(frozen=True)
class WebJobDocument:
    """One already-acquired job page."""
    source_url: str
    html: str
    content_sha256: str

@dataclass(frozen=True)
class ParsedWebJob:
    """Internal parser result."""
    job: JobPosting
    strategy: WebJobExtractionStrategy
    warnings: list[str]

class WebJobParser(Protocol):
    """Interface implemented by source-specific parsers."""
    source: WebJobSource
    parser_version: str

    def supports_url(self, url: str) -> bool:
        """Return whether this parser supports the URL."""

    def parse(self, document: WebJobDocument) -> ParsedWebJob:
        """Parse one saved web job page."""
