from dataclasses import dataclass
from typing import Any, Protocol

from bs4 import BeautifulSoup
from pydantic import HttpUrl, TypeAdapter

from career_match_agent.models.job import JobPosting
from career_match_agent.models.web_job import (
    WebJobExtractionStrategy,
    WebJobSource
)
from career_match_agent.providers.web.common import (
    build_jsonld_description,
    detect_raw_employment_types,
    detect_remote_from_text,
    extract_jobposting_jsonld,
    extract_jsonld_location,
    hostname_matches,
    is_remote_jsonld,
    jsonld_text,
    normalize_web_employment_types,
    organization_name,
    parse_posted_datetime,
    select_first_html_text,
    select_first_text,
    stable_url_identifier,
    string_list
)
from career_match_agent.services.job_normalizer import create_job_fingerprint

HTTP_URL_ADAPTER: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)

def validate_source_url(source_url: str) -> HttpUrl:
    """Validate a web-job source URL."""
    return HTTP_URL_ADAPTER.validate_python(source_url)

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

@dataclass(frozen=True)
class DomSelectorConfiguration:
    """Site-specific DOM fallback selectors."""
    title: tuple[str, ...]
    company: tuple[str, ...]
    location: tuple[str, ...]
    description: tuple[str, ...]

class BaseWebJobParser:
    """JSON-LD-first parser with site-specific DOM fallback."""
    source: WebJobSource
    parser_version: str
    supported_domains: tuple[str, ...]
    selectors: DomSelectorConfiguration

    def supports_url(self, url: str) -> bool:
        return hostname_matches(url, self.supported_domains)

    def extract_external_id(self, *, source_url: str, payload: dict[str, Any] | None) -> str:
        """Extract an external source ID or use URL fingerprint."""
        if payload is not None:
            identifier = payload.get("identifier")
            if isinstance(identifier, dict):
                identifier = (identifier.get("value") or identifier.get("name"))

            if isinstance(identifier, (str, int)):
                identifier_text = str(identifier).strip()

                if identifier_text:
                    return identifier_text

        return stable_url_identifier(source_url)

    def parse(self, document: WebJobDocument) -> ParsedWebJob:
        """Parse JSON-LD first, then fall back to DOM selectors."""
        soup = BeautifulSoup(document.html, "html.parser")
        jsonld_jobs, warnings = (extract_jobposting_jsonld(soup))
        for payload in jsonld_jobs:
            try:
                job = self._build_jsonld_job(document=document, payload=payload)
                return ParsedWebJob(job=job, strategy=(WebJobExtractionStrategy.JSON_LD), warnings=warnings)

            except InvalidWebJobDocumentError as error:
                warnings.append(f"JSON-LD candidate could not be normalized: {error}")

        job = self._build_dom_job(document=document, soup=soup)
        if jsonld_jobs:
            warnings.append("Fell back to DOM parsing after unusable JobPosting JSON-LD.")
        else:
            warnings.append("No usable JobPosting JSON-LD was found; DOM fallback was used.")

        return ParsedWebJob(job=job, strategy=(WebJobExtractionStrategy.DOM_FALLBACK), warnings=warnings,)

    def _build_jsonld_job(self, *, document: WebJobDocument, payload: dict[str, Any]) -> JobPosting:
        title = (jsonld_text(payload.get("title")) or jsonld_text(payload.get("name")))
        company = organization_name(payload.get("hiringOrganization"))
        description = (build_jsonld_description(payload))

        if not title:
            raise InvalidWebJobDocumentError("JSON-LD contains no job title.")

        if not company:
            raise InvalidWebJobDocumentError("JSON-LD contains no hiring organization.")

        if not description:
            raise InvalidWebJobDocumentError("JSON-LD contains no usable job description.")

        location = extract_jsonld_location(
            payload.get("jobLocation"))

        if location is None:
            location = jsonld_text(payload.get("applicantLocationRequirements"))

        raw_employment_types = (string_list(payload.get("employmentType")))
        tags = [*string_list(payload.get("skills")), *string_list(payload.get("occupationalCategory"))]
        tags = list(dict.fromkeys(tags))
        external_id = (self.extract_external_id(source_url=(document.source_url), payload=payload))
        validated_url = validate_source_url(document.source_url)
        return JobPosting(source_id=(f"{self.source.value}:{external_id}"),
                          provider=self.source.value,
                          external_id=external_id,
                          title=title,
                          company=company,
                          description=description,
                          location=location,
                          remote=(True if is_remote_jsonld(payload.get("jobLocationType")) else None),
                          visa_sponsorship=None,
                          employment_types=(normalize_web_employment_types(raw_employment_types)),
                          raw_employment_types=(raw_employment_types),
                          tags=tags,
                          url=validated_url,
                          posted_at=(parse_posted_datetime(payload.get("datePosted"))),
                          fingerprint=(create_job_fingerprint(title=title, company=company, location=location)))

    def _build_dom_job(self, *, document: WebJobDocument, soup: BeautifulSoup) -> JobPosting:
        title = select_first_text(soup, self.selectors.title)
        company = select_first_text(soup, self.selectors.company)
        location = select_first_text(soup, self.selectors.location)
        description = (select_first_html_text(soup, self.selectors.description))
        if not title:
            raise InvalidWebJobDocumentError("Could not locate the job title.")

        if not company:
            raise InvalidWebJobDocumentError("Could not locate the company name.")

        if not description:
            raise InvalidWebJobDocumentError("Could not locate the job description.")

        searchable_text = " ".join([title, location or "", description[:3000]])
        raw_employment_types = (detect_raw_employment_types(searchable_text))
        external_id = (self.extract_external_id(source_url=(document.source_url), payload=None))
        validated_url = validate_source_url(document.source_url)
        return JobPosting(source_id=(f"{self.source.value}:{external_id}"),
                          provider=self.source.value,
                          external_id=external_id,
                          title=title,
                          company=company,
                          description=description,
                          location=location,
                          remote=detect_remote_from_text(searchable_text),
                          visa_sponsorship=None,
                          employment_types=(normalize_web_employment_types(raw_employment_types)),
                          raw_employment_types=(raw_employment_types),
                          tags=[],
                          url=validated_url,
                          posted_at=None,
                          fingerprint=(create_job_fingerprint( title=title, company=company, location=location)))
