import hashlib
import re
import unicodedata
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser

from career_match_agent.models.candidate import EmploymentType
from career_match_agent.models.job import JobPosting


BLOCK_TAGS = {"br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ol", "p", "section", "table", "tr", "ul"}
IGNORED_TAGS = {"script", "style"}

class PlainTextHtmlParser(HTMLParser):
    """Extract readable text from a job description."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
            return

        if self._ignored_depth == 0 and tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return

        if self._ignored_depth == 0 and tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw_text = "".join(self._parts)
        cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.splitlines()]
        return "\n".join(line for line in cleaned_lines if line)

def html_to_plain_text(value: str) -> str:
    """Convert direct or escaped HTML into readable plain text."""
    parser = PlainTextHtmlParser()
    parser.feed(unescape(value))
    parser.close()
    return parser.get_text().strip()


def normalize_for_matching(value: str) -> str:
    """Create a normalized value for search and comparison."""
    normalized_value = unicodedata.normalize("NFKD", value)
    ascii_value = normalized_value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.casefold()).strip()


def normalize_employment_types(raw_values: list[str]) -> list[EmploymentType]:
    """Map provider-specific job types to common values."""
    normalized_types: list[EmploymentType] = []
    for raw_value in raw_values:
        normalized_value = normalize_for_matching(raw_value)
        employment_type: EmploymentType | None = None
        if normalized_value in {"full time", "fulltime", "permanent", "vollzeit"}:
            employment_type = EmploymentType.FULL_TIME

        elif normalized_value in {"part time", "parttime", "teilzeit"}:
            employment_type = EmploymentType.PART_TIME

        elif normalized_value in {"intern", "internship", "praktikum"}:
            employment_type = EmploymentType.INTERNSHIP

        elif normalized_value in {"contract", "contractor", "freelance", "freelancer", "temporary", "fixed term", "befristet"}:
            employment_type = EmploymentType.CONTRACT

        if (employment_type is not None and employment_type not in normalized_types):
            normalized_types.append(employment_type)

    return normalized_types


def create_job_fingerprint(*, title: str, company: str, location: str | None) -> str:
    """Create a provider-independent duplicate identifier."""
    canonical_value = "|".join([normalize_for_matching(title), normalize_for_matching(company), normalize_for_matching(location or "")])
    return hashlib.sha256(canonical_value.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class DeduplicationResult:
    """Result of removing duplicate job postings."""
    jobs: list[JobPosting]
    duplicate_count: int


def deduplicate_jobs(jobs: list[JobPosting]) -> DeduplicationResult:
    """Remove duplicate source IDs and fingerprints."""
    unique_jobs: list[JobPosting] = []
    seen_source_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    duplicate_count = 0

    for job in jobs:
        is_duplicate = (job.source_id in seen_source_ids or job.fingerprint in seen_fingerprints)
        if is_duplicate:
            duplicate_count += 1
            continue

        seen_source_ids.add(job.source_id)
        seen_fingerprints.add(job.fingerprint)
        unique_jobs.append(job)

    return DeduplicationResult(jobs=unique_jobs, duplicate_count=duplicate_count)
