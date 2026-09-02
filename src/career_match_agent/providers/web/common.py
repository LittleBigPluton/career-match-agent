import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from career_match_agent.models.candidate import EmploymentType
from career_match_agent.services.job_normalizer import (
    html_to_plain_text,
    normalize_employment_types
)


REMOTE_PATTERNS = (r"\bremote\b",
                   r"\bfully remote\b",
                   r"\bremote-first\b",
                   r"\bwork from home\b",
                   r"\bhome office\b",
                   r"\btelecommute\b")

EMPLOYMENT_TYPE_PATTERNS: tuple[tuple[str, str], ...] = ((r"\bfull[ -]?time\b", "Full Time"),
                                                         (r"\bvollzeit\b", "Vollzeit"),
                                                         (r"\bpart[ -]?time\b", "Part Time"),
                                                         (r"\bteilzeit\b", "Teilzeit"),
                                                         (r"\binternship\b", "Internship"),
                                                         (r"\bintern\b", "Internship"),
                                                         (r"\bpraktikum\b", "Praktikum"),
                                                         (r"\bcontract\b", "Contract"),
                                                         (r"\bfreelance\b", "Freelance"))


def normalize_hostname(url: str) -> str:
    """Return a lower-case hostname without a leading www."""
    hostname = (urlparse(url).hostname or "" ).casefold()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname

def hostname_matches(url: str, supported_domains: Iterable[str]) -> bool:
    """Match exact domains and their subdomains."""
    hostname = normalize_hostname(url)
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in supported_domains)

def stable_url_identifier(url: str) -> str:
    """Create a stable fallback external ID."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:24]

def select_first_text(soup: BeautifulSoup, selectors: Iterable[str]) -> str | None:
    """Return text from the first matching selector."""
    for selector in selectors:
        element = soup.select_one(selector)
        if element is None:
            continue

        text = element.get_text(" ", strip=True)
        if text:
            return text

    return None


def select_first_html_text(soup: BeautifulSoup, selectors: Iterable[str]) -> str | None:
    """Extract readable text from the first HTML container."""
    for selector in selectors:
        element = soup.select_one(selector)
        if element is None:
            continue

        text = html_to_plain_text(str(element))
        if text:
            return text

    return None


def extract_meta_content(soup: BeautifulSoup, *, property_name: str) -> str | None:
    """Read an OpenGraph-style meta value."""
    element = soup.find("meta", attrs={"property": property_name})
    if not isinstance(element, Tag):
        return None

    value = element.get("content")
    if not isinstance(value, str):
        return None

    cleaned_value = value.strip()
    return cleaned_value or None


def contains_jobposting_type(value: Any) -> bool:
    """Return whether a JSON-LD node is a JobPosting."""
    if isinstance(value, str):
        return value.casefold() == "jobposting"

    if isinstance(value, list):
        return any(contains_jobposting_type(item) for item in value)

    return False


def find_jobposting_nodes(value: Any) -> list[dict[str, Any]]:
    """Recursively find JobPosting nodes in JSON-LD."""
    matches: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if contains_jobposting_type(value.get("@type")):
            matches.append(value)

        for nested_value in value.values():
            matches.extend(find_jobposting_nodes(nested_value))

    elif isinstance(value, list):
        for item in value:
            matches.extend(find_jobposting_nodes(item))

    return matches


def extract_jobposting_jsonld(soup: BeautifulSoup) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract JobPosting JSON-LD nodes and parsing warnings."""
    jobs: list[dict[str, Any]] = []
    warnings: list[str] = []
    scripts = soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", flags=re.IGNORECASE)})
    for index, script in enumerate(scripts):
        script_text = script.string
        if script_text is None:
            script_text = script.get_text()

        script_text = (script_text.strip() if script_text else "")
        if not script_text:
            continue

        try:
            payload = json.loads(script_text)
        except json.JSONDecodeError:
            warnings.append(f"Ignored malformed JSON-LD block {index}.")
            continue

        jobs.extend(find_jobposting_nodes(payload))

    return jobs, warnings


def jsonld_text(value: Any) -> str | None:
    """Convert common JSON-LD values to readable text."""
    if isinstance(value, str):
        cleaned_value = value.strip()
        return cleaned_value or None

    if isinstance(value, list):
        values = [text for item in value if (text := jsonld_text(item))]
        return ", ".join(values) or None

    if isinstance(value, dict):
        for key in ("name", "value", "@value"):
            text = jsonld_text(value.get(key))
            if text:
                return text

    return None


def organization_name(value: Any) -> str | None:
    """Extract hiring organization name."""
    if isinstance(value, dict):
        return jsonld_text(value.get("name"))

    return jsonld_text(value)


def extract_location_part(value: Any) -> str | None:
    """Extract one readable Schema.org location."""
    if isinstance(value, str):
        return value.strip() or None

    if not isinstance(value, dict):
        return None

    address = value.get("address")
    if isinstance(address, str):
        return address.strip() or None

    if isinstance(address, dict):
        raw_components = [jsonld_text(address.get("addressLocality")), jsonld_text(address.get("addressRegion")), jsonld_text(address.get("addressCountry"))]
        components: list[str] = [component for component in raw_components if component is not None]
        if components:
            return ", ".join(components)

    return jsonld_text(value.get("name"))


def extract_jsonld_location(value: Any) -> str | None:
    """Extract one or more job locations."""
    if isinstance(value, list):
        locations = [location for item in value if (location := extract_location_part(item))]
        return "; ".join(locations) or None

    return extract_location_part(value)


def string_list(value: Any) -> list[str]:
    """Normalize JSON-LD string-or-list values."""
    if value is None:
        return []

    if isinstance(value, str):
        values = re.split(r"[,;|]", value)
        return [item.strip() for item in values if item.strip()]

    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            text = jsonld_text(item)

            if text:
                output.append(text)

        return output

    text = jsonld_text(value)
    return [text] if text else []


def parse_posted_datetime(value: Any) -> datetime | None:
    """Parse common Schema.org datePosted values."""
    if not isinstance(value, str):
        return None

    cleaned_value = value.strip()
    if not cleaned_value:
        return None

    try:
        parsed = datetime.fromisoformat(cleaned_value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(cleaned_value, "%Y-%m-%d")
        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed


def is_remote_jsonld(value: Any) -> bool:
    """Detect Schema.org TELECOMMUTE location type."""
    values = string_list(value)
    return any(item.casefold() == "telecommute" for item in values)


def detect_remote_from_text(text: str) -> bool | None:
    """Detect explicit remote wording without assuming on-site."""
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in REMOTE_PATTERNS):
        return True

    return None


def detect_raw_employment_types(text: str) -> list[str]:
    """Extract simple employment-type phrases from page text."""
    raw_types: list[str] = []
    for pattern, raw_type in (EMPLOYMENT_TYPE_PATTERNS):
        if re.search(pattern, text, flags=re.IGNORECASE):
            raw_types.append(raw_type)

    return list(dict.fromkeys(raw_types))


def normalize_web_employment_types(raw_values: list[str]) -> list[EmploymentType]:
    """Reuse CareerMatch's provider normalization."""
    return normalize_employment_types(raw_values)


def build_jsonld_description(payload: dict[str, Any]) -> str:
    """Combine useful structured text sections."""
    values: list[str] = []
    for field_name in ("description", "responsibilities", "qualifications", "experienceRequirements"):
        text = jsonld_text(payload.get(field_name))
        if not text:
            continue

        plain_text = html_to_plain_text(text)
        if (plain_text and plain_text not in values):
            values.append(plain_text)

    return "\n\n".join(values)
