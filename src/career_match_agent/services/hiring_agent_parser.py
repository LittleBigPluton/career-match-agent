import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastapi import UploadFile
from pydantic import ValidationError

from career_match_agent.models.hiring_agent import (
    HiringAgentAssessment,
    HiringAgentBonus,
    HiringAgentCategoryResult,
    HiringAgentDeductions,
    HiringAgentReportFormat
)


HIRING_AGENT_PARSER_VERSION = "hiring-agent-adapter-v1"
REPORT_CHUNK_SIZE_BYTES = 256 * 1024
ALLOWED_REPORT_EXTENSIONS = {".txt", ".log", ".json"}
ALLOWED_REPORT_CONTENT_TYPES = {"text/plain", "application/json", "application/octet-stream"}
NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"
CANDIDATE_PATTERN = re.compile(r"RESUME EVALUATION RESULTS FOR:\s*(?P<name>.+)$", flags=re.IGNORECASE)
OVERALL_PATTERN = re.compile(rf"OVERALL SCORE:\s*"
                             rf"(?P<score>{NUMBER_PATTERN})\s*/\s*"
                             rf"(?P<maximum>{NUMBER_PATTERN})", flags=re.IGNORECASE)

CATEGORY_PATTERN = re.compile(rf"^(?P<label>.+?):\s*"
                              rf"(?P<score>\d+(?:\.\d+)?)\s*/\s*"
                              rf"(?P<maximum>\d+(?:\.\d+)?)$")

LIST_ITEM_PATTERN = re.compile(r"^(?:\d+\.\s*|[-•]\s*)")

class HiringAgentReportError(ValueError):
    """Base exception for hiring-agent report errors."""

class UnsupportedHiringAgentReportError(HiringAgentReportError):
    """Raised when the uploaded report type is unsupported."""

class HiringAgentReportTooLargeError(HiringAgentReportError):
    """Raised when the uploaded report exceeds its size limit."""

class InvalidHiringAgentReportError(HiringAgentReportError):
    """Raised when a report cannot be parsed or validated."""

@dataclass
class PendingCategory:
    """Temporary category while parsing a text report."""
    label: str
    score: float
    max_score: float
    evidence: str | None = None

def validate_hiring_agent_report_metadata(filename: str | None, content_type: str | None,) -> str:
    """Validate report metadata and return a safe filename."""
    if not filename:
        raise UnsupportedHiringAgentReportError("The hiring-agent report must have a filename.")

    normalized_filename = filename.replace("\\", "/")
    safe_filename = Path(normalized_filename).name
    extension = Path(safe_filename).suffix.casefold()
    if extension not in ALLOWED_REPORT_EXTENSIONS:
        raise UnsupportedHiringAgentReportError("Only .txt, .log and .json hiring-agent reports are supported.")

    if (content_type is not None and content_type.casefold() not in ALLOWED_REPORT_CONTENT_TYPES):
        raise UnsupportedHiringAgentReportError(f"Unsupported report content type: {content_type}.")

    return safe_filename


async def read_hiring_agent_report_bytes(upload: UploadFile, *, maximum_size_bytes: int) -> bytes:
    """Read a hiring-agent report with a strict size limit."""
    contents = bytearray()
    while chunk := await upload.read(REPORT_CHUNK_SIZE_BYTES):
        contents.extend(chunk)
        if len(contents) > maximum_size_bytes:
            raise HiringAgentReportTooLargeError(f"The uploaded hiring-agent report exceeds the {maximum_size_bytes}-byte limit.")

    if not contents:
        raise InvalidHiringAgentReportError("The uploaded hiring-agent report is empty.")

    return bytes(contents)


def slugify_category_key(value: str) -> str:
    """Convert a category label or key to snake_case."""
    normalized_value = unicodedata.normalize("NFKD", value)
    ascii_value = normalized_value.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_value.casefold()).strip("_")
    if not slug:
        raise InvalidHiringAgentReportError(f"Could not generate a category key from: {value!r}.")

    return slug

def category_label_from_key(key: str) -> str:
    """Create a readable category label from a JSON key."""
    return key.replace("_", " ").strip().title()

def clean_heading_label(value: str) -> str:
    """Remove leading terminal icons from a category label."""
    cleaned_value = re.sub(r"^[^\w]+", "", value, flags=re.UNICODE).strip()
    if not cleaned_value:
        raise InvalidHiringAgentReportError("A hiring-agent category has an empty label.")

    return cleaned_value


def clean_list_item(value: str) -> str:
    """Remove numbering or bullet characters from a report item."""
    return LIST_ITEM_PATTERN.sub("", value).strip()

def decode_report(data: bytes) -> str:
    """Decode report bytes as UTF-8 text."""
    try:
        report_text = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise InvalidHiringAgentReportError("The hiring-agent report must use UTF-8 encoding.") from error

    if not report_text:
        raise InvalidHiringAgentReportError("The hiring-agent report contains no text.")

    return report_text

def require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Validate a dynamically loaded JSON object."""
    if not isinstance(value, dict):
        raise InvalidHiringAgentReportError(f"'{field_name}' must be a JSON object.")

    return cast(dict[str, Any], value)

def require_list(value: Any, *, field_name: str) -> list[Any]:
    """Validate a dynamically loaded JSON array."""
    if not isinstance(value, list):
        raise InvalidHiringAgentReportError(f"'{field_name}' must be a JSON array.")

    return value

def string_list_from_json(value: Any, *, field_name: str) -> list[str]:
    """Validate and normalize a list of strings."""
    if value is None:
        return []

    raw_values = require_list(value, field_name=field_name)
    if not all(isinstance(item, str) for item in raw_values):
        raise InvalidHiringAgentReportError(f"'{field_name}' must contain only strings.")

    return [item.strip() for item in raw_values if isinstance(item, str) and item.strip()]


def parse_hiring_agent_json(report_text: str, *, source_filename: str, role_name: str | None) -> HiringAgentAssessment:
    """Parse hiring-agent EvaluationData-style JSON."""
    try:
        loaded_value: Any = json.loads(report_text)
    except json.JSONDecodeError as error:
        raise InvalidHiringAgentReportError("The uploaded hiring-agent JSON is invalid.") from error

    payload = require_mapping(loaded_value, field_name="report")
    raw_scores = require_mapping(payload.get("scores"), field_name="scores")
    if not raw_scores:
        raise InvalidHiringAgentReportError("The hiring-agent JSON contains no score categories.")

    categories: list[HiringAgentCategoryResult] = []
    for raw_key, raw_category_value in raw_scores.items():
        raw_category = require_mapping(raw_category_value, field_name=f"scores.{raw_key}")
        raw_max_score = raw_category.get("max")
        if raw_max_score is None:
            raw_max_score = raw_category.get("max_score")
            if raw_max_score is None:
                raise InvalidHiringAgentReportError(f"Score category '{raw_key}' does not define 'max' or 'max_score'.")

        try:
            score = float(raw_category["score"])
            max_score = float(raw_max_score)
            evidence = str(raw_category["evidence"]).strip()

        except (KeyError, TypeError, ValueError) as error:
            raise InvalidHiringAgentReportError(f"Invalid score category: {raw_key}.") from error

        categories.append(HiringAgentCategoryResult(key=slugify_category_key(raw_key), label=category_label_from_key(raw_key),
                          score=score, max_score=max_score, evidence=evidence))

    raw_bonus = require_mapping(payload.get("bonus_points",{"total": 0, "breakdown": ""}), field_name="bonus_points")
    raw_deductions = require_mapping(payload.get("deductions",{"total": 0, "reasons": ""}), field_name="deductions")
    try:
        bonus = HiringAgentBonus(total=float(raw_bonus.get("total", 0)),breakdown=str(raw_bonus.get("breakdown", "")))
        deductions = HiringAgentDeductions(total=abs(float(raw_deductions.get("total", 0))),reasons=str(raw_deductions.get("reasons", "")))

    except (TypeError, ValueError, ValidationError) as error:
        raise InvalidHiringAgentReportError("The hiring-agent bonus or deduction data is invalid.") from error

    category_total = sum(category.capped_score for category in categories)
    computed_overall = (category_total + bonus.total - deductions.total)
    reported_score_value = payload.get("overall_score", computed_overall)

    try:
        reported_overall_score = float(reported_score_value)
    except (TypeError, ValueError) as error:
        raise InvalidHiringAgentReportError("The hiring-agent overall score is invalid.") from error

    try:
        return HiringAgentAssessment(candidate_name=(str(payload["candidate_name"]).strip() if payload.get("candidate_name") else None),
                                     role_name=role_name,
                                     reported_overall_score=reported_overall_score,
                                     base_max_score=sum(category.max_score for category in categories),
                                     categories=categories,
                                     bonus_points=bonus,
                                     deductions=deductions,
                                     key_strengths=string_list_from_json(payload.get("key_strengths"), field_name="key_strengths"),
                                     areas_for_improvement=string_list_from_json(payload.get("areas_for_improvement"), field_name="areas_for_improvement"),
                                     source_format=HiringAgentReportFormat.JSON,
                                     source_filename=source_filename,
                                     parser_version=HIRING_AGENT_PARSER_VERSION)

    except ValidationError as error:
        raise InvalidHiringAgentReportError("The hiring-agent JSON failed validation.") from error

def parse_hiring_agent_text(report_text: str, *, source_filename: str, role_name: str | None) -> HiringAgentAssessment:
    """Parse the human-readable hiring-agent terminal report."""
    candidate_name: str | None = None
    reported_score: float | None = None
    base_max_score: float | None = None
    categories: list[HiringAgentCategoryResult] = []
    key_strengths: list[str] = []
    areas_for_improvement: list[str] = []
    bonus_lines: list[str] = []
    deduction_lines: list[str] = []
    bonus_total = 0.0
    deduction_total = 0.0
    current_section: str | None = None
    pending_category: PendingCategory | None = None

    def flush_pending_category() -> None:
        nonlocal pending_category
        if pending_category is None:
            return

        if not pending_category.evidence:
            raise InvalidHiringAgentReportError(f"Category '{pending_category.label}' does not contain evidence.")

        categories.append(HiringAgentCategoryResult(key=slugify_category_key(pending_category.label), label=pending_category.label, score=pending_category.score,
                                                    max_score=pending_category.max_score, evidence=pending_category.evidence))
        pending_category = None

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if set(line) <= {"-", "="}:
            continue

        candidate_match = CANDIDATE_PATTERN.search(line)
        if candidate_match:
            candidate_name = (candidate_match.group("name").strip())
            continue

        overall_match = OVERALL_PATTERN.search(line)
        if overall_match:
            reported_score = float(overall_match.group("score"))
            base_max_score = float(overall_match.group("maximum"))
            continue

        uppercase_line = line.upper()
        if "DETAILED SCORES:" in uppercase_line:
            flush_pending_category()
            current_section = "categories"
            continue

        if "BONUS POINTS:" in uppercase_line:
            flush_pending_category()
            current_section = "bonus"
            number_match = re.search(NUMBER_PATTERN, line.split(":", maxsplit=1)[1])
            if number_match:
                bonus_total = float(number_match.group())
            continue

        if "DEDUCTIONS:" in uppercase_line:
            flush_pending_category()
            current_section = "deductions"
            number_match = re.search(NUMBER_PATTERN, line.split(":", maxsplit=1)[1])
            if number_match:
                deduction_total = abs(float(number_match.group()))
            continue

        if "KEY STRENGTHS:" in uppercase_line:
            flush_pending_category()
            current_section = "strengths"
            continue

        if "AREAS FOR IMPROVEMENT:" in uppercase_line:
            flush_pending_category()
            current_section = "improvements"
            continue

        if current_section == "categories":
            if line.casefold().startswith("evidence:"):
                if pending_category is None:
                    raise InvalidHiringAgentReportError("Category evidence appeared before a category score.")

                pending_category.evidence = line.split(":", maxsplit=1)[1].strip()
                continue

            category_line = clean_heading_label(line)
            category_match = CATEGORY_PATTERN.match(category_line)
            if category_match:
                flush_pending_category()
                pending_category = PendingCategory(
                    label=clean_heading_label(category_match.group("label")),
                    score=float(category_match.group("score")),
                    max_score=float(category_match.group("maximum")))

            continue

        if current_section == "bonus":
            bonus_lines.append(clean_list_item(line))
            continue

        if current_section == "deductions":
            deduction_lines.append(clean_list_item(line))
            continue

        if current_section == "strengths":
            cleaned_item = clean_list_item(line)
            if cleaned_item:
                key_strengths.append(cleaned_item)

            continue

        if current_section == "improvements":
            cleaned_item = clean_list_item(line)
            if cleaned_item:
                areas_for_improvement.append(cleaned_item)

    flush_pending_category()
    if reported_score is None or base_max_score is None:
        raise InvalidHiringAgentReportError("The hiring-agent report does not contain a valid overall score.")

    if not categories:
        raise InvalidHiringAgentReportError("The hiring-agent report does not contain any category scores.")

    try:
        return HiringAgentAssessment(candidate_name=candidate_name,
                                     role_name=role_name,
                                     reported_overall_score=reported_score,
                                     base_max_score=base_max_score,
                                     categories=categories,
                                     bonus_points=HiringAgentBonus(total=bonus_total, breakdown=" ".join(bonus_lines).strip()),
                                     deductions=HiringAgentDeductions(total=deduction_total, reasons=" ".join(deduction_lines).strip()),
                                     key_strengths=key_strengths,
                                     areas_for_improvement=areas_for_improvement,
                                     source_format=HiringAgentReportFormat.TEXT,
                                     source_filename=source_filename,
                                     parser_version=HIRING_AGENT_PARSER_VERSION)

    except ValidationError as error:
        raise InvalidHiringAgentReportError("The hiring-agent text report failed validation.") from error


def parse_hiring_agent_report(data: bytes, *, source_filename: str, role_name: str | None = None) -> HiringAgentAssessment:
    """Detect the report format and return a normalized assessment."""
    report_text = decode_report(data)
    extension = Path(source_filename).suffix.casefold()
    if extension == ".json" or report_text.lstrip().startswith("{"):
        return parse_hiring_agent_json(report_text, source_filename=source_filename, role_name=role_name)

    return parse_hiring_agent_text(report_text, source_filename=source_filename, role_name=role_name)
