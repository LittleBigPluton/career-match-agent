import hashlib
from pathlib import Path

from bs4 import UnicodeDammit
from fastapi import UploadFile
from pydantic import HttpUrl, TypeAdapter

from career_match_agent.models.web_job import (
    WebJobParseMetadata,
    WebJobParseResponse
)
from career_match_agent.providers.web.base import (
    InvalidWebJobDocumentError,
    WebJobDocument
)
from career_match_agent.providers.web.registry import WebJobParserRegistry


HTML_CHUNK_SIZE_BYTES = 1024 * 1024
ALLOWED_HTML_EXTENSIONS = {".html", ".htm"}
ALLOWED_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain", "application/octet-stream"}
HTTP_URL_ADAPTER: TypeAdapter[HttpUrl] = (TypeAdapter(HttpUrl))

class WebJobUploadError(ValueError):
    """Base error for uploaded web-job documents."""

class UnsupportedWebJobUploadError(WebJobUploadError):
    """Raised when the uploaded file type is unsupported."""

class WebJobUploadTooLargeError(WebJobUploadError):
    """Raised when uploaded HTML exceeds its size limit."""


def validate_web_job_upload(*, filename: str | None, content_type: str | None) -> str:
    """Validate the uploaded saved-page metadata."""
    if not filename:
        raise UnsupportedWebJobUploadError("The uploaded page must have a filename.")

    normalized_filename = filename.replace("\\", "/")
    safe_filename = Path(normalized_filename).name
    extension = Path(safe_filename).suffix.casefold()
    if extension not in (ALLOWED_HTML_EXTENSIONS):
        raise UnsupportedWebJobUploadError("Only .html and .htm files are supported.")

    if (content_type is not None and content_type.casefold() not in ALLOWED_HTML_CONTENT_TYPES):
        raise UnsupportedWebJobUploadError(f"Unsupported HTML content type: {content_type}.")

    return safe_filename


async def read_web_job_html(upload: UploadFile, *, maximum_size_bytes: int) -> bytes:
    """Read uploaded HTML while enforcing a maximum size."""
    contents = bytearray()
    while chunk := await upload.read(HTML_CHUNK_SIZE_BYTES):
        contents.extend(chunk)
        if len(contents) > maximum_size_bytes:
            raise WebJobUploadTooLargeError(f"The uploaded HTML exceeds the {maximum_size_bytes}-byte limit.")

    if not contents:
        raise InvalidWebJobDocumentError("The uploaded HTML file is empty.")

    return bytes(contents)


def decode_html(data: bytes) -> str:
    """Decode saved HTML using Beautiful Soup's charset detection."""
    decoded = UnicodeDammit(data).unicode_markup
    if decoded is None:
        raise InvalidWebJobDocumentError("The uploaded job page could not be decoded.")

    decoded = decoded.strip()
    if not decoded:
        raise InvalidWebJobDocumentError("The uploaded job page contains no HTML.")

    return decoded


async def ingest_web_job(*, upload: UploadFile, source_url: str, registry: WebJobParserRegistry, maximum_size_bytes: int) -> WebJobParseResponse:
    """Parse one user-supplied web job page."""
    validate_web_job_upload(filename=upload.filename, content_type=upload.content_type)
    validated_url = HTTP_URL_ADAPTER.validate_python(source_url)
    html_bytes = await read_web_job_html(upload, maximum_size_bytes=(maximum_size_bytes))
    html = decode_html(html_bytes)
    content_sha256 = hashlib.sha256(html_bytes).hexdigest()
    document = WebJobDocument(source_url=str(validated_url), html=html, content_sha256=content_sha256)
    parser, parsed = registry.parse(document)
    return WebJobParseResponse(job=parsed.job, metadata=WebJobParseMetadata(source=parser.source,
                                                                            source_url=validated_url,
                                                                            extraction_strategy=(parsed.strategy),
                                                                            parser_version=(parser.parser_version),
                                                                            content_sha256=(content_sha256),
                                                                            warnings=parsed.warnings))
