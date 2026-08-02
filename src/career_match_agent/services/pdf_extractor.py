import hashlib
from pathlib import Path

import pymupdf
from fastapi import UploadFile

from career_match_agent.models.document import PdfExtractionResponse

MAX_PDF_SIZE_BYTES = 5 * 1024 * 1024
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024

ALLOWED_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}

class PdfExtractionError(ValueError):
    """Base exception for PDF validation and extraction errors."""

class UnsupportedPdfTypeError(PdfExtractionError):
    """Raised when the uploaded file is not presented as a PDF."""

class PdfTooLargeError(PdfExtractionError):
    """Raised when an uploaded PDF exceeds the configured size limit."""

class InvalidPdfError(PdfExtractionError):
    """Raised when uploaded data cannot be parsed as a PDF."""

class NoExtractableTextError(PdfExtractionError):
    """Raised when a PDF contains no directly extractable text."""

def validate_pdf_metadata(filename: str | None, content_type: str | None) -> str:
    """Validate upload metadata and return a sanitised filename."""
    if not filename:
        raise UnsupportedPdfTypeError("The uploaded file must have a filename.")

    normalised_filename = filename.replace("\\", "/")
    safe_filename = Path(normalised_filename).name

    if Path(safe_filename).suffix.casefold() != ".pdf":
        raise UnsupportedPdfTypeError("Only files with a .pdf extension are supported.")

    if (content_type is not None and content_type.casefold() not in ALLOWED_PDF_CONTENT_TYPES):
        raise UnsupportedPdfTypeError(f"Unsupported content type: {content_type}.")

    return safe_filename

async def read_upload_bytes(upload: UploadFile, max_size_bytes: int = MAX_PDF_SIZE_BYTES) -> bytes:
    """Read an uploaded file while enforcing a maximum size."""
    contents = bytearray()
    while chunk := await upload.read(UPLOAD_CHUNK_SIZE_BYTES):
        contents.extend(chunk)
        if len(contents) > max_size_bytes:
            maximum_megabytes = max_size_bytes / (1024 * 1024)
            raise PdfTooLargeError(f"The uploaded PDF exceeds the {maximum_megabytes:.0f} MB limit.")

    if not contents:
        raise InvalidPdfError("The uploaded PDF is empty.")

    return bytes(contents)

def extract_pdf(data: bytes, *, filename: str, content_type: str | None) -> PdfExtractionResponse:
    """Extract plain text and metadata from PDF bytes."""
    if not data:
        raise InvalidPdfError("The uploaded PDF is empty.")

    if b"%PDF-" not in data[:1024]:
        raise InvalidPdfError("The uploaded file does not contain a valid PDF header.")

    try:
        with pymupdf.open(stream=data, filetype="pdf") as document: # type: ignore[no-untyped-call]
            if document.page_count < 1:
                raise InvalidPdfError("The PDF does not contain any pages.")

            page_texts: list[str] = []
            for page in document:
                page_text = page.get_text("text", sort=True).strip()
                if page_text:
                    page_texts.append(page_text)

            page_count = document.page_count

    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        raise InvalidPdfError("The uploaded file could not be parsed as a valid PDF.") from error

    extracted_text = "\n\n".join(page_texts).strip()
    if not extracted_text:
        raise NoExtractableTextError("No selectable text was found in the PDF. \n Scanned or image-only PDFs are not supported yet.")

    return PdfExtractionResponse(filename=filename,
                                 content_type=content_type,
                                 size_bytes=len(data),
                                 page_count=page_count,
                                 character_count=len(extracted_text),
                                 word_count=len(extracted_text.split()),
                                 sha256=hashlib.sha256(data).hexdigest(),
                                 text=extracted_text)

async def extract_uploaded_pdf(upload: UploadFile) -> PdfExtractionResponse:
    """Validate and extract an uploaded PDF."""
    safe_filename = validate_pdf_metadata(filename=upload.filename, content_type=upload.content_type)
    pdf_bytes = await read_upload_bytes(upload)
    return extract_pdf(pdf_bytes, filename=safe_filename, content_type=upload.content_type)
