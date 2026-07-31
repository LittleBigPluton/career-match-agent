from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from career_match_agent.models.document import PdfExtractionResponse
from career_match_agent.services.pdf_extractor import (
    InvalidPdfError,
    NoExtractableTextError,
    PdfTooLargeError,
    UnsupportedPdfTypeError,
    extract_pdf,
    read_upload_bytes,
    validate_pdf_metadata,
)

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/cv/extract", response_model=PdfExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_cv(file: Annotated[UploadFile, File(description="Candidate CV in PDF format.")]) -> PdfExtractionResponse:
    """Validate an uploaded CV and extract its text."""
    try:
        safe_filename = validate_pdf_metadata(filename=file.filename, content_type=file.content_type)
        pdf_bytes = await read_upload_bytes(file)
        return extract_pdf(pdf_bytes, filename=safe_filename, content_type=file.content_type)

    except UnsupportedPdfTypeError as error:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error)) from error

    except PdfTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error

    except (InvalidPdfError, NoExtractableTextError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    finally:
        await file.close()
