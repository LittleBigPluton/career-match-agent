from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from career_match_agent.api.dependencies import get_profile_extractor
from career_match_agent.models.profile_extraction import (
    CandidateProfileExtractionResponse,
    ProfileExtractionMetadata,
)
from career_match_agent.services.pdf_extractor import (
    InvalidPdfError,
    NoExtractableTextError,
    PdfTooLargeError,
    UnsupportedPdfTypeError,
    extract_uploaded_pdf,
)
from career_match_agent.services.profile_extractor import (
    CandidateProfileExtractor,
    CvTextTooLongError,
    EmptyCvTextError,
    ProfileModelUnavailableError,
    ProfileResponseValidationError,
)


router = APIRouter(prefix="/profiles", tags=["profiles"])
@router.post("/candidate/extract", response_model=CandidateProfileExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_candidate_profile(
    file: Annotated[UploadFile,File(description="Candidate CV in PDF format.")],
    extractor: Annotated[CandidateProfileExtractor, Depends(get_profile_extractor)]) -> CandidateProfileExtractionResponse:
    """Extract a structured candidate profile from an uploaded CV."""
    try:
        pdf_extraction = await extract_uploaded_pdf(file)
        candidate_profile = await extractor.extract(pdf_extraction.text)
        return CandidateProfileExtractionResponse(
            document=pdf_extraction.to_metadata(),
            profile=candidate_profile,
            extraction=ProfileExtractionMetadata(provider=extractor.provider_name, model=extractor.model_name, prompt_version=extractor.prompt_version))

    except UnsupportedPdfTypeError as error:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error)) from error

    except PdfTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error

    except (InvalidPdfError, NoExtractableTextError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    except (EmptyCvTextError, CvTextTooLongError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    except ProfileModelUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    except ProfileResponseValidationError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    finally:
        await file.close()
