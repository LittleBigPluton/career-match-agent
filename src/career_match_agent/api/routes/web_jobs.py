from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status
)
from pydantic import ValidationError

from career_match_agent.api.dependencies import get_web_job_parser_registry
from career_match_agent.core.config import (
    Settings,
    get_settings
)
from career_match_agent.models.web_job import WebJobParseResponse
from career_match_agent.providers.web.base import (
    InvalidWebJobDocumentError,
    UnsupportedWebJobSourceError
)
from career_match_agent.providers.web.registry import WebJobParserRegistry
from career_match_agent.services.web_job_ingestion import (
    UnsupportedWebJobUploadError,
    WebJobUploadTooLargeError,
    ingest_web_job
)


router = APIRouter(prefix="/jobs/web", tags=["web-jobs"])


@router.post("/parse", response_model=WebJobParseResponse)
async def parse_web_job(file: Annotated[UploadFile, File(description=("Saved LinkedIn, Indeed, StepStone or Glassdoor HTML job page."))],
                        source_url: Annotated[str, Form(description=("Original job listing URL."))],
                        registry: Annotated[WebJobParserRegistry, Depends(get_web_job_parser_registry)],
                        settings: Annotated[Settings, Depends(get_settings)]) -> WebJobParseResponse:
    """Parse a saved job page into a normalized JobPosting."""
    try:
        return await ingest_web_job(upload=file, source_url=source_url, registry=registry, maximum_size_bytes=(settings.max_web_job_html_bytes))

    except UnsupportedWebJobUploadError as error:
        raise HTTPException(status_code=(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE), detail=str(error)) from error

    except WebJobUploadTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error

    except (InvalidWebJobDocumentError, UnsupportedWebJobSourceError, ValidationError) as error:
        raise HTTPException(status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT), detail=str(error)) from error

    finally:
        await file.close()
