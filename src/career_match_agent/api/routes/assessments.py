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

from career_match_agent.core.config import (
    Settings,
    get_settings
)
from career_match_agent.models.hiring_agent import (
    CandidateEvidenceContext,
    CandidateEvidenceContextRequest,
    HiringAgentAssessment
)
from career_match_agent.services.candidate_enrichment import (
    build_candidate_evidence_context
)
from career_match_agent.services.hiring_agent_parser import (
    HiringAgentReportTooLargeError,
    InvalidHiringAgentReportError,
    UnsupportedHiringAgentReportError,
    parse_hiring_agent_report,
    read_hiring_agent_report_bytes,
    validate_hiring_agent_report_metadata
)


router = APIRouter(prefix="/assessments", tags=["assessments"])

@router.post("/hiring-agent/parse", response_model=HiringAgentAssessment)
async def parse_hiring_agent_assessment(report: Annotated[UploadFile, File(description=("Hiring-agent report in text, log or JSON format."))],
                                        settings: Annotated[Settings, Depends(get_settings)], role_name: Annotated[str | None,
                                        Form(description=("Name of the hiring-agent role used to create the report."))] = None) -> HiringAgentAssessment:
    """Parse and normalize a hiring-agent report."""
    try:
        safe_filename = (
            validate_hiring_agent_report_metadata(filename=report.filename, content_type=report.content_type))

        report_bytes = (await read_hiring_agent_report_bytes(report, maximum_size_bytes=(settings.max_hiring_agent_report_bytes)))
        return parse_hiring_agent_report(report_bytes, source_filename=safe_filename, role_name=role_name)

    except UnsupportedHiringAgentReportError as error:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error)) from error

    except HiringAgentReportTooLargeError as error:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(error)) from error

    except InvalidHiringAgentReportError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    finally:
        await report.close()


@router.post("/hiring-agent/context", response_model=CandidateEvidenceContext)
def create_candidate_evidence_context(request: CandidateEvidenceContextRequest) -> CandidateEvidenceContext:
    """Combine a candidate profile and hiring-agent assessment."""
    return build_candidate_evidence_context(profile=request.profile, assessment=request.assessment)
