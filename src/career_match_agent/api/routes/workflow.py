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
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from career_match_agent.api.dependencies import (
    get_embedding_provider,
    get_workflow_runtime_factory
)
from career_match_agent.core.config import (
    Settings,
    get_settings
)
from career_match_agent.models.workflow import (
    AutomatedWorkflowResponse,
    ProviderCapability,
    WorkflowCapabilities,
    WorkflowOptions
)
from career_match_agent.providers.base import (
    JobProvider,
    JobProviderResponseError,
    JobProviderUnavailableError
)
from career_match_agent.providers.llm.base import (
    LLMProviderConfigurationError,
    LLMProviderResponseError,
    LLMProviderUnavailableError
)
from career_match_agent.services.embedding import (
    EmbeddingModelUnavailableError,
    EmbeddingProvider,
    InvalidEmbeddingResponseError
)
from career_match_agent.services.hiring_agent_parser import (
    HiringAgentReportTooLargeError,
    InvalidHiringAgentReportError,
    UnsupportedHiringAgentReportError,
    parse_hiring_agent_report,
    read_hiring_agent_report_bytes,
    validate_hiring_agent_report_metadata
)
from career_match_agent.services.job_evaluator import StructuredJobReportGenerator
from career_match_agent.services.pdf_extractor import (
    InvalidPdfError,
    NoExtractableTextError,
    PdfTooLargeError,
    UnsupportedPdfTypeError,
    extract_uploaded_pdf
)
from career_match_agent.services.preference_extractor import (
    EmptyPreferenceTextError,
    PreferenceTextTooLongError,
    StructuredPreferenceExtractor
)
from career_match_agent.services.profile_extractor import StructuredCandidateProfileExtractor
from career_match_agent.services.search_planner import StructuredSearchPlanner
from career_match_agent.services.workflow_factory import WorkflowRuntimeFactory
from career_match_agent.services.workflow_runner import (
    AutomatedCareerMatchWorkflow,
    AutomatedWorkflowDependencies
)


router = APIRouter(prefix="/workflow", tags=["workflow"])

@router.post("/run", response_model=AutomatedWorkflowResponse)
async def run_automated_workflow(cv: Annotated[UploadFile, File(description="Candidate CV in PDF format.")],
                                 preference_text: Annotated[str, Form(description="Natural-language job preferences.")],
                                 options_json: Annotated[str, Form(description="Serialized WorkflowOptions.")],
                                 runtime_factory: Annotated[WorkflowRuntimeFactory, Depends(get_workflow_runtime_factory)],
                                 embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
                                 settings: Annotated[Settings, Depends(get_settings)],
                                 hiring_report: Annotated[UploadFile | None, File(description=("Optional HackerRank Hiring Agent report."))] = None) -> AutomatedWorkflowResponse:
    job_provider: JobProvider | None = None
    try:
        try:
            options = WorkflowOptions.model_validate_json(options_json)
        except ValidationError as error:
            raise HTTPException(status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT), detail="Invalid workflow options.") from error

        pdf_extraction = await extract_uploaded_pdf(cv)
        hiring_agent_assessment = None
        if hiring_report is not None:
            safe_filename = (validate_hiring_agent_report_metadata(filename=hiring_report.filename, content_type=hiring_report.content_type))
            report_bytes = (await read_hiring_agent_report_bytes(hiring_report, maximum_size_bytes=(settings.max_hiring_agent_report_bytes)))
            hiring_agent_assessment = (parse_hiring_agent_report(report_bytes, source_filename=safe_filename, role_name=(options.hiring_agent_role)))

        llm_provider = runtime_factory.create_llm(options.llm)
        job_provider = runtime_factory.create_jobs(list(options.job_providers))
        profile_extractor = (StructuredCandidateProfileExtractor(llm_provider=llm_provider, maximum_cv_characters=(settings.max_cv_text_characters)))
        preference_extractor = (StructuredPreferenceExtractor(llm_provider=llm_provider, maximum_characters=(settings.max_preferences_text_characters)))
        search_planner = StructuredSearchPlanner(llm_provider=llm_provider)
        report_generator = (StructuredJobReportGenerator(llm_provider=llm_provider))
        workflow = AutomatedCareerMatchWorkflow(AutomatedWorkflowDependencies(profile_extractor=profile_extractor,
                                                                              preference_extractor=preference_extractor,
                                                                              search_planner=search_planner,
                                                                              job_provider=job_provider,
                                                                              embedding_provider=embedding_provider,
                                                                              report_generator=report_generator,
                                                                              maximum_evaluation_jobs=(settings.maximum_evaluation_jobs)))

        return await workflow.run(cv_text=pdf_extraction.text, preference_text=preference_text, hiring_agent_assessment=(hiring_agent_assessment), configuration=options.agent)

    except (UnsupportedPdfTypeError, UnsupportedHiringAgentReportError) as error:
        raise HTTPException(status_code=(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE), detail=str(error)) from error

    except (PdfTooLargeError, HiringAgentReportTooLargeError) as error:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(error)) from error

    except (InvalidPdfError, NoExtractableTextError, InvalidHiringAgentReportError, EmptyPreferenceTextError, PreferenceTextTooLongError) as error:
        raise HTTPException(status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT), detail=str(error)) from error

    except (LLMProviderConfigurationError, LLMProviderUnavailableError, JobProviderUnavailableError, EmbeddingModelUnavailableError) as error:
        raise HTTPException(status_code=(status.HTTP_503_SERVICE_UNAVAILABLE), detail=str(error)) from error

    except (LLMProviderResponseError, JobProviderResponseError, InvalidEmbeddingResponseError) as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    except GraphRecursionError as error:
        raise HTTPException(status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR), detail=("The agent workflow exceeded its maximum execution depth.")) from error

    finally:
        await cv.close()
        if hiring_report is not None:
            await hiring_report.close()

        if job_provider is not None:
            await job_provider.aclose()

@router.get("/capabilities", response_model=WorkflowCapabilities)
def get_workflow_capabilities(settings: Annotated[Settings, Depends(get_settings)]) -> WorkflowCapabilities:
    """Return safely exposable frontend configuration."""
    return WorkflowCapabilities(llm_providers=[ProviderCapability(name="ollama", configured=True),
                                               ProviderCapability(name="gemini", configured=(settings.gemini_api_key is not None)),
                                               ProviderCapability(name="openai", configured=(settings.openai_api_key is not None))],
                                job_providers=[ProviderCapability(name="arbeitnow", configured=True),
                                               ProviderCapability(name="adzuna", configured=(settings.adzuna_app_id is not None and settings.adzuna_app_key is not None)),
                                               ProviderCapability(name="jooble", configured=(settings.jooble_api_key is not None))],
                                default_llm_provider=settings.llm_provider,
                                default_llm_model=settings.llm_model,
                                default_job_providers=settings.job_providers)
