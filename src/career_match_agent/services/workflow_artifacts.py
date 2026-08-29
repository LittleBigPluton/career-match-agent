import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile

from pydantic import (
    BaseModel,
    ValidationError
)
from career_match_agent.models.agent import (
    AgentSearchRequest,
    AgentSearchResponse
)
from career_match_agent.models.candidate import (
    CandidateProfile,
    JobPreferences
)
from career_match_agent.models.document import PdfExtractionResponse
from career_match_agent.models.hiring_agent import HiringAgentAssessment
from career_match_agent.models.workflow import PreparedWorkflowState


class WorkflowArtifactError(RuntimeError):
    """Raised when workflow artifacts cannot be recorded."""

class InvalidPreparedWorkflowError(WorkflowArtifactError):
    """Raised when an uploaded prepared state is invalid."""

@dataclass(frozen=True)
class WorkflowArtifactRun:
    """Filesystem destination for one CareerMatch run."""

    run_id: str
    directory: Path


class WorkflowArtifactStore:
    """Record local JSON artifacts for reproducibility and reuse."""

    def __init__(self, *, root_directory: str | Path) -> None:
        self.root_directory = Path(root_directory)

    def create_run(self) -> WorkflowArtifactRun:
        """Create a unique directory for one workflow."""
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = (f"{timestamp}-{uuid4().hex[:8]}")
        directory = (self.root_directory/run_id)
        try:
            directory.mkdir(parents=True, exist_ok=False)

        except OSError as error:
            raise WorkflowArtifactError("Could not create the workflow artifact directory.") from error

        return WorkflowArtifactRun(run_id=run_id, directory=directory)

    def write_model(self, *, run: WorkflowArtifactRun, filename: str, value: BaseModel) -> Path:
        """Serialize one Pydantic model atomically."""
        return self._write_json(run=run, filename=filename, payload=value.model_dump(mode="json"))

    def _write_json(self, *, run: WorkflowArtifactRun, filename: str, payload: object) -> Path:
        final_path = (run.directory/filename)
        temporary_path = (run.directory/ f".{filename}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            temporary_path.replace(final_path)

        except OSError as error:
            raise WorkflowArtifactError(f"Could not write workflow artifact '{filename}'.") from error

        return final_path


def record_preprocessing_artifacts(*, store: WorkflowArtifactStore, run: WorkflowArtifactRun, pdf_extraction: PdfExtractionResponse,
                                   profile: CandidateProfile, preferences: JobPreferences, hiring_agent_assessment: (HiringAgentAssessment | None),
                                   prepared_state: PreparedWorkflowState, agent_request: AgentSearchRequest) -> None:
    """Record artifacts available before graph execution."""
    store.write_model(run=run, filename="01_cv_extraction.json", value=pdf_extraction)
    store.write_model(run=run, filename="02_candidate_profile.json", value=profile)
    store.write_model(run=run, filename="03_job_preferences.json", value=preferences)
    if hiring_agent_assessment is not None:
        store.write_model(run=run, filename=("04_hiring_agent_assessment.json"), value=hiring_agent_assessment)

    store.write_model(run=run, filename="05_prepared_workflow.json", value=prepared_state)
    store.write_model(run=run, filename="06_agent_request.json", value=agent_request)


def record_agent_response(*, store: WorkflowArtifactStore, run: WorkflowArtifactRun, response: AgentSearchResponse) -> None:
    """Record the final LangGraph response."""
    store.write_model(run=run, filename="07_agent_response.json", value=response)


def parse_prepared_workflow(data: bytes) -> PreparedWorkflowState:
    """Validate an uploaded prepared workflow JSON."""
    if not data:
        raise InvalidPreparedWorkflowError("The prepared workflow file is empty.")

    try:
        return (PreparedWorkflowState.model_validate_json(data))

    except ValidationError as error:
        raise InvalidPreparedWorkflowError("The uploaded JSON is not a valid CareerMatch prepared workflow.") from error

async def read_prepared_workflow_upload(upload: UploadFile, *, maximum_size_bytes: int) -> bytes:
    """Read a bounded prepared-workflow upload."""
    filename = upload.filename
    if (not filename or not filename.casefold().endswith(".json")):
        raise InvalidPreparedWorkflowError("Prepared workflow input must be a JSON file.")
    contents = await upload.read(maximum_size_bytes + 1)

    if len(contents) > maximum_size_bytes:
        raise InvalidPreparedWorkflowError("Prepared workflow JSON exceeds the configured size limit.")

    return contents
