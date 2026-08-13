import math
import re
import textwrap
from dataclasses import dataclass

from career_match_agent.models.candidate import (
    CandidateProfile,
    JobPreferences
)
from career_match_agent.models.hiring_agent import (
    CandidateEvidenceSignal,
    EvidencePolarity
)
from career_match_agent.models.job import JobPosting
from career_match_agent.models.matching import JobFilterDecision
from career_match_agent.models.ranking import (
    EmbeddingMetadata,
    HybridRankingConfiguration,
    HybridRankingRequest,
    HybridRankingResponse,
    HybridRankingStatistics,
    HybridScoreBreakdown,
    RankedJob,
    SemanticMatchEvidence
)
from career_match_agent.services.embedding import (
    EmbeddingProvider,
    InvalidEmbeddingResponseError
)
from career_match_agent.services.job_classifier import (
    contains_normalized_phrase,
    create_job_searchable_text
)


@dataclass(frozen=True)
class SemanticTextChunk:
    """Internal text unit submitted for embedding."""
    identifier: str
    kind: str
    text: str
    weight: float = 1.0


def clean_semantic_text(value: str) -> str:
    """Normalize whitespace while preserving readable sections."""
    cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines() if line.strip()]
    return "\n".join(cleaned_lines)


def split_text_into_chunks(text: str, *, maximum_characters: int) -> list[str]:
    """Split text into paragraph-aware character chunks."""
    cleaned_text = clean_semantic_text(text)
    if not cleaned_text:
        return []

    paragraphs = cleaned_text.splitlines()
    wrapped_parts: list[str] = []
    for paragraph in paragraphs:
        wrapped_paragraph = textwrap.wrap(paragraph, width=maximum_characters, break_long_words=False, break_on_hyphens=False)
        wrapped_parts.extend(wrapped_paragraph or [paragraph])

    chunks: list[str] = []
    current_chunk = ""
    for part in wrapped_parts:
        candidate_chunk = (f"{current_chunk}\n{part}".strip() if current_chunk else part)
        if (current_chunk and len(candidate_chunk) > maximum_characters):
            chunks.append(current_chunk)
            current_chunk = part
        else:
            current_chunk = candidate_chunk

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def append_semantic_section(chunks: list[SemanticTextChunk], *, identifier: str, kind: str,
                            text: str, weight: float, maximum_characters: int, maximum_chunks: int) -> None:
    """Split and append one weighted semantic section."""
    if len(chunks) >= maximum_chunks:
        return

    section_parts = split_text_into_chunks(text, maximum_characters=maximum_characters)
    if not section_parts:
        return

    available_slots = maximum_chunks - len(chunks)
    section_parts = section_parts[:available_slots]
    distributed_weight = weight / len(section_parts)
    for index, section_part in enumerate(section_parts):
        chunks.append(SemanticTextChunk(identifier=f"{identifier}:{index}", kind=kind, text=section_part, weight=distributed_weight))

def build_candidate_chunks(*, profile: CandidateProfile, preferences: JobPreferences, evidence_signals: list[CandidateEvidenceSignal],
                           configuration: HybridRankingConfiguration) -> list[SemanticTextChunk]:
    """Create weighted candidate-profile retrieval chunks."""
    chunks: list[SemanticTextChunk] = []
    append_semantic_section(chunks, identifier="target-roles", kind="target_roles", text=("Target roles: "+ ", ".join(preferences.roles)),
                            weight=2.5, maximum_characters=(configuration.chunk_max_characters),maximum_chunks=(configuration.maximum_candidate_chunks))

    profile_overview_parts: list[str] = []
    if profile.professional_summary:
        profile_overview_parts.append(f"Professional summary: {profile.professional_summary}")


    if profile_overview_parts:
        append_semantic_section(chunks, identifier="profile-overview", kind="profile_overview", text="\n".join(profile_overview_parts),
                                weight=2.0, maximum_characters=(configuration.chunk_max_characters),
                                maximum_chunks=(configuration.maximum_candidate_chunks))

    for index, experience in enumerate(profile.experience):
        experience_parts = ["Experience",f"Title: {experience.job_title or 'Unknown'}", ("Organization: "f"{experience.organization or 'Unknown'}")]

        if experience.highlights:
            experience_parts.append("Highlights: "+ " ".join(experience.highlights))
        append_semantic_section(chunks, identifier=f"experience-{index}", kind="experience", text="\n".join(experience_parts),
                                weight=1.5, maximum_characters=(configuration.chunk_max_characters), maximum_chunks=(configuration.maximum_candidate_chunks))

    for index, project in enumerate(profile.projects):
        project_parts = [f"Project: {project.name}"]
        if project.summary:
            project_parts.append(f"Summary: {project.summary}")

        if project.highlights:
            project_parts.append("Highlights: "+ " ".join(project.highlights))

        append_semantic_section(chunks, identifier=f"project-{index}", kind="project", text="\n".join(project_parts),
                                weight=1.25, maximum_characters=(configuration.chunk_max_characters),
                                maximum_chunks=(configuration.maximum_candidate_chunks))

    for index, education in enumerate(profile.education):
        education_parts = [" ".join(value for value in [education.degree, education.field_of_study, education.institution] if value)]

        if education.details:
            education_parts.append("Details: " + " ".join(education.details))
        education_text = "\n".join(education_parts)
        append_semantic_section(chunks, identifier=f"education-{index}", kind="education", text=f"Education: {education_text}", weight=0.5,
                                maximum_characters=(configuration.chunk_max_characters), maximum_chunks=(configuration.maximum_candidate_chunks))

    relevant_polarities = {EvidencePolarity.POSITIVE, EvidencePolarity.NEUTRAL}
    relevant_signals = [signal for signal in evidence_signals if signal.polarity in relevant_polarities]
    for index, signal in enumerate(relevant_signals):
        append_semantic_section(chunks,identifier=f"assessment-{index}", kind="assessment_evidence",text=(f"{signal.title}: {signal.evidence}"),
                                weight=0.75,maximum_characters=(configuration.chunk_max_characters),
                                maximum_chunks=(configuration.maximum_candidate_chunks))
    return chunks

def build_job_chunks(job: JobPosting,*,configuration: HybridRankingConfiguration,) -> list[SemanticTextChunk]:
    """Create job-title and description retrieval chunks."""
    employment_text = ", ".join(employment_type.value for employment_type in job.employment_types)
    header = clean_semantic_text("\n".join([f"Job title: {job.title}",f"Company: {job.company}", f"Location: {job.location or 'Unknown'}",
                                            "Employment type: "f"{employment_text or 'Unknown'}",f"Tags: {', '.join(job.tags)}"]))

    chunks = [SemanticTextChunk(identifier=f"{job.source_id}:header", kind="job_header",text=header)]
    description_parts = split_text_into_chunks(job.description,maximum_characters=(configuration.chunk_max_characters))
    for index, description_part in enumerate(description_parts):
        if len(chunks) >= configuration.maximum_job_chunks:
            break

        chunks.append(SemanticTextChunk(identifier=(f"{job.source_id}:description:{index}"), kind="job_description", text=description_part))

    return chunks


def average_top_similarities(similarities: list[float], *, evidence_count: int,) -> float:
    """Average the strongest semantic similarities."""
    if not similarities or evidence_count <= 0:
        return 0.0

    strongest = sorted( similarities, reverse=True)[:evidence_count]
    return sum(strongest) / len(strongest)


def normalize_vector(vector: list[float]) -> list[float]:
    """Return a unit-length embedding vector."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise InvalidEmbeddingResponseError("The embedding provider returned a zero vector.")

    return [value / magnitude for value in vector]

def cosine_similarity(first_vector: list[float], second_vector: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(first_vector) != len(second_vector):
        raise InvalidEmbeddingResponseError("Embedding vectors have different dimensions.")

    first_normalized = normalize_vector(first_vector)
    second_normalized = normalize_vector(second_vector)
    similarity = sum(first_value * second_value for first_value, second_value in zip(first_normalized, second_normalized, strict=True))
    return max(0.0, min(1.0, similarity))

def excerpt(text: str, *, maximum_characters: int = 240) -> str:
    """Create a short response-safe excerpt."""
    cleaned_text = clean_semantic_text(text)
    if len(cleaned_text) <= maximum_characters:
        return cleaned_text

    return cleaned_text[: maximum_characters - 1].rstrip() + "…"

def calculate_semantic_match( *, candidate_chunks: list[SemanticTextChunk], candidate_embeddings: list[list[float]], job_chunks: list[SemanticTextChunk],
                             job_embeddings: list[list[float]], evidence_count: int) -> tuple[float, list[SemanticMatchEvidence]]:
    """Calculate strongest candidate-to-job similarity."""
    if len(candidate_chunks) != len(candidate_embeddings):
        raise InvalidEmbeddingResponseError("Candidate embedding count is inconsistent.")

    if len(job_chunks) != len(job_embeddings):
        raise InvalidEmbeddingResponseError("Job embedding count is inconsistent.")

    description_embeddings = [(job_chunk, job_vector) for job_chunk, job_vector in zip(job_chunks, job_embeddings, strict=True) if job_chunk.kind == "job_description"]
    if not description_embeddings:
        return 0.0, []

    best_similarities: list[float] = []
    match_evidence: list[SemanticMatchEvidence] = []
    for candidate_chunk, candidate_vector in zip(candidate_chunks, candidate_embeddings, strict=True):
        if candidate_chunk.kind == "target_roles":
            continue

        best_similarity = -1.0
        best_job_chunk: SemanticTextChunk | None = None
        for (job_chunk, job_vector) in description_embeddings:
            similarity = cosine_similarity(candidate_vector, job_vector)

            if similarity > best_similarity:
                best_similarity = similarity
                best_job_chunk = job_chunk

        if best_job_chunk is None:
            continue

        best_similarities.append(best_similarity)
        match_evidence.append(SemanticMatchEvidence(candidate_chunk_kind=(candidate_chunk.kind), candidate_excerpt=excerpt(candidate_chunk.text),
                                                    job_excerpt=excerpt(best_job_chunk.text), similarity=round(best_similarity, 4)))

    semantic_similarity = (average_top_similarities(best_similarities, evidence_count=evidence_count))
    sorted_evidence = sorted(match_evidence, key=lambda item: item.similarity, reverse=True)
    return (semantic_similarity, sorted_evidence[:evidence_count])

def detect_matching_skills(profile: CandidateProfile, job: JobPosting) -> list[str]:
    """Return candidate skills found in the job posting."""
    searchable_text = create_job_searchable_text(job)
    return [skill for skill in profile.skills if contains_normalized_phrase(searchable_text, skill)]

def calculate_available_weighted_score(*, components: dict[str, float | None], configured_weights: dict[str, float]) -> tuple[float,dict[str, float],dict[str, float]]:
    """Calculate a weighted score using available components."""
    available_component_names = [name for name, score in components.items()if score is not None]
    available_weight_total = math.fsum(configured_weights[name] for name in available_component_names)
    if available_weight_total <= 0:
        return 0.0, {}, {}

    normalized_weights: dict[str, float] = {}
    raw_contributions: dict[str, float] = {}

    for name in available_component_names:
        component_score = components[name]

        if component_score is None:
            continue

        normalized_weight = (configured_weights[name] / available_weight_total)
        normalized_weights[name] = round(normalized_weight, 4)
        raw_contributions[name] = (component_score * normalized_weight)

    raw_final_score = math.fsum(raw_contributions.values())
    final_score = round(min(100.0, max(0.0, raw_final_score)),2)
    contributions = {name: round(value, 2) for name, value in raw_contributions.items()}

    # Keep displayed contributions consistent with the final score.
    contribution_total = round(math.fsum(contributions.values()), 2)
    rounding_difference = round(final_score - contribution_total, 2)
    if rounding_difference and contributions:
        adjustment_key = max(raw_contributions, key=lambda name: raw_contributions[name])
        contributions[adjustment_key] = round(contributions[adjustment_key] + rounding_difference, 2)

    return (final_score, normalized_weights, contributions)

def calculate_hybrid_breakdown(*, semantic_similarity: float, decision: JobFilterDecision, profile: CandidateProfile,
                               preferences: JobPreferences, configuration: HybridRankingConfiguration) -> tuple[float, HybridScoreBreakdown]:
    """Combine semantic and deterministic ranking components."""
    matched_skills = detect_matching_skills(profile, decision.job)
    missing_skills = [skill for skill in profile.skills if skill not in matched_skills]
    skill_overlap_score = (100 * len(matched_skills) / len(profile.skills) if profile.skills else None)
    required_keyword_score = ((100* len(decision.matched_required_keywords)/ len(preferences.required_keywords))if preferences.required_keywords else None)
    role_alignment_score = (100.0 if decision.matched_roles else 0.0)
    warning_quality_score = max(0.0, 100 * (1 - (len(decision.warnings) * configuration.warning_penalty)))
    semantic_score = 100 * semantic_similarity
    components: dict[str, float | None] = {"semantic": semantic_score, "skill_overlap": skill_overlap_score, "required_keywords": required_keyword_score,
                                           "role_alignment": role_alignment_score,"warning_quality": warning_quality_score}
    weights = configuration.weights
    configured_weights = {"semantic": weights.semantic, "skill_overlap": weights.skill_overlap, "required_keywords": weights.required_keywords,
                          "role_alignment": weights.role_alignment, "warning_quality": weights.warning_quality}

    hybrid_score, normalized_weights, contributions = calculate_available_weighted_score(components=components, configured_weights=configured_weights)
    return (hybrid_score,HybridScoreBreakdown(semantic_score=round(semantic_score,2,),
                                              skill_overlap_score=(round(skill_overlap_score, 2) if skill_overlap_score is not None else None),
                                              required_keyword_score=(round(required_keyword_score, 2) if required_keyword_score is not None else None),
                                              role_alignment_score=round(role_alignment_score, 2),
                                              warning_quality_score=round(warning_quality_score, 2),
                                              matched_skills=matched_skills,
                                              missing_skills=missing_skills,
                                              component_weights=normalized_weights,
                                              component_contributions=contributions))

class HybridJobRankingService:
    """Rank accepted jobs using semantic and deterministic signals."""
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    async def rank(self, request: HybridRankingRequest) -> HybridRankingResponse:
        configuration = request.configuration

        if not request.accepted_jobs:
            return HybridRankingResponse(embedding=EmbeddingMetadata(provider=(self.embedding_provider.provider_name),
                                                                     model=self.embedding_provider.model_name,
                                                                     dimension=self.embedding_provider.dimension),
                                         ranked_jobs=[],
                                         statistics=HybridRankingStatistics(received_count=0,
                                                                            ranked_count=0,
                                                                            returned_count=0,
                                                                            candidate_chunk_count=0,
                                                                            job_chunk_count=0))

        candidate_chunks = build_candidate_chunks(profile=request.profile,
                                                  preferences=request.preferences,
                                                  evidence_signals=request.evidence_signals,
                                                  configuration=configuration)
        candidate_embeddings = await self.embedding_provider.embed_queries([chunk.text for chunk in candidate_chunks])
        job_chunk_groups: list[tuple[JobFilterDecision, list[SemanticTextChunk], int, int]] = []
        all_job_chunks: list[SemanticTextChunk] = []
        for decision in request.accepted_jobs:
            job_chunks = build_job_chunks(decision.job, configuration=configuration)
            start_index = len(all_job_chunks)
            all_job_chunks.extend(job_chunks)
            end_index = len(all_job_chunks)
            job_chunk_groups.append((decision, job_chunks, start_index, end_index))

        all_job_embeddings = (await self.embedding_provider.embed_documents([chunk.text for chunk in all_job_chunks]))
        if len(all_job_embeddings) != len(all_job_chunks):
            raise InvalidEmbeddingResponseError("The embedding provider returned an invalid number of job vectors.")
        unranked_jobs: list[RankedJob] = []
        for (decision, job_chunks, start_index, end_index) in job_chunk_groups:
            job_embeddings = all_job_embeddings[start_index:end_index]
            semantic_similarity, semantic_evidence = calculate_semantic_match(candidate_chunks=candidate_chunks,
                                                                              candidate_embeddings=candidate_embeddings,
                                                                              job_chunks=job_chunks,
                                                                              job_embeddings=job_embeddings,
                                                                              evidence_count=(configuration.semantic_evidence_count))
            hybrid_score, score_breakdown = calculate_hybrid_breakdown(semantic_similarity=semantic_similarity, decision=decision,
                                                                       profile=request.profile, preferences=request.preferences,
                                                                       configuration=configuration)
            unranked_jobs.append(RankedJob(rank=1, hybrid_score=hybrid_score, decision=decision, score_breakdown=score_breakdown, semantic_matches=semantic_evidence))

        sorted_jobs = sorted(unranked_jobs,key=lambda ranked_job: (ranked_job.hybrid_score,ranked_job.score_breakdown.semantic_score,
                                                                   (ranked_job.decision.job.posted_at.timestamp() if ranked_job.decision.job.posted_at else 0)), reverse=True)
        ranked_jobs = [ranked_job.model_copy(update={"rank": rank,}) for rank, ranked_job in enumerate(sorted_jobs, start=1)]
        returned_jobs = ranked_jobs[: configuration.top_k]

        return HybridRankingResponse(embedding=EmbeddingMetadata(provider=self.embedding_provider.provider_name,
                                                                    model=self.embedding_provider.model_name,
                                                                    dimension=self.embedding_provider.dimension,),
                                    ranked_jobs=returned_jobs,
                                    statistics=HybridRankingStatistics(received_count=len(request.accepted_jobs),
                                                                       ranked_count=len(ranked_jobs),
                                                                       returned_count=len(returned_jobs),
                                                                       candidate_chunk_count=len(candidate_chunks),
                                                                       job_chunk_count=len(all_job_chunks)))
