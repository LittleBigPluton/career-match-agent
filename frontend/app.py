import json
import os
from typing import Any, cast

import httpx
import streamlit as st


API_BASE_URL = os.getenv(
    "CAREER_MATCH_API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="CareerMatch Agent",
    page_icon="🔎",
    layout="wide",
)


def load_capabilities() -> dict[str, Any]:
    """Load safely exposable workflow configuration from FastAPI."""
    response = httpx.get(
        f"{API_BASE_URL}/workflow/capabilities",
        timeout=10.0,
    )
    response.raise_for_status()

    return cast(
        dict[str, Any],
        response.json(),
    )


def render_candidate_summary(
    result: dict[str, Any],
) -> None:
    """Show the candidate profile and interpreted preferences."""
    profile = result["profile"]
    preferences = result["preferences"]

    st.header("What CareerMatch understood")

    candidate_column, preference_column = st.columns(2)

    with candidate_column:
        st.subheader("Candidate profile")

        professional_summary = profile.get(
            "professional_summary"
        )

        if professional_summary:
            st.write(professional_summary)

        skills = profile.get(
            "skills",
            [],
        )

        st.write(
            "**Skills:**",
            ", ".join(skills)
            if skills
            else "Not identified",
        )

        languages = profile.get(
            "languages",
            [],
        )

        if languages:
            language_values: list[str] = []

            for language in languages:
                language_name = language.get(
                    "language",
                    "Unknown",
                )

                proficiency = language.get(
                    "proficiency"
                )

                if proficiency:
                    language_values.append(
                        f"{language_name} ({proficiency})"
                    )
                else:
                    language_values.append(
                        language_name
                    )

            st.write(
                "**Languages:**",
                ", ".join(language_values),
            )

    with preference_column:
        st.subheader("Job preferences")

        roles = preferences.get(
            "roles",
            [],
        )

        locations = preferences.get(
            "locations",
            [],
        )

        work_modes = preferences.get(
            "work_modes",
            [],
        )

        seniority_levels = preferences.get(
            "seniority_levels",
            [],
        )

        employment_types = preferences.get(
            "employment_types",
            [],
        )

        st.write(
            "**Roles:**",
            ", ".join(roles)
            if roles
            else "Not identified",
        )

        st.write(
            "**Locations:**",
            ", ".join(locations)
            if locations
            else "No strict location",
        )

        st.write(
            "**Work modes:**",
            ", ".join(work_modes)
            if work_modes
            else "No strict work mode",
        )

        st.write(
            "**Employment types:**",
            ", ".join(employment_types)
            if employment_types
            else "No strict employment type",
        )

        st.write(
            "**Seniority:**",
            ", ".join(seniority_levels)
            if seniority_levels
            else "No strict seniority",
        )


def render_metrics(
    result: dict[str, Any],
) -> None:
    """Show high-level workflow statistics."""
    agent = result["agent"]

    search_statistics = agent[
        "search_statistics"
    ]

    filtering_statistics = agent[
        "filtering_statistics"
    ]

    evaluation_statistics = agent[
        "evaluation"
    ]["statistics"]

    first, second, third, fourth = st.columns(4)

    first.metric(
        "Search attempts",
        agent["search_attempts"],
    )

    second.metric(
        "Jobs received",
        search_statistics["received_count"],
    )

    third.metric(
        "Suitable jobs",
        filtering_statistics["accepted_count"],
    )

    fourth.metric(
        "Detailed reports",
        evaluation_statistics[
            "completed_count"
        ],
    )


def render_agent_trace(
    result: dict[str, Any],
) -> None:
    """Show the existing LangGraph execution trace."""
    agent = result["agent"]

    with st.expander(
        "Agent execution trace"
    ):
        for entry in agent["trace"]:
            attempt = entry.get(
                "attempt"
            )

            attempt_text = (
                f" — attempt {attempt}"
                if attempt is not None
                else ""
            )

            st.markdown(
                f"**{entry['step']}**"
                f"{attempt_text}"
            )

            st.write(
                entry["message"]
            )


def render_hiring_agent_summary(
    result: dict[str, Any],
) -> None:
    """Show basic information when an external report was supplied."""
    assessment = result.get(
        "hiring_agent_assessment"
    )

    if assessment is None:
        return

    with st.expander(
        "HackerRank Hiring Agent evidence"
    ):
        st.caption(
            "This report was uploaded as optional external "
            "evidence. CareerMatch did not run HackerRank "
            "Hiring Agent itself."
        )

        candidate_name = assessment.get(
            "candidate_name"
        )

        role_name = assessment.get(
            "role_name"
        )

        if candidate_name:
            st.write(
                "**Candidate:**",
                candidate_name,
            )

        if role_name:
            st.write(
                "**Assessment role:**",
                role_name,
            )

        st.write(
            "**Evidence signals used:**",
            result.get(
                "evidence_signal_count",
                0,
            ),
        )


def render_ranked_jobs(
    result: dict[str, Any],
) -> None:
    """Show ranked jobs and grounded evaluation reports."""
    agent = result["agent"]

    ranked_jobs = agent[
        "ranking"
    ]["ranked_jobs"]

    evaluation = agent[
        "evaluation"
    ]

    reports_by_source_id = {
        report["source_id"]: report
        for report in evaluation["reports"]
    }

    st.header("Recommended jobs")

    if not ranked_jobs:
        st.info(
            "No jobs passed the current constraints."
        )
        return

    for ranked_job in ranked_jobs:
        job = ranked_job[
            "decision"
        ]["job"]

        source_id = job[
            "source_id"
        ]

        report = reports_by_source_id.get(
            source_id
        )

        with st.container(
            border=True
        ):
            heading_column, score_column = st.columns(
                [4, 1]
            )

            with heading_column:
                st.subheader(
                    f"{ranked_job['rank']}. "
                    f"{job['title']}"
                )

                st.write(
                    f"**{job['company']}**"
                )

                location = job.get(
                    "location"
                )

                if location:
                    st.caption(
                        f"📍 {location}"
                    )

                provider = job.get(
                    "provider"
                )

                if provider:
                    st.caption(
                        f"Source: {provider}"
                    )

            with score_column:
                st.metric(
                    "Hybrid score",
                    (
                        f"{ranked_job['hybrid_score']:.1f}"
                    ),
                )

            score_breakdown = ranked_job.get(
                "score_breakdown",
                {},
            )

            matched_skills = score_breakdown.get(
                "matched_skills",
                [],
            )

            if matched_skills:
                st.write(
                    "**Matched skills:** "
                    + ", ".join(
                        matched_skills
                    )
                )

            if report is not None:
                generated = report[
                    "report"
                ]

                recommendation = generated.get(
                    "recommendation"
                )

                confidence = generated.get(
                    "confidence"
                )

                recommendation_parts: list[str] = []

                if recommendation:
                    recommendation_parts.append(
                        str(recommendation).replace(
                            "_",
                            " ",
                        ).title()
                    )

                if confidence:
                    recommendation_parts.append(
                        (
                            f"{str(confidence).title()} "
                            "confidence"
                        )
                    )

                if recommendation_parts:
                    st.markdown(
                        "**Recommendation:** "
                        + " · ".join(
                            recommendation_parts
                        )
                    )

                summary = generated.get(
                    "summary"
                )

                if summary:
                    summary_text = summary.get(
                        "text"
                    )

                    if summary_text:
                        st.write(
                            summary_text
                        )

                strengths = generated.get(
                    "strengths",
                    [],
                )

                if strengths:
                    st.markdown(
                        "**Strengths**"
                    )

                    for strength in strengths:
                        st.write(
                            f"✓ **{strength['title']}** — "
                            f"{strength['explanation']}"
                        )

                gaps = generated.get(
                    "gaps",
                    [],
                )

                if gaps:
                    st.markdown(
                        "**Potential gaps**"
                    )

                    for gap in gaps:
                        st.write(
                            f"• **{gap['title']}** — "
                            f"{gap['explanation']}"
                        )

                risks = generated.get(
                    "risks",
                    [],
                )

                if risks:
                    st.markdown(
                        "**Risks / uncertainties**"
                    )

                    for risk in risks:
                        st.write(
                            f"• **{risk['title']}** — "
                            f"{risk['explanation']}"
                        )

                interview_focus = generated.get(
                    "interview_focus",
                    [],
                )

                if interview_focus:
                    with st.expander(
                        "Suggested interview focus"
                    ):
                        for item in interview_focus:
                            st.write(
                                f"• {item}"
                            )

            else:
                st.caption(
                    "No detailed evaluation report "
                    "was generated for this job."
                )

            job_url = job.get(
                "url"
            )

            if job_url:
                st.link_button(
                    "Open job posting",
                    str(job_url),
                )


st.title(
    "CareerMatch Agent"
)

st.caption(
    "Upload your CV and describe the job you want. "
    "CareerMatch will automatically search, filter, "
    "broaden, rank and evaluate available opportunities."
)


@st.cache_data(ttl=30)
def get_cached_capabilities() -> dict[str, Any]:
    """Cache frontend-safe backend capabilities briefly."""
    return load_capabilities()


try:
    capabilities = get_cached_capabilities()

except httpx.HTTPError as error:
    st.error(
        "CareerMatch backend is unavailable: "
        f"{error}"
    )

    st.info(
        "Start the FastAPI backend before running "
        "the Streamlit interface."
    )

    st.stop()


configured_llms = [
    provider["name"]
    for provider
    in capabilities["llm_providers"]
    if provider["configured"]
]

configured_jobs = [
    provider["name"]
    for provider
    in capabilities["job_providers"]
    if provider["configured"]
]


if not configured_llms:
    st.error(
        "No LLM provider is currently configured."
    )
    st.stop()


if not configured_jobs:
    st.error(
        "No job-search provider is currently configured."
    )
    st.stop()


default_llm_provider = capabilities.get(
    "default_llm_provider"
)

default_llm_index = (
    configured_llms.index(
        default_llm_provider
    )
    if default_llm_provider
    in configured_llms
    else 0
)


default_job_providers = [
    provider
    for provider
    in capabilities.get(
        "default_job_providers",
        [],
    )
    if provider in configured_jobs
]


if not default_job_providers:
    default_job_providers = (
        configured_jobs.copy()
    )


with st.form(
    "career_match_search"
):
    st.subheader(
        "Your profile"
    )

    cv_file = st.file_uploader(
        "CV",
        type=["pdf"],
        max_upload_size=5,
        help=(
            "Upload your CV as a PDF. "
            "The backend validates and extracts it."
        ),
    )

    preference_text = st.text_area(
        "What kind of job are you looking for?",
        height=150,
        placeholder=(
            "Example: I am looking for junior Machine "
            "Learning Engineer or AI Engineer jobs in "
            "Berlin or Munich. Remote or hybrid is preferred. "
            "Full-time only. English-speaking roles "
            "are preferred."
        ),
    )

    hiring_report = st.file_uploader(
        "HackerRank Hiring Agent report — optional",
        type=["json"],
        max_upload_size=1,
        help=(
            "Upload an existing JSON evaluation report "
            "generated separately by HackerRank Hiring Agent."
        ),
    )

    st.subheader(
        "AI"
    )

    llm_provider = st.selectbox(
        "LLM provider",
        options=configured_llms,
        index=default_llm_index,
    )

    llm_model = st.text_input(
        "Model",
        value=capabilities.get(
            "default_llm_model",
            "",
        ),
        help=(
            "Enter the model identifier supported by "
            "the selected provider."
        ),
    )

    if llm_provider in {
        "openai",
        "gemini",
    }:
        st.warning(
            "Selecting this hosted provider means "
            "CV-derived information will be sent to "
            f"the {llm_provider} API."
        )

    st.subheader(
        "Job sources"
    )

    job_providers = st.multiselect(
        "Search providers",
        options=configured_jobs,
        default=default_job_providers,
    )

    with st.expander(
        "Agent settings"
    ):
        minimum_accepted_jobs = st.number_input(
            "Minimum suitable jobs before stopping",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )

        maximum_search_attempts = st.number_input(
            "Maximum search attempts",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
        )

        maximum_evaluation_jobs = st.number_input(
            "Generate detailed reports for top",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )

    submitted = st.form_submit_button(
        "Find matching jobs",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if cv_file is None:
        st.error(
            "Please upload a CV."
        )
        st.stop()

    if not preference_text.strip():
        st.error(
            "Please describe the job you are looking for."
        )
        st.stop()

    if not job_providers:
        st.error(
            "Please select at least one job provider."
        )
        st.stop()

    if not llm_model.strip():
        st.error(
            "Please provide an LLM model name."
        )
        st.stop()

    # Remove a previous result while a new workflow is running.
    st.session_state.pop(
        "career_match_result",
        None,
    )

    options = {
        "llm": {
            "provider": llm_provider,
            "model": llm_model.strip(),
        },
        "job_providers": job_providers,
        "agent": {
            "minimum_accepted_jobs": int(
                minimum_accepted_jobs
            ),
            "maximum_search_attempts": int(
                maximum_search_attempts
            ),
            "evaluation": {
                "maximum_jobs": int(
                    maximum_evaluation_jobs
                )
            },
        },
    }

    files: dict[
        str,
        tuple[str, bytes, str],
    ] = {
        "cv": (
            cv_file.name,
            cv_file.getvalue(),
            "application/pdf",
        )
    }

    if hiring_report is not None:
        files[
            "hiring_report"
        ] = (
            hiring_report.name,
            hiring_report.getvalue(),
            (
                hiring_report.type
                or "application/json"
            ),
        )

    data = {
        "preference_text": (
            preference_text.strip()
        ),
        "options_json": json.dumps(
            options
        ),
    }

    with st.status(
        "CareerMatch is running...",
        expanded=True,
    ) as workflow_status:
        st.write(
            "Preparing candidate profile "
            "and running the matching workflow."
        )

        try:
            with httpx.Client(
                timeout=1800.0,
            ) as client:
                response = client.post(
                    (
                        f"{API_BASE_URL}"
                        "/workflow/run"
                    ),
                    files=files,
                    data=data,
                )

            response.raise_for_status()

        except httpx.HTTPStatusError:
            workflow_status.update(
                label="CareerMatch failed",
                state="error",
                expanded=True,
            )

            try:
                response_payload = response.json()

                detail = response_payload.get(
                    "detail",
                    response.text,
                )

            except ValueError:
                detail = response.text

            st.error(
                str(detail)
            )

            st.stop()

        except httpx.RequestError as error:
            workflow_status.update(
                label=(
                    "CareerMatch backend unavailable"
                ),
                state="error",
                expanded=True,
            )

            st.error(
                str(error)
            )

            st.stop()

        try:
            result = cast(
                dict[str, Any],
                response.json(),
            )

        except ValueError:
            workflow_status.update(
                label="Invalid backend response",
                state="error",
                expanded=True,
            )

            st.error(
                "The backend returned a response "
                "that could not be decoded as JSON."
            )

            st.stop()

        st.session_state[
            "career_match_result"
        ] = result

        workflow_status.update(
            label="CareerMatch completed",
            state="complete",
            expanded=False,
        )


if (
    "career_match_result"
    in st.session_state
):
    stored_result = cast(
        dict[str, Any],
        st.session_state[
            "career_match_result"
        ],
    )

    st.divider()

    render_candidate_summary(
        stored_result
    )

    st.divider()

    render_metrics(
        stored_result
    )

    render_hiring_agent_summary(
        stored_result
    )

    render_agent_trace(
        stored_result
    )

    st.divider()

    render_ranked_jobs(
        stored_result
    )
