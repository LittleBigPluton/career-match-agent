import json
import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("CAREER_MATCH_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="CareerMatch Agent", page_icon="🔎", layout="wide")
# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_capabilities() -> dict[str, Any]:
    """Load safely exposable backend capabilities."""
    response = httpx.get(f"{API_BASE_URL}/workflow/capabilities", timeout=10.0)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_candidate_summary(result: dict[str, Any]) -> None:
    """Render extracted candidate profile and preferences."""
    profile = result.get("profile", {})
    preferences = result.get("preferences", {})
    st.header("Candidate profile")
    professional_summary = profile.get("professional_summary")
    if professional_summary:
        st.write(professional_summary)

    skills = profile.get("skills", [])
    if skills:
        st.subheader("Skills")
        st.write(", ".join(str(skill) for skill in skills))

    languages = profile.get("languages", [])
    if languages:
        st.subheader("Languages")
        for language in languages:
            if isinstance(language, dict):
                name = language.get("language", language.get("name", "Unknown"))
                level = language.get("level")
                if level:
                    st.write(f"- {name}: {level}")
                else:
                    st.write(f"- {name}")

            else:
                st.write(f"- {language}")

    st.header("What CareerMatch understood")
    roles = preferences.get("roles", [])
    locations = preferences.get("locations", [])
    work_modes = preferences.get("work_modes", [])
    employment_types = preferences.get("employment_types", [])
    seniority_levels = preferences.get("seniority_levels", [])
    left_column, right_column = (st.columns(2))
    with left_column:
        st.markdown("**Roles**")
        if roles:
            for role in roles:
                st.write(f"- {role}")
        else:
            st.write("Not specified")

        st.markdown("**Locations**")
        if locations:
            for location in locations:
                st.write(f"- {location}")
        else:
            st.write("Not specified")

        st.markdown("**Work modes**")
        if work_modes:
            for work_mode in work_modes:
                st.write(f"- {work_mode}")
        else:
            st.write("Not specified")

    with right_column:
        st.markdown("**Employment types**")

        if employment_types:
            for employment_type in employment_types:
                st.write(f"- {employment_type}")
        else:
            st.write("Not specified")

        st.markdown("**Seniority**")
        if seniority_levels:
            for seniority in seniority_levels:
                st.write(f"- {seniority}")
        else:
            st.write("Not specified")


def render_metrics(result: dict[str, Any]) -> None:
    """Render high-level workflow metrics."""
    agent = result.get("agent", {})
    filtering_statistics = agent.get("filtering_statistics", {})
    ranking = agent.get("ranking", {})
    ranking_statistics = ranking.get("statistics", {})
    evaluation = agent.get("evaluation", {})
    evaluation_statistics = evaluation.get("statistics", {})
    search_attempts = agent.get("search_attempts", 0)
    accepted_jobs = (filtering_statistics.get("accepted_count", 0))
    ranked_jobs = (ranking_statistics.get("ranked_count", 0))
    evaluated_jobs = (evaluation_statistics.get("completed_count", 0))

    st.header("Workflow summary")
    (attempt_column, accepted_column, ranked_column, evaluated_column) = st.columns(4)
    attempt_column.metric("Search attempts", search_attempts)
    accepted_column.metric("Suitable jobs", accepted_jobs)
    ranked_column.metric("Ranked jobs", ranked_jobs)
    evaluated_column.metric("Detailed reports", evaluated_jobs)


def render_hiring_agent_summary(result: dict[str, Any]) -> None:
    """Render optional external HackerRank Hiring Agent evidence."""
    assessment = result.get("hiring_agent_assessment")

    if assessment is None:
        return

    st.header("External HackerRank evidence")
    st.caption("This assessment was produced externally. CareerMatch did not run HackerRank itself.")
    candidate_name = assessment.get("candidate_name")
    role_name = assessment.get("role_name")
    evidence_signal_count = result.get("evidence_signal_count", 0)
    if candidate_name:
        st.write(f"Candidate: {candidate_name}")

    if role_name:
        st.write(f"Role: {role_name}")

    st.write(f"Evidence signals used: {evidence_signal_count}")

def render_agent_trace(result: dict[str, Any]) -> None:
    """Render the real LangGraph execution trace."""
    agent = result.get("agent", {})
    trace = agent.get("trace", [])
    if not trace:
        return

    st.header("Agent execution trace")
    with st.expander("Show workflow trace", expanded=False):
        for index, entry in enumerate(trace, start=1):
            if not isinstance(entry, dict):
                st.write(entry)
                continue

            step = entry.get("step", "unknown")
            message = entry.get("message", "")
            st.markdown(f"**{index}. {step}**")
            if message:
                st.write(message)


def render_workflow_downloads(result: dict[str, Any]) -> None:
    """Offer reusable and debugging JSON artifacts."""
    prepared_state = result.get("prepared_state")
    agent_request = result.get("agent_request")
    agent_response = result.get("agent")
    artifact_run_id = result.get("artifact_run_id")

    if (prepared_state is None and agent_request is None and agent_response is None and not artifact_run_id):
        return

    st.header("Workflow artifacts")
    with st.expander("JSON artifacts and workflow reuse", expanded=False):
        st.warning("These files can contain information derived from your CV and job preferences. Store them appropriately.")
        if prepared_state is not None:
            st.download_button(label=("Download reusable prepared workflow"),
                               data=json.dumps(prepared_state, ensure_ascii=False, indent=2),
                               file_name=("career_match_prepared_workflow.json"),
                               mime="application/json",
                               use_container_width=True)

            st.caption("Use this file later with 'Reuse prepared workflow' to skip CV extraction, preference extraction and HackerRank preprocessing.")

        if agent_request is not None:
            st.download_button(label=("Download agent request"),
                               data=json.dumps(agent_request, ensure_ascii=False, indent=2),
                               file_name=("career_match_agent_request.json"),
                               mime="application/json",
                               use_container_width=True)

        if agent_response is not None:
            st.download_button(label=("Download agent response"),
                               data=json.dumps(agent_response, ensure_ascii=False, indent=2),
                               file_name=("career_match_agent_response.json"),
                               mime="application/json",
                               use_container_width=True)

        if artifact_run_id:
            st.success(f"Workflow artifacts were also recorded locally. Run ID: {artifact_run_id}")


def render_ranked_jobs(result: dict[str, Any]) -> None:
    """Render ranked jobs and detailed evaluations."""
    agent = result.get("agent", {})
    ranking = agent.get("ranking", {})
    ranked_jobs = ranking.get("ranked_jobs", [])
    evaluation = agent.get("evaluation", {})
    reports = evaluation.get("reports", [])
    reports_by_source_id = {report["source_id"]: report for report in reports if isinstance(report, dict) and report.get("source_id")}
    st.header("Ranked jobs")
    if not ranked_jobs:
        st.info("No suitable ranked jobs were found.")
        return

    for ranked_job in ranked_jobs:
        if not isinstance(ranked_job, dict):
            continue

        decision = ranked_job.get("decision", {})
        if not isinstance(decision, dict):
            continue

        job = decision.get("job", {})
        if not isinstance(job, dict):
            continue

        rank = ranked_job.get("rank", "?")
        title = job.get("title", "Untitled job")
        company = job.get("company", "Unknown company")
        location = job.get("location")
        source_id = job.get("source_id")
        url = job.get("url")
        hybrid_score = ranked_job.get("hybrid_score")
        report = (reports_by_source_id.get(source_id) if source_id else None)

        with st.container(border=True):
            heading_column, score_column = (st.columns([4, 1]))

            with heading_column:
                st.subheader(f"{rank}. {title}")
                st.write(f"**{company}**")
                if location:
                    st.caption(location)

            with score_column:
                if hybrid_score is not None:
                    st.metric("Hybrid score", f"{float(hybrid_score):.2f}")

            # --------------------------------------------------------------
            # Ranking evidence
            # --------------------------------------------------------------
            matched_skills = ranked_job.get("matched_skills", [])
            if matched_skills:
                st.write("**Matched skills:** "+ ", ".join(str(skill) for skill in matched_skills))

            # --------------------------------------------------------------
            # Detailed LLM report
            # --------------------------------------------------------------
            if report is not None:
                generated = report.get("report", {})
                if isinstance(generated, dict):
                    recommendation = generated.get("recommendation")
                    if recommendation:
                        st.markdown(f"**Recommendation:** {recommendation}")

                    summary = generated.get("summary")
                    if isinstance(summary, dict):
                        summary_text = (summary.get("text"))
                        if summary_text:
                            st.write(summary_text)

                    elif summary:
                        st.write(summary)

                    strengths = generated.get("strengths", [])
                    if strengths:
                        st.markdown("**Strengths**")
                        for strength in strengths:
                            if isinstance(strength, dict):
                                st.write(f"✓ **{strength.get('title', '')}** — {strength.get('explanation', '')}")

                            else:
                                st.write(f"✓ {strength}")

                    gaps = generated.get("gaps", [])
                    if gaps:
                        st.markdown("**Potential gaps**")
                        for gap in gaps:
                            if isinstance(gap, dict):
                                st.write(f"• **{gap.get('title', '')}** — {gap.get('explanation', '')}")

                            else:
                                st.write(f"• {gap}")

                    risks = generated.get("risks", [])
                    if risks:
                        st.markdown("**Risks / uncertainties**")
                        for risk in risks:
                            if isinstance(risk, dict):
                                st.write(f"• **{risk.get('title', '')}** — {risk.get('explanation', '')}")

                            else:
                                st.write(f"• {risk}")

            if url:
                st.link_button("Open job posting", str(url))

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🔎 CareerMatch Agent")
st.caption("Upload your CV and preferences or reuse a previously prepared CareerMatch workflow.")


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
try:
    capabilities = load_capabilities()

except (httpx.HTTPError, ValueError) as error:
    st.error("CareerMatch backend could not be reached. Make sure FastAPI is running.")
    st.code(str(error))
    st.stop()

llm_capabilities = capabilities.get("llm_providers", [])
job_capabilities = capabilities.get("job_providers", [])
configured_llms = [capability["name"] for capability in llm_capabilities if capability.get("configured", False)]
configured_job_providers = [capability["name"] for capability in job_capabilities if capability.get("configured", False)]
if not configured_llms:
    st.error("No configured LLM provider is available.")
    st.stop()

if not configured_job_providers:
    st.error("No configured job provider is available.")
    st.stop()

default_llm_provider = capabilities.get("default_llm_provider")
if (default_llm_provider not in configured_llms):
    default_llm_provider = (configured_llms[0])


default_llm_model = capabilities.get("default_llm_model", "")
default_job_providers = [provider for provider in capabilities.get("default_job_providers", []) if provider in configured_job_providers]
if not default_job_providers:
    default_job_providers = [configured_job_providers[0]]


# ---------------------------------------------------------------------------
# Candidate input mode
# ---------------------------------------------------------------------------
st.header("Candidate input")
input_mode = st.radio("Choose how to provide candidate information", options=["Build from CV", "Reuse prepared workflow",], horizontal=True)
if input_mode == "Reuse prepared workflow":
    st.info("A prepared workflow already contains the extracted candidate profile, structured job preferences, HackerRank-derived evidence signals"
            " and previous agent configuration. CV and preference extraction will be skipped.")


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
with st.form("career_match_workflow_form"):
    cv_file = None
    hiring_report = None
    prepared_workflow_file = None
    preference_text = ""

    if input_mode == "Build from CV":
        st.subheader("CV and preferences")
        cv_file = st.file_uploader("Candidate CV", type=["pdf"], help=("Upload the candidate CV as a PDF document."))
        preference_text = st.text_area("Job preferences", height=150,
                                       placeholder=("Example: I am looking for junior machine learning, AI engineering or applied AI roles in Germany."
                                                    "Berlin, Munich and remote positions are preferred."),
                                       help=("CareerMatch converts this text into structured JobPreferences using the selected LLM."))
        hiring_report = st.file_uploader(("HackerRank Hiring Agent report (optional)"), type=["json"],
                                          help=("Upload an externally generated HackerRank Hiring Agent JSON report. CareerMatch does not execute HackerRank."))

    else:
        st.subheader("Prepared workflow")
        prepared_workflow_file = (st.file_uploader(("Prepared CareerMatch workflow JSON"), type=["json"],
                                  help=("Upload the career_match_prepared_workflow.json file exported by a previous run.")))

    st.divider()
    st.subheader("LLM configuration")
    llm_provider = st.selectbox("LLM provider", options=configured_llms, index=(configured_llms.index(default_llm_provider)))
    llm_model = st.text_input("LLM model", value=default_llm_model,
                              help=("Examples: gemma3:4b for Ollama or the configured hosted-provider model."))

    if llm_provider in {"openai", "gemini"}:
        st.warning("This is a hosted LLM provider. Candidate-derived information will be sent to the selected provider during LLM stages.")
    st.divider()
    st.subheader("Job providers")
    job_providers = st.multiselect("Search providers", options=configured_job_providers, default=default_job_providers,
                                   help=("Arbeitnow is available without API keys. Adzuna and Jooble require their respective credentials."))

    st.divider()
    st.subheader("Agent settings")
    (minimum_column, attempts_column, reports_column) = st.columns(3)

    with minimum_column:
        minimum_accepted_jobs = (st.number_input("Minimum suitable jobs", min_value=1, max_value=50, value=5, step=1))

    with attempts_column:
        maximum_search_attempts = (st.number_input("Maximum search attempts", min_value=1, max_value=3, value=2, step=1))

    with reports_column:
        maximum_evaluation_jobs = (st.number_input("Detailed reports for top", min_value=1, max_value=20, value=2, step=1,
                                   help=("Local Ollama models can become slow when many detailed reports are generated sequentially.")))

    st.divider()
    record_artifacts = st.checkbox("Save workflow artifacts locally", value=False,
                                   help=("Records workflow JSON artifacts under the configured local artifact directory. These files can contain CV-derived personal information."))

    submitted = st.form_submit_button("Find matching jobs", use_container_width=True)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
if submitted:
    st.session_state.pop("career_match_result", None)
    # ---------------------------------------------------------------------
    # Common validation
    # ---------------------------------------------------------------------
    if not llm_model.strip():
        st.error("Please enter an LLM model.")
        st.stop()

    if not job_providers:
        st.error("Please select at least one job provider.")
        st.stop()

    # ---------------------------------------------------------------------
    # Options shared by both workflow modes
    # ---------------------------------------------------------------------
    options = {"llm": {"provider": llm_provider, "model": llm_model.strip()},
               "job_providers": job_providers,
               "record_artifacts": (record_artifacts),
               "agent": {"minimum_accepted_jobs": int(minimum_accepted_jobs),
                         "maximum_search_attempts": int(maximum_search_attempts),
                         "evaluation": {"maximum_jobs": int(maximum_evaluation_jobs)}}}

    files: dict[str, tuple[str, bytes, str]]
    data: dict[str, str]

    # ---------------------------------------------------------------------
    # Prepared workflow mode
    # ---------------------------------------------------------------------
    if (input_mode == "Reuse prepared workflow"):
        if (prepared_workflow_file is None):
            st.error("Please upload a prepared CareerMatch workflow JSON.")
            st.stop()

        files = {"prepared_workflow": (prepared_workflow_file.name, prepared_workflow_file.getvalue(), "application/json")}
        data = {"options_json": json.dumps(options)}
        endpoint = (f"{API_BASE_URL}/workflow/run-prepared")
        status_message = ("Running CareerMatch from prepared workflow...")

    # ---------------------------------------------------------------------
    # Normal CV mode
    # ---------------------------------------------------------------------
    else:
        if cv_file is None:
            st.error("Please upload a CV PDF.")
            st.stop()

        if not preference_text.strip():
            st.error("Please describe your job preferences.")
            st.stop()

        files = {"cv": (cv_file.name, cv_file.getvalue(), "application/pdf",)}
        if hiring_report is not None:
            files["hiring_report"] = (hiring_report.name, hiring_report.getvalue(), (hiring_report.type or "application/json"))

        data = {"preference_text": (preference_text.strip()), "options_json": json.dumps(options)}

        endpoint = (f"{API_BASE_URL}/workflow/run")
        status_message = ("Running complete CareerMatch workflow...")

    # ---------------------------------------------------------------------
    # Execute
    # ---------------------------------------------------------------------
    workflow_timeout = httpx.Timeout(connect=10.0, read=3600.0, write=60.0, pool=60.0)
    try:
        with st.status(status_message, expanded=True) as workflow_status:
            st.write("CareerMatch is processing the request.")
            if (input_mode == "Build from CV"):
                st.write(
                    "Candidate preprocessing, job search, filtering, ranking and evaluation will run.")
            else:
                st.write("Candidate preprocessing is skipped. Job search, filtering, ranking and evaluation will run.")

            with httpx.Client(timeout=workflow_timeout) as client:
                response = client.post(endpoint, files=files, data=data)
                response.raise_for_status()

            result = response.json()
            st.session_state["career_match_result"] = result
            workflow_status.update(
                label=("CareerMatch workflow completed."), state="complete", expanded=False)

    except httpx.HTTPStatusError as error:
        try:
            response_payload = (error.response.json())
            detail = response_payload.get("detail", error.response.text)

        except ValueError:
            detail = (error.response.text)

        st.error(f"CareerMatch request failed: {detail}")

    except httpx.TimeoutException:
        st.error("The CareerMatch request exceeded the frontend timeout. A local Ollama generation may still be running on the backend.")

    except httpx.RequestError as error:
        st.error(f"Could not reach the CareerMatch API: {error}")

    except ValueError as error:
        st.error(f"CareerMatch returned an invalid JSON response: {error}")


# ---------------------------------------------------------------------------
# Persisted result rendering
# ---------------------------------------------------------------------------
if ("career_match_result" in st.session_state):
    result = st.session_state["career_match_result"]
    st.divider()
    render_candidate_summary(result)

    st.divider()
    render_metrics(result)
    render_hiring_agent_summary(result)
    render_agent_trace(result)
    render_workflow_downloads(result)

    st.divider()
    render_ranked_jobs(result)
