# Example Usage

This guide demonstrates the current manual end-to-end workflow for **CareerMatch Agent `v0.1.0-alpha`**.

The goal is to start with a CV PDF and finish with a JSON file containing ranked job opportunities.

The current alpha exposes the individual stages through FastAPI. A future release is planned to automate these steps behind a single user-facing workflow.

---

## Workflow

```text
CV PDF
   ↓
Candidate Profile Extraction
   ↓
Job Preferences
   ↓
Optional HackerRank Hiring Agent Assessment
   ↓
CareerMatch Agent
   ↓
Job Search
   ↓
Deterministic Filtering
   ↓
Hybrid Ranking
   ↓
Evidence-Grounded Evaluation
   ↓
Ranked Jobs JSON
```

---

## Prerequisites

Before starting, make sure the following are installed:

- Python
- Git
- `curl`
- `jq`
- Ollama

Clone CareerMatch Agent:

```bash
git clone https://github.com/LittleBigPluton/career-match-agent
cd career-match-agent
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the project:

```bash
pip install -e ".[dev]"
```

If the repository contains an environment template, create your local configuration:

```bash
cp .env.example .env
```

Review `.env.example` and configure the local settings required by your installation.

Do not commit the resulting `.env` file.

---

# 1. Ollama Setup

CareerMatch Agent `v0.1.0-alpha` uses Ollama as its local LLM backend.

The project can be used with either:

- a standard system-wide Ollama installation, or
- an isolated project-local Ollama runtime.

The project-local setup is the configuration used during development and testing of this release.

---

### Standard Ollama Installation

If Ollama is installed system-wide, pull the configured model:

```bash
ollama pull gemma3:4b
```

Start Ollama:

```bash
ollama serve
```

If Ollama is already running as a background service, starting it again is not necessary.

Verify the installed models with:

```bash
ollama list
```

CareerMatch Agent expects the configured Ollama model and base URL to match the values in the local environment configuration.

---

### Project-Local Ollama Runtime

For development, CareerMatch Agent was also tested with an isolated Ollama installation stored inside the project directory.

This keeps the Ollama runtime, model files, temporary files, and local home directory separate from a system-wide Ollama installation.

Start the local runtime with:

```bash
HOME="$PWD/.local-ollama/home" \
OLLAMA_MODELS="$PWD/.local-ollama/models" \
TMPDIR="$PWD/.local-ollama/tmp" \
OLLAMA_NO_CLOUD=1 \
OLLAMA_CONTEXT_LENGTH=8192 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NUM_PARALLEL=1 \
"$PWD/.local-ollama/runtime/bin/ollama" serve
```

This configuration:

- stores Ollama-related files under `.local-ollama/`
- keeps model files local to the project
- keeps temporary files local to the project
- disables Ollama cloud functionality
- configures an 8192-token context window
- limits Ollama to one loaded model at a time
- limits inference to one parallel request

This setup can be useful on machines with limited memory or when a reproducible project-local environment is preferred.

---

### Pulling a Model with the Project-Local Runtime

If the Ollama runtime is stored under `.local-ollama/runtime/`, use the same local environment variables when pulling models:

```bash
HOME="$PWD/.local-ollama/home" \
OLLAMA_MODELS="$PWD/.local-ollama/models" \
TMPDIR="$PWD/.local-ollama/tmp" \
OLLAMA_NO_CLOUD=1 \
"$PWD/.local-ollama/runtime/bin/ollama" pull gemma3:4b
```

Verify the locally installed models with:

```bash
HOME="$PWD/.local-ollama/home" \
OLLAMA_MODELS="$PWD/.local-ollama/models" \
"$PWD/.local-ollama/runtime/bin/ollama" list
```

---

### Ollama Runtime Configuration

The development configuration uses:

```text
OLLAMA_NO_CLOUD=1
OLLAMA_CONTEXT_LENGTH=8192
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
```

These values are not mandatory for every machine.

Users with more available CPU, RAM, or GPU resources may choose different values.

The important requirement is that the selected model is available and that CareerMatch Agent can reach the local Ollama server.

---

### Local Files and Version Control

The project-local Ollama directory may contain:

- Ollama runtime binaries
- downloaded model files
- temporary files
- local Ollama state

These files should not be committed to Git.

Add the following to `.gitignore`:

```gitignore
.local-ollama/
```

---

### Verify Ollama Before Starting CareerMatch Agent

After Ollama starts successfully, verify that the configured model is available:

```bash
ollama list
```

or, when using the project-local runtime:

```bash
HOME="$PWD/.local-ollama/home" \
OLLAMA_MODELS="$PWD/.local-ollama/models" \
"$PWD/.local-ollama/runtime/bin/ollama" list
```

---

# 2. Start CareerMatch Agent

From the CareerMatch Agent project directory, activate the virtual environment:

```bash
source venv/bin/activate
```

Start the FastAPI server:

```bash
uvicorn career_match_agent.api.main:app --reload
```

The default local API address is:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

You can verify that the API is running before continuing.

---

# 3. Extract the Candidate Profile

CareerMatch Agent first converts the CV into a structured `CandidateProfile`.

Use an absolute path to the CV PDF:

```bash
curl -sS \
  -X POST \
  "http://127.0.0.1:8000/profiles/candidate/extract" \
  -H "accept: application/json" \
  -F "file=@/absolute/path/to/cv.pdf;type=application/pdf" \
  > profile_response.json
```

For example:

```bash
curl -sS \
  -X POST \
  "http://127.0.0.1:8000/profiles/candidate/extract" \
  -H "accept: application/json" \
  -F "file=@$HOME/Downloads/my_cv.pdf;type=application/pdf" \
  > profile_response.json
```

Inspect the complete extraction response:

```bash
jq '.' profile_response.json
```

Inspect only the extracted profile:

```bash
jq '.profile' profile_response.json
```

---

## Inspect Atomic Skills

CareerMatch extracts technical skills as individual atomic values.

Inspect them with:

```bash
jq '.profile.skills' profile_response.json
```

Example:

```json
[
  "LangGraph",
  "Ollama",
  "PyTorch",
  "Hugging Face Transformers",
  "scikit-learn",
  "Python",
  "FastAPI",
  "Pydantic",
  "REST APIs",
  "Docker",
  "Pandas",
  "NumPy",
  "SQL",
  "PostgreSQL",
  "AWS EC2"
]
```

Skills should not normally appear as combined category strings such as:

```text
Machine Learning & NLP: PyTorch, Hugging Face Transformers, scikit-learn
```

Instead, individual technologies and capabilities should appear as separate entries.

---

## Save the Candidate Profile

Extract only the `CandidateProfile` object:

```bash
jq '.profile' \
  profile_response.json \
  > candidate_profile.json
```

Verify it:

```bash
jq '.' candidate_profile.json
```

At this point:

```text
CV PDF
   ↓
profile_response.json
   ↓
candidate_profile.json
```

---

# 4. Define Job Preferences

CareerMatch Agent combines the candidate profile with structured job-search preferences.

Create:

```text
preferences.json
```

For example:

```json
{
  "roles": [
    "AI Engineer",
    "Machine Learning Engineer",
    "Data Scientist"
  ],
  "locations": [
    "Germany",
    "Munich",
    "Berlin"
  ],
  "work_modes": [
    "hybrid",
    "on_site"
  ],
  "required_keywords": [],
  "excluded_keywords": []
}
```

The exact `JobPreferences` schema may evolve between releases.

For the authoritative schema of the version you are running, inspect:

```text
http://127.0.0.1:8000/docs
```

or inspect an existing valid request.

Verify your preferences file:

```bash
jq '.' preferences.json
```

---

# 5. Optional: HackerRank Hiring Agent <sup>[Source](https://github.com/interviewstreet/hiring-agent)</sup>

CareerMatch Agent can optionally consume an assessment produced by the separate open-source **HackerRank Hiring Agent** project.

This step is optional.

If you do not want to use Hiring Agent, continue to:

**[6. Build the CareerMatch Agent Reques](#6. Build the CareerMatch Agent Request)**

and use an empty `evidence_signals` list.

---

## 5.1 Install HackerRank Hiring Agent

It is recommended to clone Hiring Agent outside the CareerMatch Agent repository.

For example:

```bash
cd ~/Desktop
git clone https://github.com/interviewstreet/hiring-agent
cd hiring-agent
```
hackerRank Hiring Agent requires Python 3.11 or newer so verif it is available:

```bash
python --version
```

Create a dedicated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install its dependencies:

```bash
pip install -r requirements.txt
```

Create its local environment file:

```bash
cp .env.example .env
```

The upstream project supports Ollama for local inference and Gemini as a hosted provider.

For local Ollama usage, configure the Hiring Agent `.env` according to its current `.env.example`.

For example, make sure the selected Ollama model is installed:

```bash
ollama pull gemma3:4b
```

and that Ollama is running:

```bash
ollama serve
```

Hiring Agent and CareerMatch Agent have separate environments and configuration files.

---

## 5.2 Run HackerRank Hiring Agent

Activate the Hiring Agent environment:

```bash
cd ~/Desktop/hiring-agent
source .venv/bin/activate
```

Run the assessment:

```bash
python score.py /absolute/path/to/cv.pdf
```

To preserve the terminal report in a file while still displaying it:

```bash
python score.py /absolute/path/to/cv.pdf \
  | tee hiring_agent_report.txt
```

Example:

```bash
python score.py "$HOME/Downloads/my_cv.pdf" \
  | tee hiring_agent_report.txt
```

The upstream Hiring Agent may also create caches or CSV artifacts when its development mode is enabled.

---

# 5.3 Parse the Hiring Agent Report with CareerMatch

Return to CareerMatch Agent:

```bash
cd ~/Desktop/career-match-agent
source venv/bin/activate
```

Make sure the CareerMatch API is running.

CareerMatch exposes a parser endpoint for Hiring Agent text reports:

```text
POST /assessments/hiring-agent/parse
```

Send the report:

```bash
curl -sS \
  -X POST \
  "http://127.0.0.1:8000/assessments/hiring-agent/parse" \
  -H "accept: application/json" \
  -F "report=@/absolute/path/to/hiring_agent_report.txt;type=text/plain" \
  -F "role_name=software_engineering_intern" \
  > hiring_agent_assessment.json
```

Inspect the normalized result:

```bash
jq '.' hiring_agent_assessment.json
```

The `role_name` field identifies the assessment context used by CareerMatch.

It does not have to be inferred from the CV.

---

## HackerRank Integration in `v0.1.0-alpha`

The Hiring Agent integration is currently an optional supporting-data workflow rather than a fully automated part of `/agent/search`.

The current release intentionally keeps:

```text
CandidateProfile
```

and:

```text
Hiring Agent assessment evidence
```

as separate concepts.

This prevents an external assessment from silently overwriting CV-derived candidate facts.

The planned automated workflow will eventually allow the user to provide:

```text
CV
+
preferences
+
optional Hiring Agent report
+
LLM selection
```

in one procedure.

For the current alpha, `/agent/search` can also be used without Hiring Agent evidence by passing:

```json
"evidence_signals": []
```

---

# 6. Build the CareerMatch Agent Request

The main agent expects:

```text
CandidateProfile
+
JobPreferences
+
optional evidence signals
```

For a CV-only run, create `agent_request.json` with `jq`:

```bash
jq -n \
  --slurpfile profile candidate_profile.json \
  --slurpfile preferences preferences.json \
  '{
    profile: $profile[0],
    preferences: $preferences[0],
    evidence_signals: []
  }' \
  > agent_request.json
```

Inspect it:

```bash
jq '.' agent_request.json
```

Verify that the new atomic skills are present:

```bash
jq '.profile.skills' agent_request.json
```

---

## Reusing an Existing Agent Request

During development it can be useful to preserve an existing valid configuration while replacing only the candidate profile.

For example:

```bash
jq \
  --slurpfile profile candidate_profile.json \
  '.profile = $profile[0]' \
  existing_agent_request.json \
  > agent_request.json
```

This preserves the preferences and configuration of the existing request while replacing the profile.

---

# 7. Run the CareerMatch Agent

Send the request to:

```text
POST /agent/search
```

Run:

```bash
curl -sS \
  -o agent_response.json \
  -w "HTTP %{http_code}\n" \
  -X POST \
  "http://127.0.0.1:8000/agent/search" \
  -H "Content-Type: application/json" \
  --data @agent_request.json
```

A successful HTTP request should print:

```text
HTTP 200
```

Inspect the response:

```bash
jq '.' agent_response.json
```

---

# 8. What the Agent Does

The `/agent/search` endpoint runs the bounded LangGraph workflow.

Approximately:

```text
plan_search
    ↓
search_jobs
    ↓
filter_jobs
    ↓
enough jobs?
   /      \
 yes       no
  |         ↓
  |    broaden_search
  |         ↓
  |     search_jobs
  |         ↓
  |     filter_jobs
  ↓
rank_jobs
    ↓
evaluate_jobs
```

The individual stages perform:

### Search Planning

Creates bounded role-oriented search queries from the candidate profile and preferences.

### Job Retrieval

Queries the configured job provider.

`v0.1.0-alpha` currently uses Arbeitnow.

### Deterministic Filtering

Applies hard constraints such as:

- role compatibility
- seniority
- location
- language requirements
- required keywords
- excluded keywords

### Hybrid Ranking

Ranks accepted jobs using signals including:

- semantic similarity
- skill overlap
- required keyword matches
- role alignment
- warning quality

### Evidence-Grounded Evaluation

Generates a suitability report for top-ranked jobs using candidate, job, deterministic, and semantic evidence.

Unsupported LLM findings are filtered by deterministic grounding validation.

---

# 9. Inspect Ranked Jobs

Check how many jobs were ranked:

```bash
jq '.ranking.ranked_jobs | length' \
  agent_response.json
```

Inspect the top-ranked job:

```bash
jq '.ranking.ranked_jobs[0]' \
  agent_response.json
```

Inspect a compact score breakdown:

```bash
jq '
.ranking.ranked_jobs[]
| {
    rank,
    title: .decision.job.title,
    company: .decision.job.company,
    location: .decision.job.location,
    hybrid_score,
    semantic_score: .score_breakdown.semantic_score,
    skill_overlap_score: .score_breakdown.skill_overlap_score,
    matched_skills: .score_breakdown.matched_skills
  }
' agent_response.json
```

Example:

```json
{
  "rank": 1,
  "title": "Applied AI Engineer",
  "company": "Example Company",
  "location": "Germany",
  "hybrid_score": 53.57,
  "semantic_score": 61.25,
  "skill_overlap_score": 9.8,
  "matched_skills": [
    "prompt engineering",
    "model evaluation",
    "Python",
    "REST APIs",
    "SQL"
  ]
}
```

---

# 10. Inspect Grounded Evaluation Results

Check evaluation statistics:

```bash
jq '.evaluation.statistics' \
  agent_response.json
```

Example:

```json
{
  "received_count": 4,
  "attempted_count": 4,
  "completed_count": 4,
  "failed_count": 0
}
```

Check any evaluation failures:

```bash
jq '.evaluation.failures' \
  agent_response.json
```

A completely successful run should normally return:

```json
[]
```

Inspect the first suitability report:

```bash
jq '.evaluation.reports[0]' \
  agent_response.json
```

Inspect only its recommendation and explanation sections:

```bash
jq '
.evaluation.reports[]
| {
    title,
    company,
    recommendation: .report.recommendation,
    confidence: .report.confidence,
    summary: .report.summary,
    strengths: .report.strengths,
    gaps: .report.gaps,
    risks: .report.risks
  }
' agent_response.json
```

---

# 11. Export Ranked Jobs to JSON

To save the complete ranked-job objects:

```bash
jq '.ranking.ranked_jobs' \
  agent_response.json \
  > ranked_jobs.json
```

Inspect:

```bash
jq '.' ranked_jobs.json
```

---

## Create a Smaller `jobs.json`

For a simpler final output:

```bash
jq '[
  .ranking.ranked_jobs[]
  | {
      rank,
      source_id: .decision.job.source_id,
      title: .decision.job.title,
      company: .decision.job.company,
      location: .decision.job.location,
      url: (
        .decision.job.url
        // .decision.job.source_url
        // null
      ),
      hybrid_score,
      semantic_score: .score_breakdown.semantic_score,
      skill_overlap_score: .score_breakdown.skill_overlap_score,
      matched_skills: .score_breakdown.matched_skills
    }
]' \
  agent_response.json \
  > jobs.json
```

Inspect the final job list:

```bash
jq '.' jobs.json
```

This file can now be used as the final machine-readable result of the manual CareerMatch workflow.

---

# 12. Export Suitability Reports

Grounded evaluation reports can also be stored separately:

```bash
jq '.evaluation.reports' \
  agent_response.json \
  > job_reports.json
```

Inspect:

```bash
jq '.' job_reports.json
```

---

# Final File Flow

A complete CV-only run produces approximately:

```text
my_cv.pdf
    ↓
profile_response.json
    ↓
candidate_profile.json
    +
preferences.json
    ↓
agent_request.json
    ↓
agent_response.json
    ├──→ ranked_jobs.json
    ├──→ jobs.json
    └──→ job_reports.json
```

With optional Hiring Agent assessment:

```text
my_cv.pdf
    │
    ├─────────────────────────────┐
    ↓                             ↓
CareerMatch profile         Hiring Agent
extraction                       ↓
    ↓                    hiring_agent_report.txt
candidate_profile.json           ↓
    │                    CareerMatch parser
    │                             ↓
    │                    hiring_agent_assessment.json
    │
    └──────────────┬──────────────┘
                   ↓
          CareerMatch workflow
                   ↓
          agent_response.json
                   ↓
               jobs.json
```

---

# Development and Debugging

Run the full regression suite before testing major workflow changes:

```bash
ruff check .
mypy
pytest -v
```

Run only profile extraction tests:

```bash
pytest -v tests/test_profile_extractor.py
```

Run only evaluator tests:

```bash
pytest -v tests/test_job_evaluator.py
```

---

# Troubleshooting

## `curl: (26) Failed to open/read local data`

This normally means the file path passed to `curl` does not exist.

Check the file first:

```bash
ls -l /absolute/path/to/cv.pdf
```

A common mistake is using:

```text
/../../Downloads/cv.pdf
```

The leading `/` makes the path absolute.

Instead use:

```text
../../Downloads/cv.pdf
```

or preferably:

```text
$HOME/Downloads/cv.pdf
```

---

## Profile response is empty or invalid

Inspect the complete response:

```bash
jq '.' profile_response.json
```

Also inspect the FastAPI terminal and Ollama terminal for errors.

Verify that Ollama is running:

```bash
ollama list
```

---

## No jobs are ranked

Inspect the agent response:

```bash
jq '.' agent_response.json
```

Possible causes include:

- no current provider results matching the target roles,
- deterministic constraints rejecting all retrieved jobs,
- overly restrictive preferences,
- language or location requirements,
- job-provider availability.

Search broadening is bounded and will not continue indefinitely.

---

## `skill_overlap_score` is zero

First inspect the extracted skills:

```bash
jq '.profile.skills' profile_response.json
```

They should be atomic entries such as:

```json
[
  "Python",
  "SQL",
  "FastAPI"
]
```

rather than entire categorized skill lines.

Then inspect:

```bash
jq '
.ranking.ranked_jobs[]
| {
    title: .decision.job.title,
    skill_overlap_score: .score_breakdown.skill_overlap_score,
    matched_skills: .score_breakdown.matched_skills
  }
' agent_response.json
```

A zero score can still be legitimate when the job description contains no explicit overlap with the extracted skills.

---

## Evaluation failures

Inspect:

```bash
jq '.evaluation.failures' \
  agent_response.json
```

CareerMatch deliberately rejects some malformed or insufficiently grounded LLM reports rather than silently accepting unsupported claims.

---

# Current Alpha Limitations

`v0.1.0-alpha` is a technical alpha.

The current workflow still has several limitations:

- Ollama is the primary supported CareerMatch LLM backend.
- Arbeitnow is the primary job provider.
- The workflow is currently API-driven.
- CV extraction and agent execution are separate manual steps.
- HackerRank Hiring Agent integration is optional and not yet part of a single automated request.
- There is no persistent database.
- There is no user-facing web application.
- Current job availability depends on external provider data.
- Small local LLMs may still produce imperfect outputs despite structured generation and deterministic grounding validation.

These limitations are expected to change in later releases.

---

# Planned Workflow

A future CareerMatch release is intended to reduce the manual process to something closer to:

```text
Upload CV
    +
Choose LLM provider/model
    +
Set job preferences
    +
Optional Hiring Agent report
    ↓
Run CareerMatch
    ↓
Receive current ranked jobs
and grounded suitability reports
```

Until then, this document provides a reproducible manual workflow for testing the complete `v0.1.0-alpha` pipeline.
