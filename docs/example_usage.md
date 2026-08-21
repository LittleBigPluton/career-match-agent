# Example Usage

This guide demonstrates the current manual end-to-end workflow for **CareerMatch Agent `v0.2.0-alpha`**.

The goal is to start with a CV PDF and finish with a JSON file containing ranked job opportunities and grounded suitability reports.

`v0.2.0-alpha` supports a globally selected LLM provider. The same configured provider/model is used across LLM-dependent stages of the workflow.

---

## Workflow

```text
Choose LLM Provider / Model
   ↓
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
Job Search Planning
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

You also need one supported LLM backend:

- Ollama for local inference, or
- a Gemini API key, or
- an OpenAI API key.

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

Create your local environment configuration:

```bash
cp .env.example .env
```

Do not commit the resulting `.env` file.

---

# 1. Configure the LLM Provider

CareerMatch Agent `v0.2.0-alpha` uses one globally selected LLM provider/model for all LLM-dependent stages.

Core environment variables:

```dotenv
CAREER_MATCH_LLM_PROVIDER=ollama
CAREER_MATCH_LLM_MODEL=gemma3:4b
CAREER_MATCH_LLM_TIMEOUT_SECONDS=1200

CAREER_MATCH_OLLAMA_BASE_URL=http://127.0.0.1:11434

CAREER_MATCH_OPENAI_API_KEY=
CAREER_MATCH_GEMINI_API_KEY=
```

Supported provider values:

```text
ollama
gemini
openai
```

The selected provider is used for:

- structured candidate profile extraction,
- initial search planning,
- bounded search replanning,
- evidence-grounded job suitability evaluation.

PDF parsing, Arbeitnow retrieval, deterministic filtering, embeddings, ranking, LangGraph orchestration, and grounding validation remain non-LLM stages.

## 1.1 Ollama

For local inference:

```dotenv
CAREER_MATCH_LLM_PROVIDER=ollama
CAREER_MATCH_LLM_MODEL=gemma3:4b
CAREER_MATCH_LLM_TIMEOUT_SECONDS=1200
CAREER_MATCH_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Pull the configured model:

```bash
ollama pull gemma3:4b
```

Start Ollama:

```bash
ollama serve
```

Verify installed models:

```bash
ollama list
```

### Optional Project-Local Ollama Runtime

For development, CareerMatch Agent can also be used with an isolated Ollama installation stored inside the project directory.

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

Keep `.local-ollama/` out of version control:

```gitignore
.local-ollama/
```

## 1.2 Gemini API

For hosted Gemini inference:

```dotenv
CAREER_MATCH_LLM_PROVIDER=gemini
CAREER_MATCH_LLM_MODEL=<supported-gemini-model>
CAREER_MATCH_LLM_TIMEOUT_SECONDS=300
CAREER_MATCH_GEMINI_API_KEY=<your-api-key>
```

No Gemini model weights are downloaded locally. The selected model is accessed through the Gemini API.

Do not commit the API key.

## 1.3 OpenAI API

For hosted OpenAI inference:

```dotenv
CAREER_MATCH_LLM_PROVIDER=openai
CAREER_MATCH_LLM_MODEL=<supported-openai-model>
CAREER_MATCH_LLM_TIMEOUT_SECONDS=300
CAREER_MATCH_OPENAI_API_KEY=<your-api-key>
```

No OpenAI model weights are downloaded locally. The selected model is accessed through the OpenAI API.

Do not commit the API key.

## 1.4 Verify Active Configuration

You can verify the selected provider/model without printing secrets:

```bash
python - <<'PY'
from career_match_agent.core.config import get_settings

settings = get_settings()

print("Provider:", settings.llm_provider)
print("Model:", settings.llm_model)
print("Gemini key configured:", settings.gemini_api_key is not None)
print("OpenAI key configured:", settings.openai_api_key is not None)
PY
```

---

# 2. Start CareerMatch Agent

Activate the virtual environment:

```bash
source venv/bin/activate
```

Start FastAPI:

```bash
uvicorn career_match_agent.api.main:app --reload
```

The default local API address is:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 3. Extract the Candidate Profile

CareerMatch Agent converts the CV into a structured `CandidateProfile` using the configured LLM provider.

```bash
curl -sS \
  -X POST \
  "http://127.0.0.1:8000/profiles/candidate/extract" \
  -H "accept: application/json" \
  -F "file=@$HOME/Documents/my_cv.pdf;type=application/pdf" \
  | tee profile_response.json \
  | jq
```

Inspect the complete response:

```bash
jq '.' profile_response.json
```

Inspect only the extracted profile:

```bash
jq '.profile' profile_response.json
```

## Inspect Atomic Skills

```bash
jq '.profile.skills' profile_response.json
```

## Save the Candidate Profile

```bash
jq '.profile' \
  profile_response.json \
  > candidate_profile.json
```

Verify:

```bash
jq '.' candidate_profile.json
```

---

# 4. Define Job Preferences

Create `preferences.json`, for example:

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

For the authoritative schema of the version you are running, inspect:

```text
http://127.0.0.1:8000/docs
```

Verify:

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
cd ~/hiring-agent
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
python score.py "$HOME/Documents/my_cv.pdf" \
  | tee hiring_agent_report.txt
```

The upstream Hiring Agent may also create caches or CSV artifacts when its development mode is enabled.

---

# 5.3 Parse the Hiring Agent Report with CareerMatch

Return to CareerMatch Agent:

```bash
cd ~/career-match-agent
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

## HackerRank Integration in `v0.2.0-alpha`

The Hiring Agent integration is currently an optional supporting-data workflow rather than a fully automated part of `/agent/search`.

The current release intentionally keeps:

```text
CandidateProfile
```

and:

```text
Hiring Agent assessment evidence
```

as separate concepts. This prevents an external assessment from silently overwriting CV-derived candidate facts. For the current alpha, `/agent/search` can also be used without Hiring Agent evidence by passing:

```json
"evidence_signals": []
```

---

CareerMatch exposes:

```text
POST /assessments/hiring-agent/parse
```

Example:

```bash
curl -sS \
  -X POST \
  "http://127.0.0.1:8000/assessments/hiring-agent/parse" \
  -H "accept: application/json" \
  -F "report=@/absolute/path/to/hiring_agent_report.txt;type=text/plain" \
  -F "role_name=software_engineering_intern" \
  > hiring_agent_assessment.json
```

---

# 6. Build the CareerMatch Agent Request

For a CV-only run:

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

Inspect:

```bash
jq '.' agent_request.json
```

For a CV-only run with hiring agent assessment,
```bash
jq -n \
  --slurpfile profile candidate_profile.json \
  --slurpfile preferences preferences.json \
  --slurpfile assessment hiring_agent_assessment.json \
  '{
    profile: $profile[0],
    preferences: $preferences[0],
    evidence_signals: $assessment[0].evidence_signals
  }' \
  > agent_request.json
```
Inspect:

```bash
jq '.evidence_signals' agent_request.json
jq '.' agent_request.json
```


---

# 7. Run the CareerMatch Agent

```bash
curl -sS \
  -o agent_response.json \
  -w "HTTP %{http_code}\n" \
  -X POST \
  "http://127.0.0.1:8000/agent/search" \
  -H "Content-Type: application/json" \
  --data @agent_request.json
```

A successful request should print:

```text
HTTP 200
```

Inspect:

```bash
jq '.' agent_response.json
```

---

# 8. What the Agent Does

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

- **Search planning:** configured LLM.
- **Job retrieval:** configured job provider; `v0.2.0-alpha` currently uses Arbeitnow.
- **Deterministic filtering:** hard constraints such as role, seniority, location, language, and keyword requirements.
- **Hybrid ranking:** semantic similarity, skill overlap, role alignment, required-keyword matching, and warning quality.
- **Grounded evaluation:** configured LLM plus deterministic evidence validation.

---

# 9. Inspect Agent Execution

```bash
jq '.trace' agent_response.json
jq '.final_search_plan' agent_response.json
jq '.search_statistics' agent_response.json
jq '.filtering_statistics' agent_response.json
```

These fields are especially useful when a search returns few or no accepted jobs.

---

# 10. Inspect Ranked Jobs

Count ranked jobs:

```bash
jq '.ranking.ranked_jobs | length' agent_response.json
```

Top result:

```bash
jq '.ranking.ranked_jobs[0]' agent_response.json
```

Compact ranking view:

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

---

# 11. Inspect Grounded Evaluation Results

```bash
jq '.evaluation.statistics' agent_response.json
jq '.evaluation.failures' agent_response.json
jq '.evaluation.reports[0]' agent_response.json
```

Compact report view:

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

# 12. Export Ranked Jobs

```bash
jq '.ranking.ranked_jobs' \
  agent_response.json \
  > ranked_jobs.json
```

Create a smaller final job list:

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

---

# 13. Export Suitability Reports

```bash
jq '.evaluation.reports' \
  agent_response.json \
  > job_reports.json
```

---

# Final File Flow

```text
.env
  ↓
selected LLM provider/model
  ↓
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

---

# Development and Debugging

Run the full regression suite:

```bash
ruff check .
mypy
pytest -v
```

Run LLM provider unit tests:

```bash
pytest -v \
  tests/test_llm_provider_factory.py \
  tests/test_ollama_llm_provider.py \
  tests/test_openai_llm_provider.py \
  tests/test_gemini_llm_provider.py
```

These provider tests use mocks and do not call real models or hosted APIs.

---

# Troubleshooting

## LLM provider configuration error

Verify the configured provider/model:

```bash
python - <<'PY'
from career_match_agent.core.config import get_settings

settings = get_settings()
print(settings.llm_provider)
print(settings.llm_model)
PY
```

For Gemini/OpenAI, verify the corresponding API key exists in `.env`.

For Ollama, verify the local service is running and the configured model is installed.

## Hosted provider authentication or availability errors

- confirm the corresponding API key,
- confirm the selected model is available to the account,
- confirm internet connectivity,
- inspect FastAPI logs for the provider-specific exception chain,
- never print or commit API keys.

## Ollama connection errors

```bash
ollama list
```

Verify:

```dotenv
CAREER_MATCH_LLM_PROVIDER=ollama
CAREER_MATCH_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

and ensure `CAREER_MATCH_LLM_MODEL` matches an installed model.

## `502 Bad Gateway` during job search

Inspect:

```bash
jq '.' agent_response.json
```

and FastAPI logs.

A `502` can originate from an external job-provider response or an LLM provider failure. For Arbeitnow-related failures, inspect the upstream payload/validation error before changing ranking or LLM configuration.

## No jobs are ranked

Inspect:

```bash
jq '.search_statistics' agent_response.json
jq '.filtering_statistics' agent_response.json
jq '.trace' agent_response.json
```

Possible causes include:

- no current provider results matching target roles,
- deterministic constraints rejecting all retrieved jobs,
- overly restrictive preferences,
- language, location, or seniority requirements,
- external provider data quality,
- limited provider coverage.

Search broadening is bounded.

A zero-result search is preferable to returning jobs that violate hard constraints.

## Evaluation failures

```bash
jq '.evaluation.failures' agent_response.json
```

CareerMatch deliberately rejects some malformed or insufficiently grounded LLM reports rather than silently accepting unsupported claims.

---

# Current Alpha Limitations

`v0.2.0-alpha` is a technical alpha.

Current limitations include:

- one global LLM provider/model is selected per application configuration,
- provider selection is configuration-driven rather than a user-facing runtime UI,
- Arbeitnow is the primary job provider,
- the workflow is API-driven,
- CV extraction and agent execution are separate manual steps,
- HackerRank Hiring Agent integration is optional and not yet part of a single automated request,
- there is no persistent database,
- there is no user-facing web application,
- current job availability depends on external provider data,
- external provider payloads may be incomplete or inconsistent,
- hosted providers may introduce cost, rate limits, network dependency, and provider-specific privacy considerations.

---

# Planned Workflow

A future CareerMatch release is intended to reduce the manual process to:

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

Until then, this document provides a reproducible manual workflow for testing the complete `v0.2.0-alpha` pipeline.
