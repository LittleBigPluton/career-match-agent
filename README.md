# CareerMatch Agent

CareerMatch Agent is an explainable AI-assisted job-search and recommendation system built with FastAPI, LangGraph, deterministic filtering, semantic ranking, evidence-grounded LLM evaluation and an automated Streamlit workflow.

The system accepts a CV PDF and natural-language job preferences, builds a structured candidate profile, searches multiple job providers, filters unsuitable positions, ranks relevant opportunities and produces grounded suitability reports.

Users can select the LLM provider/model and job providers at runtime. Prepared workflow state can also be exported and reused to avoid repeating expensive CV and preference preprocessing.

> Current release: `v0.3.0-alpha`
---

## Overview

Typical job-search assistants rely heavily on unconstrained LLM judgment.

CareerMatch Agent uses a hybrid architecture instead:

- deterministic rules enforce hard constraints,
- embeddings rank jobs semantically,
- structured scoring combines several matching signals,
- LLMs generate structured candidate profiles, search plans and grounded explanations,
- deterministic grounding validation checks LLM-generated evaluation claims.

LLMs assist with interpretation, planning and explanation, while explicit user constraints and core matching rules remain deterministic.

---

## Current Workflow


```text
CV PDF
   +
Natural-language preferences
   +
Optional HackerRank report
   +
LLM / job-provider selection
   ↓
Candidate Profile Extraction
   ↓
Preference Extraction + Validation
   ↓
Search Planning
   ↓
Multi-Provider Job Retrieval
   ↓
Deduplication / Normalization
   ↓
Deterministic Filtering
   ↓
Semantic / Hybrid Ranking
   ↓
Evidence Bundle Construction
   ↓
Grounded LLM Evaluation
   ↓
Ranked Job Recommendations
```

The workflow is orchestrated through a bounded LangGraph agent.

A single configured LLM provider/model is used consistently across LLM-dependent stages of the workflow.

---

## Features

### Multi-LLM Provider Support

CareerMatch uses a common structured LLM provider interface.

Supported providers:

- Ollama
- Google Gemini
- OpenAI

The LLM provider and model can be selected for each automated workflow request through the UI or API.

Environment configuration defines defaults and provider credentials, while runtime workflow options determine which configured provider/model is used for a particular search.

The selected provider is reused across LLM-dependent stages including:

- candidate profile extraction,
- preference extraction,
- search planning,
- bounded search replanning,
- evidence-grounded job evaluation.

Non-LLM stages remain provider-independent:

- PDF extraction,
- job-provider retrieval,
- deterministic filtering,
- SentenceTransformer embeddings,
- hybrid ranking,
- LangGraph orchestration,
- grounding validation.

This keeps workflow logic independent from a specific LLM vendor.

---

### CV Processing

* PDF text extraction
* Structured candidate profile generation
* Atomic skill extraction
* Experience extraction
* Education extraction
* Language extraction
* Project extraction
* Evidence-preserving structured output

Example atomic skills:

```json
[
  "Python",
  "PyTorch",
  "FastAPI",
  "SQL",
  "Docker",
  "AWS EC2",
  "LangGraph"
]
```

---

### Job Preferences

Natural-language preferences are converted into structured `JobPreferences`.

Supported preference dimensions include:

- target roles
- cities, regions, and countries
- remote / hybrid / on-site work
- employment type
- seniority
- required keywords
- excluded keywords
- preferred languages

Explicit user constraints are treated as authoritative.

CareerMatch combines structured LLM extraction with deterministic validation so that explicitly stated preferences are not silently replaced by defaults or unsupported LLM assumptions.

For example, if no work mode is specified, the system treats work mode as unrestricted rather than assuming hybrid or on-site work.

---

### Multi-Provider Job Search

CareerMatch currently supports:

- Arbeitnow
- Adzuna
- Jooble

A composite provider layer allows multiple configured providers to participate in the same workflow.

Users can select which available providers should be searched for each run.

Provider responses are normalized into the shared CareerMatch job model before filtering and ranking.

Search planning can broaden the search when an initial attempt returns too few suitable jobs.

Provider-specific retrieval behavior is handled behind a common `JobProvider` interface so that downstream filtering, ranking, and evaluation remain provider-independent.

---

### Deterministic Filtering

Hard constraints are evaluated before semantic ranking.

Examples include:

* role mismatch
* seniority mismatch
* location mismatch
* required keyword mismatch
* excluded keyword presence
* language mismatch
* language-level mismatch

The deterministic layer is designed to prevent LLMs from overriding explicit user requirements.

---

### Hybrid Ranking

Accepted jobs are ranked using several signals:

* semantic similarity
* skill overlap
* required keyword matching
* role alignment
* warning quality

A typical ranking result contains both the final hybrid score and a detailed score breakdown.

Example:

```json
{
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

### Evidence-Grounded Evaluation

Top-ranked jobs can be evaluated by the configured LLM using an explicit evidence bundle.

Evidence can include:

* candidate summary
* candidate skills
* experience
* project evidence
* job-description excerpts
* matched roles
* matched skills
* required keyword matches
* semantic comparison evidence
* deterministic warnings
* optional HackerRank evidence

Each factual finding must cite supplied evidence IDs.

The evaluator validates:

* job identity
* unknown evidence IDs
* candidate evidence usage
* job evidence usage
* finding-level grounding

Unsupported strengths, gaps and risks are removed.

If a small local model fails to generate sufficiently grounded strengths, conservative fallback findings can be constructed from trusted deterministic or semantic comparison evidence.

---

### Optional HackerRank Hiring Agent Integration

CareerMatch Agent can parse a HackerRank Hiring Agent report and convert supported findings into candidate evidence signals.

These signals can then be included in job suitability evaluation.

This integration is optional.

---

### Automated End-to-End Workflow

`v0.3.0-alpha` introduces a complete user-facing workflow.

Through the Streamlit interface, a user can:

1. upload a CV PDF,
2. describe desired jobs in natural language,
3. optionally upload a HackerRank Hiring Agent report,
4. select an LLM provider and model,
5. select one or more configured job providers,
6. configure bounded agent-search settings,
7. run the complete CareerMatch pipeline,
8. inspect the interpreted profile and preferences,
9. review search and agent metrics,
10. inspect ranked jobs and grounded reports.

The frontend communicates with the FastAPI workflow API, so the underlying processing remains reusable outside the Streamlit interface.

---

### Reusable Workflow State and Artifacts

CareerMatch can export intermediate workflow state as structured JSON.

Available artifacts include:

- PDF extraction result
- candidate profile
- job preferences
- optional HackerRank assessment
- prepared workflow state
- agent search request
- agent search response

A prepared workflow can later be uploaded through the UI or API.

This skips repeated:

- CV text extraction,
- candidate profile extraction,
- preference extraction,
- HackerRank preprocessing.

The stored candidate state is reused while the selected LLM, job providers, agent configuration, and live job search can be changed.

Local workflow artifacts are optional and should not normally be committed because they may contain CV-derived personal information.

---
### Agentic Workflow

The current LangGraph flow is approximately:

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

Search attempts are bounded to prevent uncontrolled agent loops.

---

## Technology Stack

### Backend

* Python
* FastAPI
* Pydantic

### Frontend

* Streamlit
* HTTPX

### Agentic AI / LLM

* LangGraph
* common structured LLM provider interface
* Ollama
* Google Gemini API
* OpenAI API

### Machine Learning / Ranking

* SentenceTransformers
* semantic embeddings
* hybrid ranking
* deterministic matching

### Job Retrieval

* Arbeitnow
* Adzuna
* Jooble
* composite multi-provider execution

### Quality

* pytest
* mypy
* Ruff

### Infrastructure / Development

* Git
* Linux
* local or hosted LLM backends

---

## Installation

Clone the repository:

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

Create a local environment file:

```bash
cp .env.example .env
```

Do not commit `.env`.

---

## LLM Provider Setup

CareerMatch Agent `v0.3.0-alpha` supports Ollama, Gemini and OpenAI through a common structured LLM interface.

Environment variables provide default provider/model configuration and credentials. The automated workflow can override the default provider/model for an individual request.

Core settings:

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

Only credentials for the selected hosted provider are required.

The active provider is shared across candidate profile extraction, agent search planning/replanning and grounded job evaluation.

### Ollama

For local inference:

```dotenv
CAREER_MATCH_LLM_PROVIDER=ollama
CAREER_MATCH_LLM_MODEL=gemma3:4b
CAREER_MATCH_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Install and start Ollama separately.

Pull the configured model:

```bash
ollama pull gemma3:4b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

### Gemini

For Gemini API usage:

```dotenv
CAREER_MATCH_LLM_PROVIDER=gemini
CAREER_MATCH_LLM_MODEL=<supported-gemini-model>
CAREER_MATCH_GEMINI_API_KEY=<your-api-key>
```

The model runs remotely through the Gemini API.

Never commit API keys to version control.

### OpenAI

For OpenAI API usage:

```dotenv
CAREER_MATCH_LLM_PROVIDER=openai
CAREER_MATCH_LLM_MODEL=<supported-openai-model>
CAREER_MATCH_OPENAI_API_KEY=<your-api-key>
```

The model runs remotely through the OpenAI API.

Never commit API keys to version control.

## Job Provider Setup

Arbeitnow can be used without API credentials.

Optional hosted job providers require their respective credentials:

```dotenv
CAREER_MATCH_ADZUNA_APP_ID=
CAREER_MATCH_ADZUNA_APP_KEY=

CAREER_MATCH_JOOBLE_API_KEY=
```
---

## Running the API

Start FastAPI with:

```bash
uvicorn career_match_agent.api.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```
---

## Running the User Interface

Start the FastAPI backend first:

```bash
uvicorn career_match_agent.api.main:app --reload
```

Then, from another terminal:

```bash
streamlit run frontend/app.py
```

By default, the frontend expects CareerMatch at:
```bash
http://127.0.0.1:8000
```

A different backend can be configured with:
```bash
CAREER_MATCH_API_URL=http://127.0.0.1:8000
```

---

## Main API Capabilities

The project exposes routes for:

```text
/documents
/profiles
/assessments
/jobs
/matching
/ranking
/agent
/workflow
```

The exact request schemas are available through the FastAPI OpenAPI documentation.

---

## Quick Start

For a complete end-to-end example including:

- LLM provider selection
- CV extraction
- job preferences
- optional HackerRank Hiring Agent integration
- agent execution
- exporting ranked jobs to JSON

see:

[End-to-End Example Usage](docs/example_usage.md)

---

## Benchmarking

The project includes a benchmark dataset and runner for evaluating:

* deterministic filtering
* reason-code quality
* ranking precision
* ranking recall
* NDCG
* MRR
* latency

The benchmark is intended to make changes to the matching pipeline measurable rather than relying only on subjective inspection.

---

## Current Status

`v0.3.0-alpha`

CareerMatch now provides a working automated end-to-end workflow including:

* CV PDF processing
* structured candidate profile extraction
* natural-language preference extraction
* deterministic preference validation
* runtime Ollama / Gemini / OpenAI selection
* runtime job-provider selection
* Arbeitnow, Adzuna, and Jooble integration
* composite multi-provider search
* bounded LangGraph search and replanning
* deterministic filtering
* semantic / hybrid ranking
* evidence-grounded job evaluation
* optional HackerRank Hiring Agent evidence
* Streamlit user interface
* reusable prepared workflow state
* optional local workflow artifact recording

The release remains a technical alpha rather than a production-ready application.

---

## Privacy

CareerMatch Agent processes CV-derived information that may contain personal or sensitive data.

With a local Ollama configuration, LLM inference can remain on the user's machine.

When Gemini or OpenAI is selected, LLM-dependent workflow inputs are sent to the configured external API. Users should review the selected provider's privacy, retention and data-processing terms before submitting CV-derived information.

Generated files such as candidate profiles, agent requests, assessment reports and job-evaluation outputs may contain personal information and should normally remain outside public version control.

API keys must be stored in local environment configuration and must not be committed to the repository.

---

## Roadmap

### v0.2.0-alpha — Multi-LLM Support

Implemented:

* common structured LLM provider interface
* Ollama
* Gemini
* OpenAI
* provider-independent LLM services

### v0.3.0-alpha — Automated User Workflow

Implemented:

* Streamlit end-to-end interface
* CV upload workflow
* natural-language preference input
* runtime LLM provider/model selection
* runtime job-provider selection
* Arbeitnow, Adzuna, and Jooble providers
* composite multi-provider retrieval
* automated FastAPI workflow endpoint
* reusable prepared workflow state
* JSON workflow artifact export
* optional local artifact recording
* deterministic validation of explicit preferences

### Later

* stronger provider-aware search planning
* additional job providers
* improved retrieval diversity
* expanded benchmark and holdout datasets
* persistence / database support
* authentication
* deployment
* job monitoring and scheduled searches
* application workflow support

---

## Design Philosophy

CareerMatch Agent follows several principles:

1. Hard user constraints should be deterministic.
2. LLM output should be structured.
3. LLM-dependent services should not be coupled to a specific vendor.
4. One configured provider/model should be used consistently across the workflow.
5. LLM claims should be evidence-grounded.
6. Unknown or unsupported information should not be invented.
7. Semantic ranking and deterministic matching should complement each other.
8. Agent workflows should be bounded and observable.
9. Evaluation should be measurable through benchmarks.

---

## Disclaimer

CareerMatch Agent is an experimental project.

Generated recommendations should not be treated as definitive hiring or career decisions.

External job listings may change or become unavailable and LLM-generated explanations may still contain errors despite grounding and validation mechanisms.

Hosted LLM providers are external services and may have their own availability, pricing, rate limits, privacy terms and model lifecycle policies.

---

## License

CareerMatch Agent is licensed under the **GNU Affero General Public License v3.0 only (`AGPL-3.0-only`)**.

You may use, study, modify, redistribute and use the software commercially, subject to the terms of the AGPLv3.

The AGPL includes copyleft requirements for modified versions of the software, including when a modified version is made available for users to interact with over a network.

See the [LICENSE](LICENSE) file for the complete license terms.

Copyright © 2026 A. Umut Gökdemir.
