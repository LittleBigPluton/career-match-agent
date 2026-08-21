# CareerMatch Agent

CareerMatch Agent is an explainable AI-assisted job search and recommendation system built with FastAPI, LangGraph, deterministic filtering, semantic ranking, and evidence-grounded LLM evaluation.

The system takes a candidate profile derived from a CV, combines it with job preferences, searches available jobs, filters unsuitable roles, ranks relevant opportunities, and produces grounded suitability reports.

> Current release: `v0.2.0-alpha`

---

## Overview

Typical job-search assistants rely heavily on unconstrained LLM judgment.

CareerMatch Agent uses a hybrid architecture instead:

- deterministic rules enforce hard constraints,
- embeddings rank jobs semantically,
- structured scoring combines several matching signals,
- LLMs generate structured candidate profiles, search plans, and grounded explanations,
- deterministic grounding validation checks LLM-generated evaluation claims.

LLMs assist with interpretation, planning, and explanation, while explicit user constraints and core matching rules remain deterministic.

---

## Current Workflow

```text
CV PDF
   ↓
Candidate Profile Extraction
   ↓
Job Preferences
   ↓
Search Planning
   ↓
Job Provider
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

`v0.2.0-alpha` introduces a common structured LLM provider layer.

Supported providers:

- Ollama
- Google Gemini
- OpenAI

The active provider and model are selected globally through environment configuration.

The selected provider is used for LLM-dependent stages including:

- candidate profile extraction,
- search planning,
- bounded search replanning,
- evidence-grounded job suitability evaluation.

Non-LLM stages remain provider-independent:

- PDF extraction,
- job-provider retrieval,
- deterministic filtering,
- SentenceTransformer embeddings,
- hybrid ranking,
- LangGraph orchestration,
- grounding validation.

This keeps business logic independent from a specific LLM vendor and allows the complete workflow to switch providers without changing service code.

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

The system supports structured preferences such as:

* target roles
* locations
* remote / hybrid preferences
* seniority
* required keywords
* excluded keywords
* language requirements

---

### Job Search

The current release integrates the Arbeitnow public job provider.

Search planning can broaden the search when the first attempt returns too few suitable jobs.

Role search is intentionally constrained to relevant job-title and tag matches to reduce unrelated retrieval results.

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

Unsupported strengths, gaps, and risks are removed.

If a small local model fails to generate sufficiently grounded strengths, conservative fallback findings can be constructed from trusted deterministic or semantic comparison evidence.

---

### Optional HackerRank Hiring Agent Integration

CareerMatch Agent can parse a HackerRank Hiring Agent report and convert supported findings into candidate evidence signals.

These signals can then be included in job suitability evaluation.

This integration is optional.

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

CareerMatch Agent `v0.2.0-alpha` supports one globally selected LLM provider/model per application configuration.

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

The active provider is shared across candidate profile extraction, agent search planning/replanning, and grounded job evaluation.

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

## Main API Capabilities

The project currently exposes API routes for:

```text
/documents
/profiles
/assessments
/jobs
/matching
/ranking
/agent
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

`v0.2.0-alpha`

The core pipeline is working end-to-end with configurable LLM backends:

* CV extraction
* structured profile generation
* global Ollama / Gemini / OpenAI provider selection
* structured LLM output validation
* job retrieval
* deterministic filtering
* semantic ranking
* skill-overlap matching
* grounded evaluation
* LangGraph orchestration

The current release should be considered a technical alpha rather than a production-ready application.

---

## Privacy

CareerMatch Agent processes CV-derived information that may contain personal or sensitive data.

With a local Ollama configuration, LLM inference can remain on the user's machine.

When Gemini or OpenAI is selected, LLM-dependent workflow inputs are sent to the configured external API. Users should review the selected provider's privacy, retention, and data-processing terms before submitting CV-derived information.

Generated files such as candidate profiles, agent requests, assessment reports, and job-evaluation outputs may contain personal information and should normally remain outside public version control.

API keys must be stored in local environment configuration and must not be committed to the repository.

---

## Roadmap

### v0.2.0-alpha — Multi-LLM Support

Implemented:

* common structured LLM provider interface
* Ollama provider
* Gemini provider
* OpenAI provider
* global provider/model configuration
* shared provider injection across LLM-dependent services
* provider-independent profile extraction, search planning, and evaluation services

### v0.3 — Automated User Workflow

Target workflow:

```text
Upload CV
    +
Choose LLM
    +
Set preferences
    +
Optional HackerRank report
    ↓
Run CareerMatch
    ↓
Receive ranked available jobs
```

The goal is to remove the need to manually call multiple internal API endpoints.

### Later

* additional job providers
* stronger provider-data normalization
* persistence
* better benchmarking and calibration
* user interface
* authentication
* deployment
* job monitoring
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

External job listings may change or become unavailable, and LLM-generated explanations may still contain errors despite grounding and validation mechanisms.

Hosted LLM providers are external services and may have their own availability, pricing, rate limits, privacy terms, and model lifecycle policies.

---

## License

CareerMatch Agent is licensed under the **GNU Affero General Public License v3.0 only (`AGPL-3.0-only`)**.

You may use, study, modify, redistribute, and use the software commercially, subject to the terms of the AGPLv3.

The AGPL includes copyleft requirements for modified versions of the software, including when a modified version is made available for users to interact with over a network.

See the [LICENSE](LICENSE) file for the complete license terms.

Copyright © 2026 A. Umut Gökdemir.
