# Vision-Link AI Agent 🏥⚡

> **Multi-Agent Autonomous Healthcare Assistant**
> Built with CrewAI + LangGraph for the [AI Agent Olympics — Lablab.ai](https://lablab.ai)
> **Deadline: May 19, 2026**

-----

## Architecture

```
Patient Input
     │
     ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Clinical Data  │────▶│  Localization    │────▶│   Validation     │
│  Agent (CrewAI) │     │  Agent (CrewAI)  │     │   Agent (CrewAI) │
│  MedLlama-3     │     │  Llama-3-8B      │     │   Llama-3-8B     │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                              ┌────────────▼────────────┐
                                              │   Conditional Routing    │
                                              │   (LangGraph edges)      │
                                              └────┬──────────────┬──────┘
                                                   │ FAIL         │ PASS
                                                   ▼              ▼
                                            ┌──────────┐   ┌───────────────┐
                                            │  Self-   │   │  Self-Evolve  │
                                            │  Healing │   │  Engine       │
                                            │  Loop    │   │  (Genetic AI) │
                                            └────┬─────┘   └──────┬────────┘
                                                 │                │
                                                 └────────────────▼
                                                          Final Response
```

-----

## Core Modules

|File                       |Owner     |Description                                                                        |
|---------------------------|----------|-----------------------------------------------------------------------------------|
|`workspace/state_schema.py`|Zentomo   |Pydantic v2 shared state — single source of truth across all agents                |
|`workspace/orchestrator.py`|Zentomo   |LangGraph StateGraph, conditional routing, self-healing loop, self-evolving backend|
|`workspace/model_loader.py`|Zentomo   |HuggingFace model loader — Llama-3-8B-Instruct & MedLlama, node-aware routing      |
|`workspace/crew_agents.py` |Subhalaxmi|CrewAI agent definitions, task schemas, Pydantic I/O contracts                     |

-----

## Self-Healing Loop

When the Validation Agent flags **errors or missing fields**, the orchestrator:

1. Identifies which agent produced the failure (clinical or localization)
1. Re-routes to **only that agent** — not the full pipeline
1. Retries up to `MAX_RETRIES` (default: 3)
1. Falls back to LLM-assisted patching for soft errors (e.g. low confidence score)
1. Marks run `failed` only after all retries are exhausted

## Self-Evolving Backend

Every successful run triggers the evolution engine:

1. **3 variants** of the pipeline are generated concurrently via LLM
1. Each variant compiles and runs in an **isolated sandbox**
1. Fitness scored: `latency_ms + error_penalty − throughput_bonus`
1. **Winner atomically replaces** the live module via Python hot-swap
1. Zero human intervention. Zero process restarts. Zero downtime.

-----

## Quick Start

```bash
# 1. Clone
git clone https://github.com/BuhariSalisuAI/vision-link-ai-agent.git
cd vision-link-ai-agent

# 2. Install
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 4. Smoke test
cd workspace
python orchestrator.py
```

-----

## Environment Variables

|Variable           |Required|Where to get it                                                         |
|-------------------|--------|------------------------------------------------------------------------|
|`HF_TOKEN`         |✅       |[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)|
|`EVOLUTION_ENABLED`|Optional|`true` / `false` (default: `true`)                                      |
|`MAX_RETRIES`      |Optional|Integer (default: `3`)                                                  |


> ⚠️ **Never share tokens in chat or commit `.env` to GitHub.**
> Use GitHub Actions secrets: Settings → Secrets and variables → Actions

-----

## CI / CD

GitHub Actions runs on every push to `main` and `dev`:

- ✅ Ruff linting
- ✅ Pydantic schema validation
- ✅ LangGraph graph compilation check
- ✅ HuggingFace model loader health check

See `.github/workflows/ci.yml`

-----

## Models

|Node                 |Model                                   |Reason                         |
|---------------------|----------------------------------------|-------------------------------|
|Clinical Agent       |`ProbeMedicalYonseiMAILab/medllama3-v20`|Highest diagnostic accuracy    |
|Localization Agent   |`meta-llama/Meta-Llama-3-8B-Instruct`   |Strong multilingual performance|
|Validation Agent     |`meta-llama/Meta-Llama-3-8B-Instruct`   |Reliable structured output     |
|Self-Healing / Evolve|`meta-llama/Meta-Llama-3-8B-Instruct`   |Balanced reasoning + speed     |
|Fallback             |`mistralai/Mistral-7B-Instruct-v0.3`    |Lightweight state transitions  |

-----

## Team

|Member       |GitHub                                              |Role                                                      |
|-------------|----------------------------------------------------|----------------------------------------------------------|
|Buhari Salisu|[@BuhariSalisuAI](https://github.com/BuhariSalisuAI)|Founder & CEO                                             |
|Subhalaxmi   |[@Subhalaxmi](https://github.com/Subhalaxmi)        |Data Engineer — CrewAI Agents & Tasks                     |
|Zentomo      |[@Zentomo](https://github.com/Zentomo)              |Systems Engineer — LangGraph Orchestrator & Infrastructure|
|Meghana      |[@Meghana](https://github.com/Meghana)              |TBD                                                       |

-----

## File Structure

```
vision-link-ai-agent/
├── .github/
│   └── workflows/
│       └── ci.yml
├── workspace/
│   ├── state_schema.py
│   ├── orchestrator.py
│   ├── model_loader.py
│   └── crew_agents.py
├── .env.example
├── requirements.txt
└── README.md
```

-----

*Vision-Link AI HUB · AI Agent Olympics · Lablab.ai · May 2026*
