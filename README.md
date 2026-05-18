# Vision-Link AI Agent 🏥⚡

> **Multi-Agent Autonomous Healthcare Assistant**  
> Built with **CrewAI** + **LangGraph** for intelligent medical diagnosis and validation  
> Orchestrated pipeline with self-healing loop and self-evolving backend
> Built for the [AI Agent Olympics — Lablab.ai](https://lablab.ai)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Core Modules](#core-modules)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Smoke Testing](#smoke-testing)
- [Self-Healing Loop](#self-healing-loop)
- [Self-Evolving Backend](#self-evolving-backend)
- [Models & LLM Strategy](#models--llm-strategy)
- [Project Structure](#project-structure)
- [Team](#team)

---

## Architecture

The Vision-Link AI Agent is a **7-node LangGraph StateGraph** that orchestrates three CrewAI agents through intelligent conditional routing:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Patient Input / Query                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Initialize State  │
                    │  (LangGraph Node 1) │
                    └──────────┬──────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Clinical Data Agent       │
                │   (CrewAI + MedLlama-3)       │
                │   (LangGraph Node 2)        │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Localization Agent        │
                │   (CrewAI + Llama-3-8B)     │
                │   (LangGraph Node 3)        │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Validation Agent          │
                │   (CrewAI + Llama-3-8B)     │
                │   (LangGraph Node 4)        │
                └──────────────┬──────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  Conditional Routing Logic  │
                    │  (LangGraph Edge Conditions)│
                    └──┬──────────────────────┬───┘
                       │ VALIDATION FAILED    │ VALIDATION PASSED
                       │                      │
          ┌────────────▼──┐          ┌────────▼─────────┐
          │ Self-Healing   │          │ Self-Evolving    │
          │ Loop Node      │          │ Backend Node     │
          │(LangGraph 5)   │          │ (LangGraph 6)    │
          └────┬───────────┘          └────────┬─────────┘
               │                               │
               └───────────────┬───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Final Output Node  │
                    │  (LangGraph Node 7) │
                    └──────────┬──────────┘
                               │
                      ┌────────▼────────┐
                      │ Final Response  │
                      └─────────────────┘
```

### Data Flow

1. **State Initialization** → Creates a shared `PipelineState` object (Pydantic v2)
2. **Clinical Agent** → Analyzes patient data using MedLlama-3 fine-tuned model
3. **Localization Agent** → Identifies geographic/contextual information using Llama-3-8B
4. **Validation Agent** → Validates output integrity, schemas, and confidence scores
5. **Conditional Routing** → Checks validation results
   - **FAIL** → Routes to Self-Healing Loop (retry failed agent, max 3 attempts)
   - **PASS** → Routes to Self-Evolving Backend
6. **Self-Healing Loop** → Re-runs only the failed agent, applies LLM patches for soft errors
7. **Self-Evolving Backend** → Generates and races 3 code variants, replaces live process with winner
8. **Final Output** → Returns enriched response to user

---

## Key Features

### Self-Healing Loop

When the Validation Agent detects errors or missing fields:

1. **Identifies the failure source** — determines which agent (Clinical or Localization) produced the error
2. **Targeted re-routing** — re-executes **only that agent**, not the entire pipeline
3. **Retry mechanism** — attempts up to `MAX_RETRIES` (default: 3) with exponential backoff
4. **LLM-assisted patching** — for soft errors (e.g., low confidence scores), uses intelligent prompt engineering to refine outputs
5. **Graceful degradation** — marks run as `failed` only after all retries are exhausted; provides best-effort fallback

**Benefit:** Reduces latency by avoiding redundant pipeline re-runs while maximizing recovery chances.

### Self-Evolving Backend

Every successful validation triggers an autonomous evolution cycle:

1. **Variant generation** — Concurrently generates **3 LLM-optimized code variants** of the orchestration pipeline
2. **Isolated sandboxing** — Each variant compiles and runs in a separate, containerized execution context
3. **Fitness scoring** — Evaluates each variant using:
   - **Latency** (ms) — time to completion
   - **Error rate** — validation failure count
   - **Throughput** — requests processed per second
   - **Formula:** `fitness = (1000 / latency_ms) + (1 - error_rate) × 100 − throughput_penalty`

4. **Atomic hot-swap** — **Winner automatically replaces the running module** via Python hot-reload (no process restarts, no downtime)
5. **Zero human intervention** — Fully autonomous; decisions logged but require no manual approval

**Benefit:** Continuous performance optimization without operational overhead.

### Shared State Schema

A single **Pydantic v2 `PipelineState` class** serves as the source of truth.

- Enforces schema consistency across all agents
- Enables type-safe transitions between nodes
- Simplifies debugging and audit trails

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **HuggingFace account** with valid API token
- **Git** for cloning the repository

### 1. Clone the Repository

```bash
git clone https://github.com/BuhariSalisuAI/vision-link-ai-agent.git
cd vision-link-ai-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example .env file
cp .env.example .env

# Edit .env and add your HuggingFace token
# HF_TOKEN=hf_your_token_here
```

### 4. Run Smoke Test

```bash
cd workspace
python orchestrator.py
```

> **First-run tip:** Set `evolution_enabled=False` in `orchestrator.py` to skip the self-evolving backend and focus on core pipeline validation.

---

## Environment Setup

### Required Variables

| Variable | Required | Type | Source | Example |
|----------|----------|------|--------|---------|
| `HF_TOKEN` | ✅ Yes | String | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | `hf_a1B2c3D4e5F6g7H8i9J0k...` |

### Optional Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EVOLUTION_ENABLED` | `true` | Enable/disable self-evolving backend (`true` or `false`) |
| `MAX_RETRIES` | `3` | Maximum retry attempts in self-healing loop |

### Example `.env` File

```dotenv
# HuggingFace API Token (required)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Feature flags
EVOLUTION_ENABLED=false

# Retry and performance tuning
MAX_RETRIES=3
```

> **Security Note:** Never commit `.env` to version control. Use [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) for CI/CD.

---

## Smoke Testing

The smoke test validates the core pipeline without triggering the self-evolving backend:

### Basic Smoke Test

```bash
cd workspace
python orchestrator.py
```

### Expected Output

```
[INFO] Pipeline initialized...
[INFO] Node: Initialize State
[INFO] Node: Clinical Agent
[CLINICAL OUTPUT] diagnosis: [...], confidence: 0.87
[INFO] Node: Localization Agent
[LOCALIZATION OUTPUT] region: [...], context: [...]
[INFO] Node: Validation Agent
[VALIDATION] ✓ All checks passed
[INFO] Node: Self-Evolving Backend (skipped - evolution_enabled=False)
[INFO] Node: Final Output
[SUCCESS] Pipeline completed in 2341ms
```

### Smoke Test with Evolution (Optional)

To test the full pipeline including self-evolving backend:

```bash
cd workspace
EVOLUTION_ENABLED=true python orchestrator.py
```

### Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `HF_TOKEN not found` | Missing `.env` file or invalid token | Verify `.env` exists and token is valid at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `Model download timeout` | Network issue or token insufficient | Check internet connection; ensure token has read access to gated models |
| `LangGraph compilation error` | Invalid state schema | Run `python -m pydantic validate workspace/state_schema.py` |
| `CUDA out of memory` | Insufficient GPU memory | Reduce batch size or use CPU-only mode (set `device="cpu"` in `model_loader.py`) |

---

## Self-Healing Loop

### How It Works

```
Validation Failed
    │
    ├─→ Identify Failed Node (Clinical or Localization)
    │
    ├─→ Retry Loop (Max 3 attempts):
    │   ├─→ Attempt 1: Re-run agent with same input
    │   ├─→ Attempt 2: Inject corrective prompt guidance
    │   └─→ Attempt 3: Combine outputs with LLM merging
    │
    ├─→ Soft Error Detection:
    │   ├─→ Low confidence? → LLM-based refinement
    │   ├─→ Missing fields? → Intelligent imputation
    │   └─→ Ambiguous output? → Prompt clarification
    │
    └─→ If still failing → Mark failed, return partial result
```

### Configuration

In `workspace/orchestrator.py`:

```python
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
```

### Example Scenario

**Input:** Patient with incomplete symptom data  
**Clinical Agent Output:** `{"diagnosis": "...", "confidence": 0.42}`  
**Validation:** Confidence below threshold → FAIL

**Self-Healing:**
1. Retry 1: Re-run Clinical Agent → Still low confidence
2. Retry 2: Inject prompt: *"Provide the most likely diagnosis with supporting evidence"* → Confidence improves to 0.68
3. Validation: ✓ PASS → Proceed to self-evolving backend

---

## Self-Evolving Backend

### Evolution Cycle

```
Pipeline Validation PASSED
    │
    ├─→ Generate 3 Variants:
    │   ├─→ Variant A: Optimized for latency (streaming outputs)
    │   ├─→ Variant B: Optimized for accuracy (ensemble voting)
    │   └─→ Variant C: Optimized for throughput (batch processing)
    │
    ├─→ Compile & Sandbox:
    │   ├─→ Each variant spun up in isolated Python process
    │   ├─→ Syntax/logic validation performed
    │   └─→ Ready for execution
    │
    ├─→ Race Execution (on same test dataset):
    │   ├─→ Variant A completes in 1200ms, 2 errors, 50 requests/sec
    │   ├─→ Variant B completes in 1800ms, 0 errors, 25 requests/sec
    │   └─→ Variant C completes in 950ms, 1 error, 100 requests/sec
    │
    ├─→ Fitness Scoring:
    │   ├─→ Score = (throughput_bonus) + (accuracy_bonus) − (latency_penalty)
    │   └─→ Variant B wins with highest overall fitness
    │
    └─→ Atomic Hot-Swap:
        ├─→ Replace orchestrator.py with Variant B code
        ├─→ Reload modules in-process
        └─→ No restarts, no downtime
```

### Fitness Formula

```
fitness_score = (1000 / avg_latency_ms)
              + (100 × accuracy_rate)
              − (5 × error_count)
              + (throughput_requests_per_sec / 10)
```

Minimum threshold to replace current: **FITNESS_THRESHOLD** (default: `0.75`)

### Evolution Configuration

In `workspace/orchestrator.py`:

```python
EVOLUTION_ENABLED = os.getenv("EVOLUTION_ENABLED", "false").lower() == "true"
```
---

## Model Selection

| Node | Model | Rationale |
|------|-------|-----------|
| **Clinical Agent** | `ProbeMedicalYonseiMAILab/medllama3-v20` | Specialized medical knowledge; highest diagnostic accuracy via domain fine-tuning |
| **Localization Agent** | `meta-llama/Meta-Llama-3-8B-Instruct` | Strong multilingual support; efficient context understanding |
| **Validation Agent** | `meta-llama/Meta-Llama-3-8B-Instruct` | Reliable structured output generation; excellent for schema validation |
| **Self-Healing Engine** | `meta-llama/Meta-Llama-3-8B-Instruct` | Balanced reasoning capability + inference speed; effective at error analysis |
| **Fallback** | `mistralai/Mistral-7B-Instruct-v0.3` | Lightweight alternative; used if primary models unavailable |

---

## Project Structure

```
vision-link-ai-agent/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD pipeline
├── workspace/
│   ├── state_schema.py               # Pydantic v2 state definitions
│   ├── orchestrator.py               # LangGraph 7-node pipeline
│   ├── model_loader.py               # HuggingFace model initialization
│   └── crew_agents.py                # CrewAI agent definitions
├── .env.example                       # Environment template (copy to .env)
├── .gitignore                         # Git ignore patterns
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Dependencies

Key packages and versions:

```
langgraph>=0.2.28              # LangGraph orchestration framework
langchain-core>=0.3.0          # LangChain core utilities
crewai>=0.80.0                 # CrewAI multi-agent framework
langchain-huggingface>=0.1.0   # HuggingFace LLM integration
huggingface-hub>=0.23.0        # HuggingFace Hub API
transformers>=4.44.0           # Transformer models library
pydantic>=2.7.0                # Data validation (v2)
python-dotenv>=1.0.0           # Environment variable management
```

See `requirements.txt` for complete list.

---

## CI / CD

GitHub Actions workflows run on every push to `main` and `dev` branches:

**Checks performed:**

- **Ruff linting** — Code style validation
- **Pydantic schema validation** — State schema integrity
- **LangGraph compilation check** — Graph topology validation
- **HuggingFace model loader health check** — Model availability verification

**Workflow file:** `.github/workflows/ci.yml`

To view CI status: [Vision-Link AI Agent — Actions](https://github.com/BuhariSalisuAI/vision-link-ai-agent/actions)

---

## Team

| Member            | GitHub                                               | Role |
|-------------------|------------------------------------------------------|------|
| **Buhari Salisu** | [@BuhariSalisuAI](https://github.com/BuhariSalisuAI) | Founder & CEO |
| **Subhalaxmi**    | [@Subhalaxmi](https://github.com/Subhalaxmi)         | Data Engineer — CrewAI Agents & Tasks |
| **Zentomo**       | [@Zentomo](https://github.com/Zentomo)               | Systems Engineer — LangGraph Orchestration & Infrastructure |
| **Ramitha**       | [@Ramitha](https://github.com/Rami2212)              | Documentation   |
| **Meghana**       | [@Meghana](https://github.com/Meghana)               | Frontend Developer |

---

**Vision-Link AI Agent** · AI Agent Olympics · Lablab.ai · May 2026
