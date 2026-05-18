"""
vision-link-ai-agent / workspace / model_loader.py

HuggingFace Model Loader  —  Zentomo's module
==============================================
Initializes Meta Llama-3-8B-Instruct (or MedLlama fallback) via
HuggingFaceEndpoint for use across all LangGraph nodes and CrewAI agents.

Inference mode: REMOTE — models run on HuggingFace Inference API servers.
No local GPU is required. All inference is billed against the HF account
associated with HF_TOKEN. The HuggingFaceEndpoint class makes authenticated
HTTP requests to the HF Inference API; it does NOT load weights locally.

No quantization is applied at this layer. Precision is managed server-side
by HuggingFace based on the model configuration on the Hub.

HF_TOKEN is read exclusively from the environment / GitHub Actions secret.
It is NEVER hardcoded or logged.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models.llms import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry  —  ordered by preference
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "primary":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "clinical":  "ProbeMedicalYonseiMAILab/medllama3-v20",   # MedLlama fine-tune
    "fallback":  "mistralai/Mistral-7B-Instruct-v0.3",        # lighter fallback
}

# Generation defaults tuned for clinical structured-output tasks
_INFERENCE_DEFAULTS = dict(
    max_new_tokens=512,
    temperature=0.2,          # low = more deterministic JSON outputs
    repetition_penalty=1.1,
    return_full_text=False,
)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_llm(
    model_key: str = "primary",
    task: str = "text-generation",
    temperature: float = 0.2,
    max_new_tokens: int = 512,
) -> BaseLLM:
    """
    Returns a cached HuggingFaceEndpoint LLM instance.

    Args:
        model_key:      Key from MODEL_REGISTRY ('primary', 'clinical', 'fallback')
        task:           HF inference task string
        temperature:    Sampling temperature (lower = more deterministic)
        max_new_tokens: Max tokens to generate

    Returns:
        LangChain-compatible BaseLLM bound to the HF Inference API

    Raises:
        EnvironmentError: if HF_TOKEN is not set
        ValueError:       if model_key is not in MODEL_REGISTRY
    """
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add it to your GitHub Actions secrets or "
            "local .env file. Never hardcode it."
        )

    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_key '{model_key}'. "
            f"Valid options: {list(MODEL_REGISTRY.keys())}"
        )

    repo_id = MODEL_REGISTRY[model_key]
    logger.info(f"[model_loader] Loading model: {repo_id} (key={model_key})")

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task=task,
        huggingfacehub_api_token=hf_token,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=_INFERENCE_DEFAULTS["repetition_penalty"],
        return_full_text=_INFERENCE_DEFAULTS["return_full_text"],
    )

    logger.info(f"[model_loader] Model ready: {repo_id}")
    return llm


def load_clinical_llm() -> BaseLLM:
    """Convenience wrapper — loads the clinical fine-tune for diagnostic tasks."""
    return load_llm(model_key="clinical", temperature=0.1, max_new_tokens=768)


def load_primary_llm() -> BaseLLM:
    """Convenience wrapper — loads Llama-3-8B-Instruct (main pipeline model)."""
    return load_llm(model_key="primary", temperature=0.2, max_new_tokens=512)


def load_fallback_llm() -> BaseLLM:
    """Convenience wrapper — loads Mistral-7B for lightweight state transitions."""
    return load_llm(model_key="fallback", temperature=0.3, max_new_tokens=256)


# ---------------------------------------------------------------------------
# Health check  (called during pipeline startup)
# ---------------------------------------------------------------------------

def health_check(model_key: str = "primary") -> dict:
    """
    Runs a minimal inference probe to confirm the model endpoint is reachable.
    Returns a status dict — does NOT raise on failure (logs instead).
    """
    try:
        llm = load_llm(model_key=model_key)
        probe = llm.invoke("Respond with one word: ready")
        return {"status": "ok", "model": MODEL_REGISTRY[model_key], "probe": probe.strip()}
    except Exception as exc:
        logger.error(f"[model_loader] Health check failed for '{model_key}': {exc}")
        return {"status": "error", "model": MODEL_REGISTRY.get(model_key), "error": str(exc)}


# ---------------------------------------------------------------------------
# LangGraph / orchestrator integration helper
# ---------------------------------------------------------------------------

def get_llm_for_node(node_name: str) -> BaseLLM:
    """
    Routes each LangGraph node to the most appropriate model.

    node_name           → model
    ─────────────────────────────
    clinical            → MedLlama (highest accuracy)
    validation          → Llama-3-8B (balanced)
    localization        → Llama-3-8B (multilingual strong)
    self_healing        → Llama-3-8B (reasoning)
    evolve              → Llama-3-8B (code generation)
    *                   → Llama-3-8B (default)
    """
    routing = {
        "clinical":     load_clinical_llm,
        "validation":   load_primary_llm,
        "localization": load_primary_llm,
        "self_healing": load_primary_llm,
        "evolve":       load_primary_llm,
    }
    loader = routing.get(node_name, load_primary_llm)
    return loader()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    print("Running model loader health checks...\n")
    for key in MODEL_REGISTRY:
        result = health_check(key)
        print(f"[{key}] {json.dumps(result, indent=2)}")
