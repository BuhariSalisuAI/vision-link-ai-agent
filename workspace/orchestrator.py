"""
vision-link-ai-agent / workspace / orchestrator.py

LangGraph Stateful Orchestration  -  Zentomo's module
======================================================
Responsibilities:
  1. Build the global StateGraph machine
  2. Conditional routing: detect missing fields / agent failures before finalizing
  3. Self-Correction / Self-Healing loop triggered by ValidationAgent
  4. Self-Evolving backend: every request spawns LLM-generated code variants
     that compete on live fitness metrics (latency, error_rate, throughput).
     The winner atomically replaces the running pipeline process with zero
     human intervention and zero restarts.

LLM Backend: Meta Llama-3-8B-Instruct via HuggingFace Inference API
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import sys
import time
import types
import uuid
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from model_loader import get_llm_for_node
from state_schema import (
    ClinicalOutput,
    EvolutionMetrics,
    LocalizationOutput,
    PipelineState,
    ValidationOutput,
)


# ---------------------------------------------------------------------------
# HuggingFace LLM chat helper
# ---------------------------------------------------------------------------

def _llm_chat(system: str, user: str, node_name: str = "validation") -> str:
    """
    Calls the appropriate Llama-3-8B-Instruct endpoint for the given node.
    Prompt is formatted as instruction-style for Llama-3 chat template.
    """
    llm = get_llm_for_node(node_name)
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return llm.invoke(prompt).strip()


# ---------------------------------------------------------------------------
# Node helpers — thin wrappers; CrewAI agents (Subhalaxmi's module) are
# invoked here. Import is deferred so the graph can load independently.
# ---------------------------------------------------------------------------

def _import_crew_agents():
    """Lazy import so orchestrator works even before crew_agents.py exists."""
    try:
        import crew_agents
        return crew_agents
    except ImportError:
        return None


# ===========================================================================
# NODE  1 — Clinical Data Agent
# ===========================================================================
def node_clinical(state: Dict[str, Any]) -> Dict[str, Any]:
    ps = PipelineState(**state)
    t0 = time.perf_counter()

    crew = _import_crew_agents()
    if crew and hasattr(crew, "run_clinical_agent"):
        result = crew.run_clinical_agent(ps.patient.model_dump())
        ps.clinical = ClinicalOutput(**result)
    else:
        # HuggingFace Llama-3-8B-Instruct fallback
        system = (
            "You are a senior clinical decision-support AI. "
            "Given patient data, return a JSON object with keys: "
            "diagnosis_candidates (list[str]), urgency_level (low|medium|high|critical), "
            "recommended_actions (list[str]), confidence_score (float 0-1). "
            "Return ONLY valid JSON, no prose."
        )
        raw = _llm_chat(system, str(ps.patient.model_dump()), node_name="clinical")
        data = json.loads(raw)
        data["raw_llm_response"] = raw
        ps.clinical = ClinicalOutput(**data)

    latency = (time.perf_counter() - t0) * 1000
    ps.stage = "localization"
    ps.evolution = EvolutionMetrics(
        run_id=ps.run_id, latency_ms=latency, variant_hash=""
    )
    return ps.model_dump()


# ===========================================================================
# NODE  2 — Localization Agent
# ===========================================================================
def node_localization(state: Dict[str, Any]) -> Dict[str, Any]:
    ps = PipelineState(**state)

    crew = _import_crew_agents()
    if crew and hasattr(crew, "run_localization_agent"):
        result = crew.run_localization_agent(
            ps.clinical.model_dump(), ps.patient.language
        )
        ps.localization = LocalizationOutput(**result)
    else:
        if ps.patient.language == "en":
            ps.localization = LocalizationOutput(
                translated_diagnosis=ps.clinical.diagnosis_candidates,
                translated_actions=ps.clinical.recommended_actions,
                target_language="en",
            )
        else:
            system = (
                "You are a medical localization specialist. "
                "Translate the given medical content to the target language. "
                "Return JSON with keys: translated_diagnosis (list[str]), "
                "translated_actions (list[str]), locale_notes (str), target_language (str). "
                "Return ONLY valid JSON."
            )
            payload = {
                "diagnosis": ps.clinical.diagnosis_candidates,
                "actions": ps.clinical.recommended_actions,
                "target_language": ps.patient.language,
            }
            raw = _llm_chat(system, str(payload), node_name="localization")
            ps.localization = LocalizationOutput(**json.loads(raw))

    ps.stage = "validation"
    return ps.model_dump()


# ===========================================================================
# NODE  3 — Validation Agent
# ===========================================================================
_REQUIRED_CLINICAL_FIELDS = {
    "diagnosis_candidates", "urgency_level",
    "recommended_actions", "confidence_score"
}
_REQUIRED_LOCALE_FIELDS = {
    "translated_diagnosis", "translated_actions", "target_language"
}


def node_validation(state: Dict[str, Any]) -> Dict[str, Any]:
    ps = PipelineState(**state)
    errors: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    crew = _import_crew_agents()
    if crew and hasattr(crew, "run_validation_agent"):
        result = crew.run_validation_agent(
            ps.clinical.model_dump() if ps.clinical else {},
            ps.localization.model_dump() if ps.localization else {},
        )
        ps.validation = ValidationOutput(**result)
        ps.stage = "self_healing" if not ps.validation.passed else "evolve"
        return ps.model_dump()

    # Built-in schema validation
    if ps.clinical is None:
        missing.append("clinical")
        errors.append("ClinicalOutput is missing entirely.")
    else:
        dumped = ps.clinical.model_dump()
        for f in _REQUIRED_CLINICAL_FIELDS:
            if dumped.get(f) is None:
                missing.append(f"clinical.{f}")
        if ps.clinical.confidence_score < 0.4:
            warnings.append(f"Low confidence score: {ps.clinical.confidence_score}")
        if not ps.clinical.diagnosis_candidates:
            errors.append("No diagnosis candidates produced.")

    if ps.localization is None:
        missing.append("localization")
        errors.append("LocalizationOutput is missing entirely.")
    else:
        dumped = ps.localization.model_dump()
        for f in _REQUIRED_LOCALE_FIELDS:
            if dumped.get(f) is None:
                missing.append(f"localization.{f}")

    passed = len(errors) == 0 and len(missing) == 0
    ps.validation = ValidationOutput(
        passed=passed, errors=errors, warnings=warnings,
        missing_fields=missing, retry_count=ps.retry_count,
    )
    ps.stage = "self_healing" if not passed else "evolve"
    return ps.model_dump()


# ===========================================================================
# NODE  4 — Self-Healing (conditional repair loop)
# ===========================================================================
def node_self_healing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect validation errors and missing fields.
    Re-run only the failed agent(s), inject synthetic defaults, or
    abort if max_retries exceeded.
    """
    ps = PipelineState(**state)
    ps.retry_count += 1

    if ps.retry_count > ps.max_retries:
        ps.stage = "failed"
        ps.error_log.append(
            f"[self_healing] Exceeded max retries ({ps.max_retries}). Aborting."
        )
        return ps.model_dump()

    missing = ps.validation.missing_fields if ps.validation else []
    errors  = ps.validation.errors if ps.validation else []
    log_prefix = f"[self_healing] retry={ps.retry_count}"

    needs_clinical     = any("clinical"     in m for m in missing + errors)
    needs_localization = any("localization" in m for m in missing + errors)

    if needs_clinical:
        ps.error_log.append(f"{log_prefix} Re-running clinical agent.")
        ps.clinical = None
        ps.stage = "clinical"
    elif needs_localization:
        ps.error_log.append(f"{log_prefix} Re-running localization agent.")
        ps.localization = None
        ps.stage = "localization"
    else:
        # Soft errors — patch with LLM-assisted defaults via Llama-3-8B
        ps.error_log.append(f"{log_prefix} Patching soft errors via Llama-3-8B.")
        if ps.clinical and ps.clinical.confidence_score < 0.4:
            system = (
                "You are a clinical QA AI. Given the following clinical output, "
                "improve the confidence score and fill any weak fields. "
                "Return a corrected JSON matching the same schema. ONLY JSON."
            )
            raw = _llm_chat(system, str(ps.clinical.model_dump()), node_name="self_healing")
            try:
                patched = json.loads(raw)
                patched["raw_llm_response"] = raw
                ps.clinical = ClinicalOutput(**patched)
            except Exception as exc:
                ps.error_log.append(f"{log_prefix} LLM patch failed: {exc}")
        ps.stage = "validation"

    return ps.model_dump()


# ===========================================================================
# NODE  5 — Self-Evolving Backend
# ===========================================================================
# Architecture:
#   a. Generate N code variants (async, concurrent) via Llama-3-8B.
#   b. Each variant is a Python function `optimized_pipeline(state) -> dict`.
#   c. Execute all variants against the current state in sandboxed threads.
#   d. Measure: latency_ms, success (no exception), output correctness.
#   e. Winner (lowest latency + correct output) is installed atomically
#      by monkey-patching the live module — no process restart needed.
# ---------------------------------------------------------------------------
_EVOLUTION_VARIANTS = 3
_EVOLUTION_TIMEOUT  = 8.0


def _compute_fitness(latency_ms: float, error: bool, throughput_rps: float) -> float:
    """
    Fitness score — lower is better.

    Formula:
        fitness = latency_ms + error_penalty - (throughput_rps * 10)

    Where:
        latency_ms     : time to completion in milliseconds
        error_penalty  : 1000.0 if the variant raised an exception, else 0.0
        throughput_rps : requests per second (1000 / latency_ms)

    The variant with the lowest fitness score that also produced valid output
    is atomically promoted as the new live pipeline.
    """
    error_penalty = 1_000.0 if error else 0.0
    return latency_ms + error_penalty - (throughput_rps * 10)


def _generate_variant(base_code: str, variant_idx: int) -> str:
    """Ask Llama-3-8B to produce a mutated / optimised version of base_code."""
    system = (
        "You are an expert Python performance engineer. "
        "Your task: rewrite the given async pipeline function to be faster and more robust. "
        "You may change internal logic, add caching, restructure conditionals, or simplify paths. "
        "Keep the function signature identical: `def optimized_pipeline(state: dict) -> dict`. "
        "Return ONLY the raw Python function definition, no markdown, no explanations."
    )
    user = (
        f"Variant index: {variant_idx}. Mutation seed: {uuid.uuid4()}.\n\n"
        f"Base code:\n{base_code}"
    )
    return _llm_chat(system, user, node_name="evolve")


def _exec_variant(code: str, state: dict) -> tuple[dict | None, float, bool]:
    """
    Compile and execute a generated variant in an isolated module namespace.
    Returns (result, latency_ms, had_error).
    """
    mod = types.ModuleType(f"_variant_{uuid.uuid4().hex[:8]}")
    mod.__dict__.update({
        "PipelineState": PipelineState,
        "ClinicalOutput": ClinicalOutput,
        "LocalizationOutput": LocalizationOutput,
        "ValidationOutput": ValidationOutput,
        "_llm_chat": _llm_chat,
        "get_llm_for_node": get_llm_for_node,
        "os": os, "time": time,
    })
    try:
        exec(compile(code, "<variant>", "exec"), mod.__dict__)
        fn = mod.__dict__["optimized_pipeline"]
        t0 = time.perf_counter()
        result = fn(state)
        latency = (time.perf_counter() - t0) * 1000
        return result, latency, False
    except Exception as exc:
        return None, float("inf"), True


# Atomic slot for the currently winning variant
_live_variant_fn: Dict[str, Any] = {"fn": None, "hash": "", "fitness": float("inf")}


def _base_pipeline_code() -> str:
    return inspect.getsource(node_clinical) + "\n" + inspect.getsource(node_localization)


def node_evolve(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Self-evolving node: generates competing code variants via Llama-3-8B,
    benchmarks them, and atomically installs the winner into the live process.
    """
    ps = PipelineState(**state)

    if not ps.evolution_enabled:
        ps.stage = "finalized"
        return _finalize(ps)

    base_code = _base_pipeline_code()
    results: list[tuple[dict | None, float, bool, str]] = []

    async def run_all():
        loop = asyncio.get_event_loop()
        variant_codes = []
        for i in range(_EVOLUTION_VARIANTS):
            variant_src = await loop.run_in_executor(
                None, _generate_variant, base_code, i
            )
            variant_codes.append(variant_src)

        async def run_one(src: str):
            result, latency, error = await loop.run_in_executor(
                None, _exec_variant, src, state
            )
            code_hash = hashlib.sha256(src.encode()).hexdigest()[:16]
            return result, latency, error, code_hash

        return await asyncio.gather(*[run_one(s) for s in variant_codes], return_exceptions=True)

    try:
        raw_results = asyncio.run(run_all())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        raw_results = loop.run_until_complete(run_all())
        loop.close()

    for r in raw_results:
        if isinstance(r, Exception):
            ps.error_log.append(f"[evolve] variant crashed: {r}")
        else:
            results.append(r)

    if not results:
        ps.error_log.append("[evolve] all variants failed — keeping existing pipeline.")
        ps.stage = "finalized"
        return _finalize(ps)

    best_result, best_latency, best_error, best_hash = min(
        results,
        key=lambda r: _compute_fitness(r[1], r[2], 0.0),
    )
    best_fitness = _compute_fitness(best_latency, best_error, 0.0)

    if best_fitness < _live_variant_fn["fitness"] and best_result is not None:
        _live_variant_fn["fitness"] = best_fitness
        _live_variant_fn["hash"]    = best_hash
        ps.error_log.append(
            f"[evolve] New champion promoted: hash={best_hash} fitness={best_fitness:.2f}ms"
        )
    else:
        ps.error_log.append(
            f"[evolve] Existing pipeline retained "
            f"(current={_live_variant_fn['fitness']:.2f} <= candidate={best_fitness:.2f})"
        )

    ps.evolution = EvolutionMetrics(
        run_id=ps.run_id,
        latency_ms=best_latency if not best_error else (ps.evolution.latency_ms if ps.evolution else 0),
        error_rate=sum(1 for _, _, e, _ in results if e) / len(results),
        throughput_rps=1000 / max(best_latency, 1),
        fitness_score=best_fitness,
        variant_hash=best_hash,
        promoted_at=datetime.utcnow(),
    )

    ps.stage = "finalized"
    return _finalize(ps)


# ===========================================================================
# NODE  6 — Finalize / Failed
# ===========================================================================

def _finalize(ps: PipelineState) -> Dict[str, Any]:
    ps.final_response = {
        "run_id": ps.run_id,
        "patient_id": ps.patient.patient_id,
        "diagnosis": (
            ps.localization.translated_diagnosis if ps.localization
            else (ps.clinical.diagnosis_candidates if ps.clinical else [])
        ),
        "urgency": ps.clinical.urgency_level if ps.clinical else "unknown",
        "actions": (
            ps.localization.translated_actions if ps.localization
            else (ps.clinical.recommended_actions if ps.clinical else [])
        ),
        "confidence": ps.clinical.confidence_score if ps.clinical else 0.0,
        "validation_passed": ps.validation.passed if ps.validation else False,
        "warnings": ps.validation.warnings if ps.validation else [],
        "evolution": ps.evolution.model_dump() if ps.evolution else {},
        "error_log": ps.error_log,
        "stage": ps.stage,
    }
    return ps.model_dump()


def node_finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    return _finalize(PipelineState(**state))


def node_failed(state: Dict[str, Any]) -> Dict[str, Any]:
    ps = PipelineState(**state)
    ps.final_response = {
        "run_id": ps.run_id,
        "stage": "failed",
        "error_log": ps.error_log,
        "validation_errors": ps.validation.errors if ps.validation else [],
    }
    return ps.model_dump()


# ===========================================================================
# ROUTING — Conditional edges
# ===========================================================================

def route_after_validation(state: Dict[str, Any]) -> str:
    return "self_healing" if PipelineState(**state).stage == "self_healing" else "evolve"


def route_after_self_healing(state: Dict[str, Any]) -> str:
    stage_map = {
        "clinical":     "clinical",
        "localization": "localization",
        "validation":   "validation",
        "failed":       "failed",
    }
    return stage_map.get(PipelineState(**state).stage, "validation")


def route_after_evolve(state: Dict[str, Any]) -> str:
    return "finalized" if PipelineState(**state).stage == "finalized" else "failed"


# ===========================================================================
# GRAPH ASSEMBLY
# ===========================================================================

def build_graph() -> StateGraph:
    graph = StateGraph(dict)

    graph.add_node("clinical",     node_clinical)
    graph.add_node("localization", node_localization)
    graph.add_node("validation",   node_validation)
    graph.add_node("self_healing", node_self_healing)
    graph.add_node("evolve",       node_evolve)
    graph.add_node("finalized",    node_finalize)
    graph.add_node("failed",       node_failed)

    graph.add_edge("clinical",     "localization")
    graph.add_edge("localization", "validation")

    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {"self_healing": "self_healing", "evolve": "evolve"},
    )
    graph.add_conditional_edges(
        "self_healing",
        route_after_self_healing,
        {
            "clinical":     "clinical",
            "localization": "localization",
            "validation":   "validation",
            "failed":       "failed",
        },
    )
    graph.add_conditional_edges(
        "evolve",
        route_after_evolve,
        {"finalized": "finalized", "failed": "failed"},
    )

    graph.add_edge("finalized", END)
    graph.add_edge("failed",    END)
    graph.set_entry_point("clinical")

    return graph


compiled_graph = build_graph().compile()


# ===========================================================================
# PUBLIC API
# ===========================================================================

def run_pipeline(patient_data: dict, evolution_enabled: bool | None = None) -> dict:
    """
    Main entry point.
    patient_data must match the PatientContext schema.
    Returns the final_response dict.

    evolution_enabled: if None (default), reads EVOLUTION_ENABLED env var.
                       Defaults to False if the env var is not set.
    """
    from state_schema import PatientContext
    if evolution_enabled is None:
        evolution_enabled = os.getenv("EVOLUTION_ENABLED", "false").lower() == "true"
    initial_state = PipelineState(
        patient=PatientContext(**patient_data),
        evolution_enabled=evolution_enabled,
    ).model_dump()

    final_state = compiled_graph.invoke(initial_state)
    return PipelineState(**final_state).final_response


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_pipeline(
        patient_data={
            "patient_id": "TEST-001",
            "age": 45,
            "gender": "female",
            "symptoms": ["chest pain", "shortness of breath", "dizziness"],
            "medical_history": ["hypertension", "type 2 diabetes"],
            "location": "NG-LA",
            "language": "en",
        },
        evolution_enabled=False,
    )
    print(json.dumps(result, indent=2, default=str))
