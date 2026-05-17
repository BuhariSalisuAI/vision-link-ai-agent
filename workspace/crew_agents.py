"""
vision-link-ai-agent / workspace / crew_agents.py

CrewAI Agents & Tasks  –  Subhalaxmi's module
==============================================
This file defines the three core CrewAI agents and their tasks.
The orchestrator (orchestrator.py) calls the three public functions:
  - run_clinical_agent(patient_dict) -> dict
  - run_localization_agent(clinical_dict, language) -> dict
  - run_validation_agent(clinical_dict, locale_dict) -> dict

Subhalaxmi: fill in the agent backstories, task descriptions,
and tool integrations here. The Pydantic schemas in state_schema.py
define the exact input/output contracts.
"""

from __future__ import annotations
import os
from typing import Any, Dict

from crewai import Agent, Crew, Process, Task
from crewai_tools import BaseTool
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# LLM config  (Meta Llama-3-8B-Instruct via HuggingFace Inference API)
# HF_TOKEN is read from environment / GitHub Actions secret — never hardcoded
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"

_llm_config = dict(
    model=f"huggingface/{HF_MODEL}",
    api_key=HF_TOKEN,
)


# ---------------------------------------------------------------------------
# Pydantic I/O contracts (mirrors state_schema.py — kept here for CrewAI
# structured output integration)
# ---------------------------------------------------------------------------

class ClinicalAgentOutput(BaseModel):
    diagnosis_candidates: list[str]
    urgency_level: str
    recommended_actions: list[str]
    confidence_score: float
    raw_llm_response: str = ""


class LocalizationAgentOutput(BaseModel):
    translated_diagnosis: list[str]
    translated_actions: list[str]
    locale_notes: str = ""
    target_language: str


class ValidationAgentOutput(BaseModel):
    passed: bool
    errors: list[str]
    warnings: list[str]
    missing_fields: list[str]
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Custom tools (extend as needed)
# ---------------------------------------------------------------------------

class MedicalKnowledgeTool(BaseTool):
    """
    Placeholder for a medical knowledge base lookup.
    TODO (Subhalaxmi): replace with actual vector-store retrieval.
    """
    name: str = "medical_knowledge_lookup"
    description: str = "Look up symptoms, conditions, and treatment guidelines."

    def _run(self, query: str) -> str:
        # Stub — returns a formatted prompt hint
        return f"[MedDB] Query '{query}': Refer to WHO ICD-11 guidelines."


class LocaleResourceTool(BaseTool):
    """
    Placeholder for locale/language resource lookup.
    TODO (Subhalaxmi): connect to a translation API or local glossary.
    """
    name: str = "locale_resource_lookup"
    description: str = "Fetch locale-specific medical terminology and translation resources."

    def _run(self, language_code: str) -> str:
        return f"[Locale] Resources for '{language_code}': WHO terminology available."


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

def _make_clinical_agent() -> Agent:
    return Agent(
        role="Clinical Data Agent",
        goal=(
            "Analyze patient symptoms, medical history, and context to produce "
            "a ranked list of diagnosis candidates with urgency levels and "
            "recommended clinical actions."
        ),
        backstory=(
            "You are a board-certified physician AI with 20 years of virtual clinical "
            "experience across sub-Saharan Africa and Southeast Asia. You specialize "
            "in resource-constrained primary care and triage."
        ),
        tools=[MedicalKnowledgeTool()],
        llm=_llm_config,
        verbose=True,
        allow_delegation=False,
        output_pydantic=ClinicalAgentOutput,
    )


def _make_localization_agent() -> Agent:
    return Agent(
        role="Localization Agent",
        goal=(
            "Translate and culturally adapt clinical outputs to the patient's "
            "language and region, ensuring medical accuracy is preserved."
        ),
        backstory=(
            "You are a multilingual medical translator with deep expertise in "
            "community health communication across 40+ languages. You ensure "
            "that translated content is both medically precise and culturally appropriate."
        ),
        tools=[LocaleResourceTool()],
        llm=_llm_config,
        verbose=True,
        allow_delegation=False,
        output_pydantic=LocalizationAgentOutput,
    )


def _make_validation_agent() -> Agent:
    return Agent(
        role="Validation Agent",
        goal=(
            "Validate all upstream agent outputs for completeness, schema "
            "conformance, clinical plausibility, and absence of missing fields. "
            "Flag any issues clearly so the self-healing loop can act on them."
        ),
        backstory=(
            "You are a clinical QA specialist and AI output auditor. You have "
            "reviewed thousands of AI-generated medical reports and know exactly "
            "what separates a safe, actionable output from a dangerous hallucination."
        ),
        tools=[],
        llm=_llm_config,
        verbose=True,
        allow_delegation=False,
        output_pydantic=ValidationAgentOutput,
    )


# ---------------------------------------------------------------------------
# Public runner functions (called by orchestrator.py)
# ---------------------------------------------------------------------------

def run_clinical_agent(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    agent = _make_clinical_agent()
    task = Task(
        description=(
            f"Analyze the following patient data and produce a structured "
            f"clinical assessment:\n\n{patient_dict}\n\n"
            "Return: diagnosis_candidates, urgency_level, recommended_actions, confidence_score."
        ),
        expected_output="A JSON-serializable ClinicalAgentOutput object.",
        agent=agent,
        output_pydantic=ClinicalAgentOutput,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    # CrewAI returns the pydantic model in result.pydantic when output_pydantic is set
    if hasattr(result, "pydantic") and result.pydantic:
        return result.pydantic.model_dump()
    # Fallback: parse raw string
    import json
    return json.loads(result.raw)


def run_localization_agent(clinical_dict: Dict[str, Any], language: str) -> Dict[str, Any]:
    agent = _make_localization_agent()
    task = Task(
        description=(
            f"Translate and localize the following clinical output to language '{language}':\n\n"
            f"{clinical_dict}\n\n"
            "Return: translated_diagnosis, translated_actions, locale_notes, target_language."
        ),
        expected_output="A JSON-serializable LocalizationAgentOutput object.",
        agent=agent,
        output_pydantic=LocalizationAgentOutput,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    if hasattr(result, "pydantic") and result.pydantic:
        return result.pydantic.model_dump()
    import json
    return json.loads(result.raw)


def run_validation_agent(
    clinical_dict: Dict[str, Any],
    locale_dict: Dict[str, Any],
) -> Dict[str, Any]:
    agent = _make_validation_agent()
    task = Task(
        description=(
            "Validate the following agent outputs for correctness and completeness:\n\n"
            f"Clinical Output:\n{clinical_dict}\n\n"
            f"Localization Output:\n{locale_dict}\n\n"
            "Check: all required fields present, urgency_level is valid, "
            "confidence_score >= 0.4, no empty lists, translation matches source count. "
            "Return: passed (bool), errors (list), warnings (list), missing_fields (list)."
        ),
        expected_output="A JSON-serializable ValidationAgentOutput object.",
        agent=agent,
        output_pydantic=ValidationAgentOutput,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    if hasattr(result, "pydantic") and result.pydantic:
        return result.pydantic.model_dump()
    import json
    return json.loads(result.raw)
