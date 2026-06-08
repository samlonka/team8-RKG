"""
api/query_service.py — Natural-language query API (handbook §5 four-agent pipeline).

Flow: question → Supervisor → Planner → Doer → Critic
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from importlib import import_module

_pipeline = import_module("04_agent_pipeline")


def _serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return str(obj)


def run_nl_query(question: str, anchor_sku: str | None = None) -> dict:
    """
    Execute a handbook-style NL question through the full agent pipeline.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    result = _pipeline.run_pipeline(question, use_llm=True, anchor_sku=anchor_sku)

    best = result.critic_result.best()
    payload = {
        "question":           result.question,
        "latency_seconds":    result.latency_seconds,
        "spec":               _serialize(result.spec),
        "planner_rationale":  result.tasks.llm_rationale,
        "tasks":              _serialize(result.tasks.tasks),
        "candidates_count":   len(result.candidates),
        "acceptance_rate":    result.critic_result.acceptance_rate,
        "validated_chains":   _serialize(result.critic_result.validated),
        "rejected_count":     len(result.critic_result.rejected),
        "summary":            result.summary(),
    }
    if best:
        from agents.critic import CriticAgent
        payload["best_classification"] = CriticAgent().classify(best)
        payload["best_confidence"] = best.confidence
        payload["best_reasoning"] = best.reasoning
    return payload
