"""
api/query_service.py — Natural-language query API (handbook §5 four-agent pipeline).

Flow:
  question → Supervisor → Planner → Doer → Critic
  (all hackathon intents: catalog_match, lifecycle scenarios 1–6, open-ended graph queries)
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib import import_module
from typing import Any

from agents.dim_match import normalize_query_dims
from agents.lifecycle_doer import SCENARIO_QUESTIONS
from agents.pipeline_trace import PipelineTracer
from config import AGENT_USE_LLM


_pipeline = import_module("04_agent_pipeline")

_DISPLAY_LIMIT_LABELS = {
    "scenario3_risk_rank": "at-risk GlobalSKUs",
    "scenario5_shared_sku": "shared GlobalSKUs",
    "scenario6_auto_map": "wrong auto-map SKUs",
    "dynamic_rank": "GlobalSKUs by anomaly score",
    "catalog_match": "catalog match candidates",
    "duplicate_upc": "duplicate UPC groups",
    "duplicate_brand_package": "duplicate brand+package groups",
}


def _build_display_limits(
    *,
    duplicate_report: dict | None,
    result_meta: dict | None,
    match_result: dict | None,
) -> list[dict[str, int | str]]:
    limits: list[dict[str, int | str]] = []

    if result_meta:
        for key, counts in result_meta.items():
            shown = int(counts.get("shown") or 0)
            total = int(counts.get("total") or 0)
            if total > shown:
                limits.append({
                    "key": key,
                    "label": _DISPLAY_LIMIT_LABELS.get(key, key.replace("_", " ")),
                    "shown": shown,
                    "total": total,
                })

    if duplicate_report:
        for kind, total_key, groups_key in (
            ("duplicate_upc", "upc_groups_total", "upc_duplicate_groups"),
            (
                "duplicate_brand_package",
                "brand_package_groups_total",
                "brand_package_duplicate_groups",
            ),
        ):
            total = int(duplicate_report.get(total_key) or 0)
            shown = len(duplicate_report.get(groups_key) or [])
            if total > shown:
                limits.append({
                    "key": kind,
                    "label": _DISPLAY_LIMIT_LABELS[kind],
                    "shown": shown,
                    "total": total,
                })

    if match_result:
        pipeline = match_result.get("pipeline") or {}
        total = int(pipeline.get("total_candidates") or 0)
        shown = len(match_result.get("matched_skus") or [])
        if total > shown:
            limits.append({
                "key": "catalog_match",
                "label": _DISPLAY_LIMIT_LABELS["catalog_match"],
                "shown": shown,
                "total": total,
            })

    return limits


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


def run_nl_query(
    question: str,
    anchor_sku: str | None = None,
    scenario: int | None = None,
    query_dims: dict | None = None,
    on_event=None,
) -> dict:
    """
    Execute a natural-language question through the agent pipeline.

    All routing is handled by the Supervisor (heuristic + Bedrock).
    Optional `scenario` forces a hackathon demo scenario 1–6.
    Optional `on_event` receives PipelineEvent objects for live UI streaming.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    if scenario is not None and scenario not in SCENARIO_QUESTIONS:
        raise ValueError(f"scenario must be 1–6, got {scenario}")

    with PipelineTracer(on_event=on_event) as tracer:
        result = _pipeline.run_pipeline(
            question,
            use_llm=AGENT_USE_LLM,
            anchor_sku=anchor_sku,
            forced_scenario=scenario,
            extra_query_dims=normalize_query_dims(query_dims),
        )

    scenario_num = result.spec.scenario_num if result.spec.scenario_num else scenario
    best = result.critic_result.best()
    closed_world = getattr(result, "_closed_world_rows", None)
    match_result = getattr(result, "_match_result", None)
    duplicate_report = getattr(result, "_duplicate_report", None)
    result_meta = getattr(result, "_result_meta", None)

    payload: dict[str, Any] = {
        "question":           result.question,
        "latency_seconds":    result.latency_seconds,
        "scenario":           scenario_num,
        "task_type":          result.spec.task_type,
        "spec":               _serialize(result.spec),
        "planner_rationale":  result.tasks.llm_rationale,
        "tasks":              _serialize(result.tasks.tasks),
        "candidates_count":   len(result.candidates),
        "acceptance_rate":    result.critic_result.acceptance_rate,
        "validated_chains":   _serialize(result.critic_result.validated),
        "rejected_count":     len(result.critic_result.rejected),
        "summary":            result.summary(),
        "pipeline_events":    [e.to_dict() for e in tracer.events],
    }

    if result.spec.task_type == "catalog_match":
        payload["catalog_query"] = {
            "brand_name":   result.spec.brand_name,
            "package_type": result.spec.package_type,
            "query_dims": normalize_query_dims(result.spec.query_dims),
        }
        if match_result:
            payload["match_result"] = match_result
            if match_result.get("product_risk"):
                payload["product_risk"] = match_result["product_risk"]

    if result.spec.task_type == "catalog_duplicate":
        if duplicate_report:
            payload["duplicate_report"] = duplicate_report

    if closed_world is not None:
        payload["closed_world_rows"] = closed_world
        payload["reflexive_finding"] = (
            "Reflexive KG found brand-mismatch evidence"
            if result.critic_result.validated
            else "No validated reflexive chain"
        )

    if best:
        from agents.critic import CriticAgent
        payload["best_classification"] = CriticAgent().classify(best)
        payload["best_confidence"] = best.confidence
        payload["best_reasoning"] = best.reasoning
        payload["best_chain"] = _serialize(best)

    doer_by_id = {
        c.chain_id: c.llm_summary
        for c in result.candidates
        if getattr(c, "llm_summary", "")
    }
    if best and best.chain_id in doer_by_id:
        payload["doer_summary"] = doer_by_id[best.chain_id]
    elif doer_by_id:
        payload["doer_summary"] = next(iter(doer_by_id.values()))

    if result.candidates:
        payload["candidate_summaries"] = [
            {"chain_id": c.chain_id, "summary": c.llm_summary or ""}
            for c in result.candidates[:5]
        ]

    if not best and scenario_num is not None:
        payload["hint"] = (
            "No chain passed the Critic threshold. "
            "Run `python 05_synthesize_lifecycle.py --cohort 300` and retry."
        )

    payload["display_limits"] = _build_display_limits(
        duplicate_report=duplicate_report,
        result_meta=result_meta,
        match_result=match_result,
    )

    return payload
