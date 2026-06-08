"""
agents/pipeline_trace.py — Live pipeline trace events for UI streaming.

Agents emit user-facing step updates (Supervisor → Planner → Doer → Critic)
via a context-scoped tracer so the API can stream progress to the frontend.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

_tracer: ContextVar["PipelineTracer | None"] = ContextVar("pipeline_tracer", default=None)


@dataclass
class PipelineEvent:
    id: str
    phase: str
    status: str
    title: str
    detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineTracer:
    """Collects pipeline events; optionally forwards each to a callback (SSE)."""

    def __init__(self, on_event: Callable[[PipelineEvent], None] | None = None):
        self.on_event = on_event
        self.events: list[PipelineEvent] = []
        self._counter = 0
        self._token = None

    def __enter__(self) -> PipelineTracer:
        self._token = _tracer.set(self)
        return self

    def __exit__(self, *_args) -> None:
        if self._token is not None:
            _tracer.reset(self._token)

    def emit(
        self,
        phase: str,
        status: str,
        title: str,
        detail: str = "",
        **meta: Any,
    ) -> PipelineEvent:
        self._counter += 1
        ev = PipelineEvent(
            id=f"{phase}-{self._counter}",
            phase=phase,
            status=status,
            title=title,
            detail=detail,
            meta=meta,
        )
        self.events.append(ev)
        if self.on_event:
            self.on_event(ev)
        return ev


def get_tracer() -> PipelineTracer | None:
    return _tracer.get()


def trace(
    phase: str,
    status: str,
    title: str,
    detail: str = "",
    **meta: Any,
) -> PipelineEvent | None:
    t = get_tracer()
    if t is None:
        return None
    return t.emit(phase, status, title, detail, **meta)


TASK_TYPE_LABELS: dict[str, str] = {
    "catalog_match": "Master catalog lookup",
    "catalog_duplicate": "Duplicate master scan",
    "root_cause": "Root-cause investigation",
    "risk_rank": "Risk ranking",
    "anomaly_explain": "Anomaly deep-dive",
}

DOER_TASK_LABELS: dict[str, str] = {
    "graph_exact": "Pinpoint search in Neo4j",
    "graph_fuzzy": "Fuzzy brand & package match",
    "graph_semantic": "Semantic vector search",
    "master_match": "Match against master catalog",
    "master_duplicate_check": "Scan for duplicate SKUs",
    "cypher_traverse": "Walk the knowledge graph",
    "ann_self": "Find similar entities (embeddings)",
    "ann_reflect": "Neighborhood context search",
    "anomaly_rank": "Rank by anomaly signals",
    "lifecycle_cypher": "Query cohort graph",
}
