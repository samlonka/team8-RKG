"""
agents/models.py — Shared data models used across all four agents.

Flow:
  NL question
    → Supervisor  → QuerySpec
    → Planner     → TaskList  (list[QueryTask])
    → Doer        → list[CandidateChain]
    → Critic      → list[ValidatedChain]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

TASK_TYPES = ("root_cause", "risk_rank", "anomaly_explain", "catalog_match", "catalog_duplicate")

@dataclass
class QuerySpec:
    """
    Structured extraction of a natural-language question.
    Produced by the Supervisor agent.
    """
    question:          str                    # original NL question
    task_type:         str                    # root_cause | risk_rank | anomaly_explain | catalog_match
    entity_types:      list[str]              # e.g. ["GlobalSKU", "Brand"]
    anchor_label:      Optional[str]  = None  # "GlobalSKU" | "Brand" | ...
    anchor_entity_id:  Optional[str]  = None  # specific ID extracted from question
    time_window:       Optional[dict] = None  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    traversal_depth:   int            = 3     # how many hops the Planner should build
    brand_name:        Optional[str]  = None  # catalog_match — query brand
    package_type:      Optional[str]  = None  # catalog_match — query package descriptor
    query_dims:        dict           = field(default_factory=dict)  # weight/height/length/width hints
    scenario_num:      Optional[int]  = None  # hackathon demo scenario 1–6 when matched

    def __post_init__(self):
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got '{self.task_type}'")
        if self.scenario_num is not None and self.scenario_num not in range(1, 7):
            raise ValueError(f"scenario_num must be 1–6, got {self.scenario_num}")
        if self.task_type == "catalog_match":
            if not (self.brand_name and self.package_type):
                raise ValueError("catalog_match requires brand_name and package_type")


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

QUERY_TASK_TYPES = (
    "cypher_traverse",  # Graph traversal via Cypher
    "ann_self",         # ANN search on self_emb (semantic similarity)
    "ann_reflect",      # ANN search on reflect_emb (neighbourhood context)
    "anomaly_rank",     # Rank all entities by anomaly score
    "master_match",     # Match brand+package against master GlobalSKU catalog
    "lifecycle_cypher", # Hackathon demo scenario traversal (agents/lifecycle_doer.py)
    "graph_exact",      # Exact match on SKU/UPC/brand/package from user question
    "graph_fuzzy",      # Fuzzy brand/package graph match
    "graph_semantic",   # ANN semantic search from embedded question text
    "master_duplicate_check",  # Scan Postgres/Neo4j for duplicate master SKUs
)

@dataclass
class QueryTask:
    """
    A single executable step produced by the Planner.
    The Doer executes each task in order.
    """
    step:         int
    task_type:    str              # one of QUERY_TASK_TYPES
    label:        str              # Neo4j node label to operate on
    description:  str = ""        # human-readable description for logging

    # Cypher task fields
    cypher:       Optional[str]  = None   # Cypher query template
    cypher_params: dict          = field(default_factory=dict)

    # ANN task fields
    anchor_id:    Optional[str]  = None   # entity_id to use as ANN query vector
    index_name:   Optional[str]  = None   # Neo4j vector index name
    top_k:        int            = 20

    # Which embedding to fetch from anchor
    use_self_emb:    bool = False
    use_reflect_emb: bool = False

    # master_match task fields
    brand_name:   Optional[str] = None
    package_type: Optional[str] = None
    query_dims:   dict          = field(default_factory=dict)

    # lifecycle_cypher task fields
    scenario_num: Optional[int] = None

    # graph search task fields (exact / fuzzy / semantic)
    search_mode:   Optional[str] = None   # exact | fuzzy | semantic
    search_query:  Optional[str] = None   # original NL question
    search_terms:  dict           = field(default_factory=dict)


@dataclass
class TaskList:
    """Ordered list of QueryTask objects. Produced by the Planner."""
    spec:  QuerySpec
    tasks: list[QueryTask] = field(default_factory=list)
    llm_rationale: str = ""   # Bedrock summary of the planned retrieval strategy


# ─────────────────────────────────────────────────────────────────────────────
# DOER INPUT / OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityNode:
    """
    A single node in a candidate causal chain.
    Carries all properties needed for Critic validation.
    """
    entity_id:     str
    label:         str            # Neo4j node label
    display_name:  str            # human-readable (brand_family, product_id, etc.)
    properties:    dict = field(default_factory=dict)
    anomaly_score: Optional[float] = None   # 1 - cosine(self_emb, reflect_emb)
    timestamp:     Optional[str]  = None    # creation_date or scan_timestamp
    source:        str = "unknown"          # "cypher" | "ann_self" | "ann_reflect"

    def has_timestamp(self) -> bool:
        return self.timestamp is not None and self.timestamp != ""


@dataclass
class CandidateChain:
    """
    A raw causal chain assembled by the Doer from graph + ANN results.
    Not yet validated — the Critic will score and accept/reject it.
    """
    chain_id:    str
    path:        list[EntityNode]     # ordered entity path (root → leaf)
    source:      str                  # "cypher" | "ann_self" | "ann_reflect" | "union"
    hop_count:   int = 0
    llm_summary: str = ""             # Doer LLM interpretation of this chain

    def __post_init__(self):
        self.hop_count = len(self.path)


# ─────────────────────────────────────────────────────────────────────────────
# CRITIC OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidatedChain:
    """
    A causal chain that passed Critic validation.
    Includes per-hop reasoning and final confidence score.
    """
    chain_id:          str
    path:              list[EntityNode]
    confidence:        float          # composite score [0, 1]
    temporal_validity: float          # fraction of hops with valid timestamp ordering
    evidence_density:  float          # fraction of hops meeting min entity count
    avg_anomaly_score: float          # mean anomaly score across path entities
    reasoning:         str            # human-readable explanation of the chain
    source:            str


@dataclass
class RejectedChain:
    """A chain the Critic discarded, with the reason."""
    chain_id:   str
    reason:     str
    confidence: float


@dataclass
class CriticResult:
    """Full Critic output: top validated chains + rejection log."""
    validated:  list[ValidatedChain]   # top-N chains above threshold
    rejected:   list[RejectedChain]    # chains that did not pass
    acceptance_rate: float             # fraction of candidates accepted

    def best(self) -> Optional[ValidatedChain]:
        return self.validated[0] if self.validated else None


# ─────────────────────────────────────────────────────────────────────────────
# FINAL PIPELINE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    End-to-end result returned to the caller.
    Wraps all intermediate outputs for traceability.
    """
    question:         str
    spec:             QuerySpec
    tasks:            TaskList
    candidates:       list[CandidateChain]
    critic_result:    CriticResult
    latency_seconds:  float = 0.0

    def summary(self) -> str:
        best = self.critic_result.best()
        if not best:
            return (
                f"No validated chains found for: '{self.question}'\n"
                f"Candidates: {len(self.candidates)} | Accepted: 0 | "
                f"Acceptance rate: {self.critic_result.acceptance_rate:.0%}"
            )
        if self.spec.task_type == "catalog_match":
            top = best.path[0] if best.path else None
            sku = top.entity_id if top else "—"
            status = (top.properties.get("match_status") if top else None) or "—"
            return (
                f"Catalog lookup: '{self.question}'\n"
                f"Brand: {self.spec.brand_name} | Package: {self.spec.package_type}\n"
                f"Best match: GlobalSKU {sku} | status={status} | confidence={best.confidence:.3f}\n"
                f"Reasoning: {best.reasoning}"
            )
        if self.spec.task_type == "catalog_duplicate":
            return (
                f"Master duplicate scan: '{self.question}'\n"
                f"Entities in report: {len(best.path)} | confidence={best.confidence:.3f}\n"
                f"Reasoning: {best.reasoning}"
            )
        return (
            f"Query: '{self.question}'\n"
            f"Task type: {self.spec.task_type} | Depth: {self.spec.traversal_depth}\n"
            f"Candidates: {len(self.candidates)} | "
            f"Accepted: {len(self.critic_result.validated)} | "
            f"Acceptance rate: {self.critic_result.acceptance_rate:.0%}\n"
            f"Best chain confidence: {best.confidence:.3f} | "
            f"Avg anomaly: {best.avg_anomaly_score:.3f}\n"
            f"Reasoning: {best.reasoning}"
        )
