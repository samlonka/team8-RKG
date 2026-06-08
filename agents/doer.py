"""
agents/doer.py — Doer Agent

Responsibilities:
  - Execute each QueryTask from the Planner's TaskList
  - Run: (1) Cypher traversal on Neo4j, (2) ANN on self_emb, (3) ANN on reflect_emb
  - Union all three result sets, deduplicate by entity_id
  - Compute anomaly score per entity (1 - cosine similarity)
  - Assemble candidate causal chains ordered by anomaly score

WHY DUAL-SPACE ANN:
  self_emb ANN  → finds entities that ARE similar (same brand, package, etc.)
  reflect_emb ANN → finds entities in a SIMILAR NEIGHBOURHOOD CONTEXT
                    even if they look different on their own attributes.
  The union exposes relationships that single-space search misses entirely.
"""

from __future__ import annotations

import uuid
import numpy as np
from neo4j import GraphDatabase

from agents.graph_search import LABEL_INDEX, LABEL_REFLECT_INDEX, run_graph_search_task
from agents.llm import LLMError, get_llm
from agents.models import (
    QuerySpec, QueryTask, TaskList, EntityNode, CandidateChain,
)
from agents.pipeline_trace import DOER_TASK_LABELS, trace
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ── Primary key per label ─────────────────────────────────────────────────────
PK = {
    "GlobalSKU":     "sku_id",
    "TenantSKU":     "tenant_sku_id",
    "Brand":         "brand_id",
    "Customer":      "customer_id",
    "PackageType":   "package_type_id",
    "TrainingImage": "image_id",
    "MergeEvent":    "merge_id",
    "Pallet":        "pallet_id",
    "Manufacturer":  "name",
    "Supplier":      "name",
    "ProductClass":  "name",
}

DISPLAY_FIELD = {
    "GlobalSKU":     "brand_family",
    "TenantSKU":     "brand",
    "Brand":         "brand_family",
    "Customer":      "name",
    "PackageType":   "package_category_name",
    "TrainingImage": "image_id",
    "MergeEvent":    "merge_id",
    "Pallet":        "pallet_id",
    "Manufacturer":  "name",
    "Supplier":      "name",
    "ProductClass":  "name",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-8 else 0.0


def _anomaly_score(self_emb, reflect_emb) -> float:
    return round(1.0 - _cosine_sim(self_emb, reflect_emb), 4)


def _entity_key(label: str, entity_id: str) -> str:
    return f"{label}::{entity_id}"


def _fetch_entity_embedding(session, label: str, entity_id: str) -> dict | None:
    """Fetch self_emb + reflect_emb for a specific entity (used as ANN query vector)."""
    pk = PK.get(label, "sku_id")
    result = session.run(
        f"MATCH (n:{label} {{{pk}: $eid}}) "
        f"RETURN n.self_emb AS se, n.reflect_emb AS re",
        eid=entity_id,
    ).single()
    return dict(result) if result else None


# ─────────────────────────────────────────────────────────────────────────────
# TASK EXECUTORS
# ─────────────────────────────────────────────────────────────────────────────

def _run_cypher_traverse(session, task: QueryTask) -> list[EntityNode]:
    """Execute a Cypher traversal query and return EntityNode list."""
    nodes: dict[str, EntityNode] = {}

    result = session.run(task.cypher, task.cypher_params)
    records = list(result)

    for rec in records:
        rec_data = dict(rec)

        # Handle path-style results (root_cause Cypher)
        if "path_ids" in rec_data and "path_labels" in rec_data:
            path_ids    = rec_data.get("path_ids", [])
            path_labels = rec_data.get("path_labels", [])
            for eid, elabel in zip(path_ids, path_labels):
                if eid is None:
                    continue
                key = _entity_key(elabel, str(eid))
                if key not in nodes:
                    nodes[key] = EntityNode(
                        entity_id=str(eid),
                        label=elabel,
                        display_name=str(eid),
                        properties=rec_data,
                        anomaly_score=None,
                        timestamp=rec_data.get("timestamp", ""),
                        source="cypher",
                    )

        # Handle related_id / anchor_id style results
        for id_field, label_field in [
            ("related_id", "related_label"),
            ("neighbour_id", "neighbour_label"),
            ("sku_id", None),
            ("entity_id", "label"),
        ]:
            if id_field not in rec_data:
                continue
            eid   = rec_data.get(id_field)
            label = rec_data.get(label_field) if label_field else task.label
            if eid is None or label is None:
                continue

            se  = rec_data.get("self_emb")
            re_ = rec_data.get("reflect_emb")
            score = _anomaly_score(se, re_) if (se and re_) else None
            display = (
                rec_data.get("display_name") or
                rec_data.get("brand_family") or
                rec_data.get("brand") or
                str(eid)
            )

            key = _entity_key(label, str(eid))
            if key not in nodes:
                nodes[key] = EntityNode(
                    entity_id=str(eid),
                    label=label,
                    display_name=display,
                    properties={k: v for k, v in rec_data.items()
                                 if k not in ("self_emb", "reflect_emb")},
                    anomaly_score=score,
                    timestamp=rec_data.get("timestamp", ""),
                    source="cypher",
                )
            break  # Only process first matching id_field per record

    return list(nodes.values())


def _rows_to_entity_nodes(rows: list[dict]) -> list[EntityNode]:
    nodes = []
    for row in rows:
        se = row.get("self_emb")
        re_ = row.get("reflect_emb")
        score = _anomaly_score(se, re_) if (se and re_) else None
        props = dict(row.get("properties") or {})
        if row.get("match_score") is not None:
            props["match_score"] = row["match_score"]
        nodes.append(EntityNode(
            entity_id=str(row["entity_id"]),
            label=row["label"],
            display_name=str(row.get("display_name") or row["entity_id"]),
            properties=props,
            anomaly_score=score,
            timestamp=row.get("timestamp") or "",
            source=row.get("source") or "graph_search",
        ))
    return nodes


def _run_ann_search(
    session, task: QueryTask, emb_type: str
) -> list[EntityNode]:
    """
    Run ANN search using Neo4j vector index.

    emb_type: "self_emb" | "reflect_emb"
    """
    if not task.anchor_id or not task.index_name:
        return []

    # Fetch the query vector from the anchor entity
    anchor_embs = _fetch_entity_embedding(session, task.label, task.anchor_id)
    if not anchor_embs:
        print(f"    [Doer] ANN: anchor {task.label} {task.anchor_id} not found, skipping")
        return []

    query_vector = anchor_embs.get(
        "se" if emb_type == "self_emb" else "re"
    )
    if query_vector is None:
        print(f"    [Doer] ANN: anchor has no {emb_type}, skipping")
        return []

    pk = PK.get(task.label, "sku_id")
    display_prop = DISPLAY_FIELD.get(task.label, "name")

    cypher = f"""
    CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
    YIELD node AS n, score AS similarity
    WHERE n.{pk} <> $anchor_id
    RETURN
        n.{pk}                                                    AS entity_id,
        '{task.label}'                                            AS label,
        coalesce(n.{display_prop}, n.{pk})                       AS display_name,
        n.self_emb                                                AS self_emb,
        n.reflect_emb                                             AS reflect_emb,
        coalesce(n.creation_date, '')                             AS timestamp,
        similarity
    LIMIT $top_k
    """

    result = session.run(
        cypher,
        index_name=task.index_name,
        top_k=task.top_k,
        query_vector=query_vector,
        anchor_id=task.anchor_id,
    )

    nodes = []
    for rec in result:
        se  = rec["self_emb"]
        re_ = rec["reflect_emb"]
        score = _anomaly_score(se, re_) if (se and re_) else None
        nodes.append(EntityNode(
            entity_id=str(rec["entity_id"]),
            label=rec["label"],
            display_name=rec["display_name"] or str(rec["entity_id"]),
            properties={"similarity": rec["similarity"]},
            anomaly_score=score,
            timestamp=rec.get("timestamp", ""),
            source=f"ann_{emb_type.replace('_emb', '')}",
        ))
    return nodes


def _trace_doer_task(task: QueryTask, *, hits: int | None = None) -> None:
    title = DOER_TASK_LABELS.get(task.task_type, task.description or task.task_type)
    if hits is None:
        trace(
            "doer", "running",
            title,
            task.description or f"Step {task.step}",
            step=task.step,
            task_type=task.task_type,
        )
        return
    trace(
        "doer", "done",
        title,
        f"Found {hits} result(s)" if hits else "No hits",
        step=task.step,
        task_type=task.task_type,
        hits=hits,
    )


def _run_master_match(task: QueryTask) -> tuple[list[EntityNode], dict]:
    """Run api.agent_matcher against master catalog; return nodes + raw result."""
    from api.agent_matcher import agent_match
    from agents.dim_match import normalize_query_dims

    brand = (task.brand_name or "").strip()
    package = (task.package_type or "").strip()
    if not brand or not package:
        return [], {}

    query_dims = normalize_query_dims(task.query_dims)
    result = agent_match(brand, package, query_dims)
    nodes: list[EntityNode] = []

    for sku in result.get("matched_skus", [])[:5]:
        breakdown = sku.get("score_breakdown") or {}
        nodes.append(EntityNode(
            entity_id=str(sku.get("sku_id", "")),
            label="GlobalSKU",
            display_name=(
                f"{sku.get('brand_name', brand)} / "
                f"{sku.get('package_category_name', package)}"
            ),
            properties={
                "brand_name": sku.get("brand_name"),
                "package_category_name": sku.get("package_category_name"),
                "package_name": sku.get("package_name"),
                "weight": sku.get("weight"),
                "height": sku.get("height"),
                "length": sku.get("length"),
                "width": sku.get("width"),
                "match_confidence": sku.get("confidence"),
                "match_status": result.get("status"),
                "score_status": result.get("score_status"),
                "llm_indicator": result.get("llm_indicator"),
                "score_breakdown": breakdown,
                "signals": sku.get("signals", []),
                "dim_boost": sku.get("dim_boost"),
                "dim_distance": sku.get("dim_distance"),
                "kg_available": result.get("kg_available", True),
                "query_brand": brand,
                "query_package": package,
                "query_dims": result.get("query", {}).get("query_dims") or query_dims,
                "dim_applied": result.get("dim_applied", False),
                "dim_mode": result.get("dim_mode", "none"),
                "product_risk": result.get("product_risk"),
            },
            anomaly_score=breakdown.get("anomaly_attn"),
            source="master_match",
        ))

    return nodes, result


def _assemble_catalog_chains(
    nodes: list[EntityNode],
    match_result: dict,
) -> list[CandidateChain]:
    """One validated chain containing ranked master-match candidates."""
    master_nodes = [n for n in nodes if n.source == "master_match"]
    graph_nodes = [n for n in nodes if n.source != "master_match"]
    path = master_nodes + graph_nodes if master_nodes else graph_nodes

    if not path:
        return [
            CandidateChain(
                chain_id="catalog_no_match",
                path=[],
                source="master_match",
            )
        ]
    return [
        CandidateChain(
            chain_id="catalog_match",
            path=path,
            source="master_match",
        )
    ]


def _run_anomaly_rank(session, task: QueryTask) -> list[EntityNode]:
    """Execute anomaly_rank task: fetch all entities, score in Python, return sorted."""
    if not task.cypher:
        return []

    result = session.run(task.cypher, task.cypher_params)
    nodes = []
    for rec in result:
        se  = rec.get("self_emb")
        re_ = rec.get("reflect_emb")
        score = _anomaly_score(se, re_) if (se and re_) else None
        nodes.append(EntityNode(
            entity_id=str(rec.get("entity_id", "")),
            label=rec.get("label", task.label),
            display_name=str(rec.get("display_name", "")),
            properties={k: v for k, v in dict(rec).items()
                        if k not in ("self_emb", "reflect_emb")},
            anomaly_score=score,
            timestamp=rec.get("timestamp", ""),
            source="anomaly_rank",
        ))

    nodes.sort(key=lambda n: n.anomaly_score or 0.0, reverse=True)
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

# Ordered labels for anchor-centric evidence paths (dynamic Doer).
_PATH_LABEL_ORDER: dict[str, int] = {
    "TenantSKU": 0,
    "MergeEvent": 1,
    "Customer": 2,
    "GlobalSKU": 3,
    "Brand": 4,
    "TrainingImage": 5,
    "PackageType": 6,
    "Pallet": 7,
}
_DYNAMIC_RANK_TOP_N = 30
_ANCHOR_ANN_CONTEXT = 3


def _assemble_anchor_chain(
    ordered_evidence: list[EntityNode],
    all_nodes: list[EntityNode],
    anchor_id: str,
) -> CandidateChain:
    """Merge pre-built SKU evidence with neighbourhood hits and limited ANN context."""
    path = list(ordered_evidence)
    seen = {_entity_key(n.label, n.entity_id) for n in path}

    extras = [
        n for n in all_nodes
        if n.source == "cypher" and _entity_key(n.label, n.entity_id) not in seen
    ]
    extras.sort(
        key=lambda n: (_PATH_LABEL_ORDER.get(n.label, 99), -(n.anomaly_score or 0)),
    )
    for n in extras:
        key = _entity_key(n.label, n.entity_id)
        if key in seen:
            continue
        seen.add(key)
        path.append(n)

    ann_nodes = sorted(
        [
            n for n in all_nodes
            if n.source.startswith("ann_")
            and _entity_key(n.label, n.entity_id) not in seen
        ],
        key=lambda n: -(n.anomaly_score or 0),
    )
    for n in ann_nodes[:_ANCHOR_ANN_CONTEXT]:
        key = _entity_key(n.label, n.entity_id)
        seen.add(key)
        path.append(n)

    return CandidateChain(
        chain_id=f"anchor_{anchor_id}",
        path=path,
        source="cypher",
    )


def _assemble_chains(
    all_nodes: list[EntityNode],
    spec_task_type: str,
    anchor_id: str | None,
    anchor_label: str | None = None,
    ordered_evidence: list[EntityNode] | None = None,
) -> tuple[list[CandidateChain], dict[str, dict[str, int]]]:
    """
    Convert a flat list of entities into candidate causal chains.

    Dynamic modes:
    - anchor + root_cause/anomaly_explain → ordered evidence path for the SKU
    - risk_rank → single ranked GlobalSKU chain
    """
    meta: dict[str, dict[str, int]] = {}
    chains: list[CandidateChain] = []

    if (
        ordered_evidence
        and anchor_id
        and spec_task_type in ("root_cause", "anomaly_explain")
    ):
        chain = _assemble_anchor_chain(ordered_evidence, all_nodes, anchor_id)
        return ([chain] if chain.path else []), meta

    if spec_task_type == "risk_rank":
        sorted_nodes = sorted(
            all_nodes,
            key=lambda n: n.anomaly_score or 0.0,
            reverse=True,
        )
        shown = sorted_nodes[:_DYNAMIC_RANK_TOP_N]
        meta["dynamic_rank"] = {"shown": len(shown), "total": len(sorted_nodes)}
        if shown:
            chains.append(CandidateChain(
                chain_id="dynamic_rank",
                path=shown,
                source="anomaly_rank",
            ))
        return chains, meta

    if spec_task_type == "anomaly_explain":
        sorted_nodes = sorted(
            all_nodes,
            key=lambda n: n.anomaly_score or 0.0,
            reverse=True,
        )
        for node in sorted_nodes[:_DYNAMIC_RANK_TOP_N]:
            chains.append(CandidateChain(
                chain_id=str(uuid.uuid4())[:8],
                path=[node],
                source=node.source,
            ))
        return chains, meta

    # root_cause without anchor — group by source, union top anomalies
    by_source: dict[str, list[EntityNode]] = {}
    for n in all_nodes:
        by_source.setdefault(n.source, []).append(n)

    for source, nodes in by_source.items():
        if not nodes:
            continue
        nodes_sorted = sorted(nodes, key=lambda n: n.anomaly_score or 0.0, reverse=True)
        chains.append(CandidateChain(
            chain_id=str(uuid.uuid4())[:8],
            path=nodes_sorted[:10],
            source=source,
        ))

    all_sorted = sorted(all_nodes, key=lambda n: n.anomaly_score or 0.0, reverse=True)
    seen: set[str] = set()
    union_path: list[EntityNode] = []
    for n in all_sorted:
        key = _entity_key(n.label, n.entity_id)
        if key in seen:
            continue
        seen.add(key)
        union_path.append(n)
        if len(union_path) >= 15:
            break

    if union_path:
        chains.append(CandidateChain(
            chain_id=str(uuid.uuid4())[:8],
            path=union_path,
            source="union",
        ))

    return chains, meta


def _chain_path_summary(chain: CandidateChain, limit: int = 10) -> str:
    nodes = chain.path[:limit]
    summary = " → ".join(f"{n.label}:{n.display_name}" for n in nodes)
    if len(chain.path) > limit:
        summary += f" (+{len(chain.path) - limit} more)"
    return summary


def _top_sku_lines(chain: CandidateChain, n: int = 3) -> str:
    """Compact top-N SKU lines for Doer LLM prompts (avoids truncating long paths)."""
    skus = [node for node in chain.path if node.label == "GlobalSKU"]
    skus.sort(key=lambda x: x.anomaly_score or 0.0, reverse=True)
    lines = []
    for node in skus[:n]:
        score = node.anomaly_score
        score_txt = f"{score:.2f}" if score is not None else "—"
        lines.append(f"  {node.entity_id}: {node.display_name} (anomaly {score_txt})")
    return "\n".join(lines)


def _llm_interpret_chain(question: str, spec: QuerySpec, chain: CandidateChain) -> str:
    """Doer LLM: explain how this chain answers the user's question."""
    large_chain = len(chain.path) > 5
    path = _chain_path_summary(chain, limit=5 if large_chain else 10)
    high = [n for n in chain.path if (n.anomaly_score or 0) >= 0.5]
    top_skus = _top_sku_lines(chain, 3) if large_chain else ""

    if large_chain:
        instruction = (
            "In 2–3 complete sentences, explain what this evidence shows and how it "
            "answers the question. The full list is in the UI table — name at most the "
            "top 3 SKUs; do not enumerate every entity or repeat long package strings. "
            "Finish every sentence."
        )
    else:
        instruction = (
            "In 2 complete sentences, explain what this evidence chain shows and how it "
            "answers the question. Mention specific SKUs/brands if present. "
            "Finish every sentence."
        )

    prompt = (
        f"User question: {question}\n"
        f"Task type: {spec.task_type} | Scenario: {spec.scenario_num}\n"
        f"Chain source: {chain.source} | Entities ({len(chain.path)}): {path}\n"
        f"High-anomaly entities: {len(high)}\n"
    )
    if top_skus:
        prompt += f"\nTop SKUs by anomaly score:\n{top_skus}\n"
    prompt += f"\n{instruction}"

    try:
        return get_llm().complete(
            prompt,
            system="You are the Doer agent for a reflexive SKU knowledge graph.",
            max_tokens=384,
        )
    except LLMError:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# DOER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class DoerAgent:
    """
    Agent 3 — Query Executor.

    Executes each task in the TaskList:
    1. Cypher traversal on Neo4j
    2. ANN on self_emb (semantic similarity)
    3. ANN on reflect_emb (neighbourhood context)

    Unions all results, deduplicates by entity_id,
    and assembles candidate causal chains for the Critic.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def _enrich_with_llm(self, spec: QuerySpec, chains: list[CandidateChain]) -> list[CandidateChain]:
        if not self.use_llm or not chains:
            return chains
        question = spec.question
        print(f"  [Doer] LLM interpreting top {min(3, len(chains))} chain(s) ...")
        for chain in chains[:3]:
            summary = _llm_interpret_chain(question, spec, chain)
            if summary:
                chain.llm_summary = summary
                print(f"    chain={chain.chain_id}: {summary[:100]}...")
        return chains

    def execute(self, task_list: TaskList) -> list[CandidateChain]:
        spec = task_list.spec
        print(f"\n[Doer] Executing {len(task_list.tasks)} tasks ...")

        if spec.scenario_num is not None:
            return self._execute_lifecycle(spec)

        if spec.task_type == "catalog_duplicate":
            return self._execute_master_duplicate(spec, task_list)

        if spec.task_type == "catalog_match":
            all_nodes: list[EntityNode] = []
            match_result: dict = {}
            for task in task_list.tasks:
                print(f"  Step {task.step}: [{task.task_type}] {task.description}")
                _trace_doer_task(task)
                if task.task_type in ("graph_exact", "graph_fuzzy", "graph_semantic"):
                    with self.driver.session() as session:
                        rows = run_graph_search_task(session, task)
                        nodes = _rows_to_entity_nodes(rows)
                        all_nodes.extend(nodes)
                        print(f"    → {len(nodes)} graph search hits")
                        _trace_doer_task(task, hits=len(nodes))
                elif task.task_type == "master_match":
                    nodes, match_result = _run_master_match(task)
                    all_nodes.extend(nodes)
                    print(f"    → {len(nodes)} master catalog candidates")
                    _trace_doer_task(task, hits=len(nodes))
            chains = _assemble_catalog_chains(all_nodes, match_result)
            self._last_match_result = match_result  # type: ignore[attr-defined]
            print(f"  [Doer] Assembled {len(chains)} catalog match chain(s)")
            trace(
                "doer", "done",
                "Evidence assembled",
                f"{len(all_nodes)} graph signals · {len(chains)} match chain(s)",
                chains=len(chains),
                entities=len(all_nodes),
            )
            return self._enrich_with_llm(spec, chains)

        all_nodes_map: dict[str, EntityNode] = {}
        discovered_anchor: str | None = spec.anchor_entity_id
        discovered_label: str = spec.anchor_label or "GlobalSKU"
        ordered_evidence: list[EntityNode] = []

        with self.driver.session() as session:
            anchor_label = spec.anchor_label or "GlobalSKU"
            if (
                spec.anchor_entity_id
                and anchor_label == "GlobalSKU"
                and spec.task_type in ("root_cause", "anomaly_explain")
            ):
                from agents.lifecycle_doer import LifecycleDoer

                lc = LifecycleDoer(session)
                ordered_evidence = lc.sku_root_cause_chain(spec.anchor_entity_id).path
                for node in ordered_evidence:
                    all_nodes_map[_entity_key(node.label, node.entity_id)] = node
                print(
                    f"  [Doer] Anchor evidence path for GlobalSKU {spec.anchor_entity_id}: "
                    f"{len(ordered_evidence)} hop(s)"
                )

            for task in task_list.tasks:
                print(f"  Step {task.step}: [{task.task_type}] {task.description}")
                _trace_doer_task(task)

                nodes = []

                if task.task_type in ("graph_exact", "graph_fuzzy", "graph_semantic"):
                    rows = run_graph_search_task(session, task)
                    nodes = _rows_to_entity_nodes(rows)
                    if not discovered_anchor and nodes:
                        discovered_anchor = nodes[0].entity_id
                        discovered_label = task.label
                        print(f"    → discovered anchor {discovered_label} {discovered_anchor}")

                elif task.task_type == "cypher_traverse":
                    nodes = _run_cypher_traverse(session, task)

                elif task.task_type == "ann_self":
                    if not task.anchor_id and discovered_anchor:
                        task.anchor_id = discovered_anchor
                        task.index_name = (
                            task.index_name
                            or LABEL_INDEX.get(discovered_label)
                            or LABEL_INDEX.get(task.label)
                        )
                    nodes = _run_ann_search(session, task, "self_emb")

                elif task.task_type == "ann_reflect":
                    if not task.anchor_id and discovered_anchor:
                        task.anchor_id = discovered_anchor
                        task.index_name = (
                            task.index_name
                            or LABEL_REFLECT_INDEX.get(discovered_label)
                            or LABEL_REFLECT_INDEX.get(task.label)
                        )
                    nodes = _run_ann_search(session, task, "reflect_emb")

                elif task.task_type == "anomaly_rank":
                    nodes = _run_anomaly_rank(session, task)

                elif task.task_type == "lifecycle_cypher":
                    raise RuntimeError("lifecycle_cypher must be handled via spec.scenario_num")

                print(f"    → {len(nodes)} entities from step {task.step}")
                _trace_doer_task(task, hits=len(nodes))

                for node in nodes:
                    key = _entity_key(node.label, node.entity_id)
                    existing = all_nodes_map.get(key)
                    if existing is None:
                        all_nodes_map[key] = node
                    elif (node.anomaly_score or 0) > (existing.anomaly_score or 0):
                        all_nodes_map[key] = node
                    elif node.source.startswith("graph_") and not existing.source.startswith("graph_"):
                        # Prefer graph search provenance when scores tie
                        all_nodes_map[key] = node

        unique_nodes = list(all_nodes_map.values())
        print(f"\n  [Doer] Total unique entities after union+dedup: {len(unique_nodes)}")

        chains, chain_meta = _assemble_chains(
            unique_nodes,
            spec.task_type,
            spec.anchor_entity_id,
            spec.anchor_label,
            ordered_evidence=ordered_evidence or None,
        )
        if chain_meta:
            self._result_meta = {**getattr(self, "_result_meta", {}), **chain_meta}  # type: ignore[attr-defined]
        print(f"  [Doer] Assembled {len(chains)} candidate chains")
        trace(
            "doer", "done",
            "Evidence assembled",
            f"{len(unique_nodes)} unique entities · {len(chains)} candidate chain(s)",
            entities=len(unique_nodes),
            chains=len(chains),
        )
        return self._enrich_with_llm(spec, chains)

    def _execute_master_duplicate(self, spec, task_list: TaskList) -> list[CandidateChain]:
        from agents.master_duplicate import build_duplicate_chain, scan_master_duplicates

        print(f"  [Doer] Master catalog duplicate scan ...")
        with self.driver.session() as session:
            report = scan_master_duplicates(session=session)
        chain = build_duplicate_chain(report)
        self._last_duplicate_report = report.to_dict()  # type: ignore[attr-defined]
        n = report.total_groups
        print(f"    → {n} duplicate group(s) from {report.source}")
        return self._enrich_with_llm(spec, [chain])

    def _execute_lifecycle(self, spec) -> list[CandidateChain]:
        from agents.lifecycle_doer import LifecycleDoer

        scenario_num = spec.scenario_num
        print(f"  [Doer] Lifecycle scenario {scenario_num} (cohort graph Cypher)")

        with self.driver.session() as session:
            doer = LifecycleDoer(session)
            if scenario_num == 4:
                self._closed_world_rows = doer.closed_world_brand_dupes()  # type: ignore[attr-defined]
                n_closed = len(self._closed_world_rows)
                print(f"    → closed-world query: {n_closed} rows")
                chain = doer.brand_mismatch_chain()
                chain.chain_id = "scenario4_ab"
                candidates = [chain]
            else:
                self._closed_world_rows = None  # type: ignore[attr-defined]
                candidates = doer.chain_for_scenario(scenario_num)
            self._result_meta = dict(doer.last_meta)  # type: ignore[attr-defined]

        print(f"  [Doer] Assembled {len(candidates)} lifecycle chain(s)")
        return self._enrich_with_llm(spec, candidates)

    def close(self):
        self.driver.close()
