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

from agents.models import (
    QueryTask, TaskList, EntityNode, CandidateChain,
)
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

    # Sort by anomaly score descending
    nodes.sort(key=lambda n: n.anomaly_score or 0.0, reverse=True)
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_chains(
    all_nodes: list[EntityNode],
    spec_task_type: str,
    anchor_id: str | None,
) -> list[CandidateChain]:
    """
    Convert a flat list of entities into candidate causal chains.

    Strategy:
    - For risk_rank / anomaly_explain: one chain per entity, sorted by score
    - For root_cause: group by source, build ordered path

    In the POC, we represent each meaningful cluster as a chain.
    The Critic will score and rank them.
    """
    chains = []

    if spec_task_type in ("risk_rank", "anomaly_explain"):
        # Each entity is its own "chain" — Critic scores them individually
        sorted_nodes = sorted(
            all_nodes,
            key=lambda n: n.anomaly_score or 0.0,
            reverse=True,
        )
        for node in sorted_nodes[:30]:  # cap candidates
            chains.append(CandidateChain(
                chain_id=str(uuid.uuid4())[:8],
                path=[node],
                source=node.source,
            ))
    else:
        # root_cause: group nodes by source, then build a multi-hop path
        # Anchor first (if known), then related entities by anomaly score desc
        by_source: dict[str, list[EntityNode]] = {}
        for n in all_nodes:
            by_source.setdefault(n.source, []).append(n)

        for source, nodes in by_source.items():
            if not nodes:
                continue
            nodes_sorted = sorted(nodes, key=lambda n: n.anomaly_score or 0.0, reverse=True)

            # Build chain: anchor → top anomalous nodes
            path = nodes_sorted[:10]  # up to 10 nodes per chain
            chains.append(CandidateChain(
                chain_id=str(uuid.uuid4())[:8],
                path=path,
                source=source,
            ))

        # Also build a UNION chain from the top nodes across all sources
        all_sorted = sorted(all_nodes, key=lambda n: n.anomaly_score or 0.0, reverse=True)
        seen = set()
        union_path = []
        for n in all_sorted:
            key = _entity_key(n.label, n.entity_id)
            if key not in seen:
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

    return chains


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

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def execute(self, task_list: TaskList) -> list[CandidateChain]:
        spec = task_list.spec
        print(f"\n[Doer] Executing {len(task_list.tasks)} tasks ...")

        all_nodes: dict[str, EntityNode] = {}

        with self.driver.session() as session:
            for task in task_list.tasks:
                print(f"  Step {task.step}: [{task.task_type}] {task.description}")

                nodes = []

                if task.task_type == "cypher_traverse":
                    nodes = _run_cypher_traverse(session, task)

                elif task.task_type == "ann_self":
                    nodes = _run_ann_search(session, task, "self_emb")

                elif task.task_type == "ann_reflect":
                    nodes = _run_ann_search(session, task, "reflect_emb")

                elif task.task_type == "anomaly_rank":
                    nodes = _run_anomaly_rank(session, task)

                print(f"    → {len(nodes)} entities from step {task.step}")

                # Deduplicate: keep higher anomaly score version
                for node in nodes:
                    key = _entity_key(node.label, node.entity_id)
                    existing = all_nodes.get(key)
                    if existing is None:
                        all_nodes[key] = node
                    elif (node.anomaly_score or 0) > (existing.anomaly_score or 0):
                        all_nodes[key] = node

        unique_nodes = list(all_nodes.values())
        print(f"\n  [Doer] Total unique entities after union+dedup: {len(unique_nodes)}")

        chains = _assemble_chains(unique_nodes, spec.task_type, spec.anchor_entity_id)
        print(f"  [Doer] Assembled {len(chains)} candidate chains")
        return chains

    def close(self):
        self.driver.close()
