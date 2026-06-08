"""
agents/planner.py — Planner Agent

Responsibilities:
  - Receive a QuerySpec from the Supervisor
  - Build an ordered TaskList of Cypher traversals + ANN searches
  - Decide per-hop whether to use self_emb, reflect_emb, or both
  - Return a TaskList for the Doer

Strategy by task_type:
  root_cause     → backward Cypher trace + reflect_emb ANN at anchor
  risk_rank      → anomaly_rank query (no traversal needed)
  anomaly_explain → local Cypher neighbourhood + both self + reflect ANN
"""

from __future__ import annotations

from agents.llm import bedrock_model_label, get_llm
from agents.models import QuerySpec, QueryTask, TaskList

# ── Neo4j vector index names (must match 01_schema.py) ───────────────────────
SELF_INDEXES = {
    "GlobalSKU":    "idx_global_sku_self",
    "TenantSKU":    "idx_tenant_sku_self",
    "Brand":        "idx_brand_self",
    "PackageType":  "idx_package_self",
    "Manufacturer": "idx_mfr_self",
    "Supplier":     "idx_supplier_self",
    "ProductClass": "idx_class_self",
}
REFLECT_INDEXES = {
    "GlobalSKU":    "idx_global_sku_reflect",
    "TenantSKU":    "idx_tenant_sku_reflect",
    "Brand":        "idx_brand_reflect",
    "PackageType":  "idx_package_reflect",
    "Manufacturer": "idx_mfr_reflect",
    "Supplier":     "idx_supplier_reflect",
    "ProductClass": "idx_class_reflect",
}

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

# ── Relationship paths (handbook §3.2) ───────────────────────────────────────
GRAPH_PATHS = [
    ("TenantSKU",     "MAPS_TO",          "GlobalSKU"),
    ("GlobalSKU",     "BELONGS_TO_BRAND", "Brand"),
    ("GlobalSKU",     "HAS_PACKAGE",      "PackageType"),
    ("GlobalSKU",     "MADE_BY",          "Manufacturer"),
    ("GlobalSKU",     "USED_BY",          "Customer"),
    ("GlobalSKU",     "MERGED_INTO",      "MergeEvent"),
    ("TrainingImage", "TRAINED_WITH",     "GlobalSKU"),
    ("Pallet",        "SCANNED_ON",       "GlobalSKU"),
    ("Brand",         "FUZZY_MATCH",      "Brand"),
    ("TenantSKU",     "SUPPLIED_BY",      "Supplier"),
    ("TenantSKU",     "IN_CLASS",         "ProductClass"),
]


# ─────────────────────────────────────────────────────────────────────────────
# CYPHER TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

def _cypher_root_cause(anchor_label: str, anchor_id: str, depth: int) -> str:
    """
    Backward causal trace from an anchor entity.
    Traverses all relationship types up to `depth` hops.
    Returns entity paths with timestamps for the Critic.
    """
    pk = PK.get(anchor_label, "sku_id")
    return f"""
    MATCH path = (anchor:{anchor_label} {{{pk}: $anchor_id}})-[*1..{depth}]-(related)
    WHERE related.self_emb IS NOT NULL
    WITH anchor, related, relationships(path) AS rels, nodes(path) AS path_nodes
    RETURN
        anchor.{pk}                          AS anchor_id,
        labels(anchor)[0]                    AS anchor_label,
        related.{pk}                         AS related_id,
        labels(related)[0]                   AS related_label,
        [r IN rels | type(r)]                AS rel_types,
        [n IN path_nodes | labels(n)[0]]     AS path_labels,
        [n IN path_nodes | n.{pk}]           AS path_ids,
        related.self_emb                     AS self_emb,
        related.reflect_emb                  AS reflect_emb,
        coalesce(related.creation_date, related.scan_timestamp, '') AS timestamp
    ORDER BY size(path_nodes)
    LIMIT 200
    """


def _cypher_neighbourhood(anchor_label: str, anchor_id: str) -> str:
    """
    1-hop neighbourhood of a specific entity.
    Used for anomaly_explain — get immediate context.
    """
    pk = PK.get(anchor_label, "sku_id")
    return f"""
    MATCH (anchor:{anchor_label} {{{pk}: $anchor_id}})-[r]-(neighbour)
    RETURN
        type(r)                              AS rel_type,
        labels(neighbour)[0]                 AS neighbour_label,
        neighbour.{pk}                       AS neighbour_id,
        neighbour.self_emb                   AS self_emb,
        neighbour.reflect_emb                AS reflect_emb,
        coalesce(neighbour.brand_family, neighbour.name,
                 neighbour.product_description, neighbour.{pk}) AS display_name,
        coalesce(neighbour.creation_date, '') AS timestamp
    LIMIT 100
    """


def _cypher_import_brand_chain() -> str:
    """
    Handbook §5 example traversal:
    TenantSKU → MAPS_TO → GlobalSKU → BELONGS_TO_BRAND → Brand → FUZZY_MATCH
    """
    return """
    MATCH (t:TenantSKU)-[:MAPS_TO]->(g:GlobalSKU)-[:BELONGS_TO_BRAND]->(b:Brand)
    OPTIONAL MATCH (b)-[f:FUZZY_MATCH]->(b2:Brand)
    OPTIONAL MATCH (img:TrainingImage)-[:TRAINED_WITH]->(g)
    OPTIONAL MATCH (p:Pallet)-[:SCANNED_ON]->(g)
    WITH t, g, b, count(DISTINCT f) AS fuzzy_count,
         count(DISTINCT img) AS img_count,
         sum(CASE WHEN p.outcome = 'failure' THEN 1 ELSE 0 END) AS scan_failures
    WHERE fuzzy_count >= 1 OR b.canonical = false OR coalesce(b.canonical, true) = false
    RETURN
        t.tenant_sku_id                     AS related_id,
        'TenantSKU'                         AS related_label,
        g.sku_id                            AS sku_id,
        b.brand_id                          AS brand_id,
        ['MAPS_TO','BELONGS_TO_BRAND']      AS rel_types,
        g.self_emb                          AS self_emb,
        g.reflect_emb                       AS reflect_emb,
        coalesce(t.creation_date, g.creation_date, '') AS timestamp,
        fuzzy_count,
        img_count,
        scan_failures
    ORDER BY fuzzy_count DESC, scan_failures DESC
    LIMIT 200
    """


def _cypher_brand_fragmentation() -> str:
    """
    Scenario 1: Detect brands with high FUZZY_MATCH fan-out.
    A brand with many fuzzy matches is likely fragmented.
    """
    return """
    MATCH (b:Brand)-[f:FUZZY_MATCH]->(b2:Brand)
    WITH b, count(f) AS fuzzy_count
    WHERE fuzzy_count >= 2
    MATCH (s:GlobalSKU)-[:BELONGS_TO_BRAND]->(b)
    RETURN
        b.brand_id                          AS anchor_id,
        'Brand'                             AS anchor_label,
        s.sku_id                            AS related_id,
        'GlobalSKU'                         AS related_label,
        ['FUZZY_MATCH', 'BELONGS_TO_BRAND'] AS rel_types,
        s.self_emb                          AS self_emb,
        s.reflect_emb                       AS reflect_emb,
        s.creation_date                     AS timestamp,
        fuzzy_count                         AS fuzzy_match_count
    ORDER BY fuzzy_count DESC
    LIMIT 200
    """


def _cypher_multi_signal_risk() -> str:
    """
    Scenario 2: Cross-source weak signal aggregation.
    Find GlobalSKUs that have: missing UPC + no TenantSKU mapping + review needed.
    """
    return """
    MATCH (s:GlobalSKU)
    WHERE s.upc_missing = true
       OR s.is_review_needed = true
       OR s.is_imaged_on_training_station = false
    OPTIONAL MATCH (t:TenantSKU)-[:MAPS_TO]->(s)
    WITH s,
         count(t) AS tenant_mappings,
         s.upc_missing                         AS missing_upc,
         s.is_review_needed                    AS review_needed,
         NOT s.is_imaged_on_training_station   AS not_imaged
    WITH s, tenant_mappings, missing_upc, review_needed, not_imaged,
         (CASE WHEN missing_upc   THEN 1 ELSE 0 END +
          CASE WHEN review_needed THEN 1 ELSE 0 END +
          CASE WHEN not_imaged    THEN 1 ELSE 0 END +
          CASE WHEN tenant_mappings = 0 THEN 1 ELSE 0 END) AS risk_signals
    WHERE risk_signals >= 2
    RETURN
        s.sku_id              AS sku_id,
        s.brand_family        AS brand_family,
        s.package_category_name AS package,
        risk_signals,
        missing_upc,
        review_needed,
        not_imaged,
        tenant_mappings,
        s.self_emb            AS self_emb,
        s.reflect_emb         AS reflect_emb,
        s.creation_date       AS timestamp
    ORDER BY risk_signals DESC
    LIMIT 100
    """


def _cypher_anomaly_rank(label: str, top_n: int = 20) -> str:
    """
    Scenario 3: Rank entities by anomaly score (no rule required).
    Returns anchor + reflect embeddings for Python-side scoring.
    """
    pk = PK.get(label, "sku_id")
    display = {
        "GlobalSKU": "coalesce(n.brand_family, n.sku_id)",
        "TenantSKU": "coalesce(n.brand, n.tenant_sku_id)",
        "Brand":     "n.brand_family",
    }.get(label, "n.name")
    return f"""
    MATCH (n:{label})
    WHERE n.self_emb IS NOT NULL AND n.reflect_emb IS NOT NULL
    RETURN
        n.{pk}       AS entity_id,
        '{label}'    AS label,
        {display}    AS display_name,
        n.self_emb   AS self_emb,
        n.reflect_emb AS reflect_emb,
        coalesce(n.creation_date, '') AS timestamp
    LIMIT {top_n * 5}
    """


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    Agent 2 — Query Decomposer.

    Converts a QuerySpec into an ordered TaskList (Cypher + ANN steps).
    Bedrock (Claude Opus 4.7) produces a required rationale for the plan; execution
    steps remain template-based for reproducible graph access.
    """

    def plan(self, spec: QuerySpec) -> TaskList:
        print(f"\n[Planner] Building task list for task_type='{spec.task_type}'")
        task_list = TaskList(spec=spec)

        if spec.task_type == "root_cause":
            self._plan_root_cause(spec, task_list)
        elif spec.task_type == "risk_rank":
            self._plan_risk_rank(spec, task_list)
        elif spec.task_type == "anomaly_explain":
            self._plan_anomaly_explain(spec, task_list)

        print(f"  [Planner] → {len(task_list.tasks)} tasks planned:")
        for t in task_list.tasks:
            print(f"    Step {t.step}: [{t.task_type}] {t.description}")

        steps = "\n".join(f"  {t.step}. [{t.task_type}] {t.description}" for t in task_list.tasks)
        task_list.llm_rationale = get_llm().complete(
            f"Summarize this reflexive-KG query plan in 2 sentences for an analyst.\n"
            f"task_type={spec.task_type} depth={spec.traversal_depth} anchor={spec.anchor_entity_id}\n"
            f"Steps:\n{steps}",
            system="You are the Planner agent for a SKU knowledge graph. Be concise.",
            max_tokens=200,
        )
        print(f"  [Planner] Rationale ({bedrock_model_label()}): {task_list.llm_rationale[:120]}...")
        return task_list

    # ── Root cause: backward trace + reflect ANN ──────────────────────────────

    def _plan_root_cause(self, spec: QuerySpec, tl: TaskList):
        anchor_label = spec.anchor_label or "GlobalSKU"
        anchor_id    = spec.anchor_entity_id

        step = 1

        # Handbook import → brand mismatch chain (TenantSKU → Brand → FUZZY_MATCH)
        if not anchor_id and (
            "Brand" in spec.entity_types
            or "TenantSKU" in spec.entity_types
            or "Customer" in spec.entity_types
        ):
            tl.tasks.append(QueryTask(
                step=step,
                task_type="cypher_traverse",
                label="TenantSKU",
                description="Import chain: TenantSKU → GlobalSKU → Brand → FUZZY_MATCH",
                cypher=_cypher_import_brand_chain(),
                cypher_params={},
            ))
            step += 1
            tl.tasks.append(QueryTask(
                step=step,
                task_type="cypher_traverse",
                label="Brand",
                description="Detect brand fragmentation via FUZZY_MATCH fan-out",
                cypher=_cypher_brand_fragmentation(),
                cypher_params={},
            ))
            step += 1

        # General backward trace from anchor
        elif anchor_id:
            tl.tasks.append(QueryTask(
                step=step,
                task_type="cypher_traverse",
                label=anchor_label,
                description=f"Backward causal trace from {anchor_label} {anchor_id} ({spec.traversal_depth} hops)",
                cypher=_cypher_root_cause(anchor_label, anchor_id, spec.traversal_depth),
                cypher_params={"anchor_id": anchor_id},
            ))
            step += 1

        # Multi-signal risk detection (Scenario 2)
        else:
            tl.tasks.append(QueryTask(
                step=step,
                task_type="cypher_traverse",
                label="GlobalSKU",
                description="Multi-signal risk: SKUs with 2+ weak risk signals",
                cypher=_cypher_multi_signal_risk(),
                cypher_params={},
            ))
            step += 1

        # ANN on reflect_emb — find entities in similar neighbourhood context
        primary_label = anchor_label if anchor_id else "GlobalSKU"
        tl.tasks.append(QueryTask(
            step=step,
            task_type="ann_reflect",
            label=primary_label,
            description=f"Contextual ANN on reflect_emb ({primary_label}) — finds entities in similar neighbourhood",
            index_name=REFLECT_INDEXES.get(primary_label),
            anchor_id=anchor_id,
            use_reflect_emb=True,
            top_k=20,
        ))
        step += 1

        # ANN on self_emb — semantically similar entities
        tl.tasks.append(QueryTask(
            step=step,
            task_type="ann_self",
            label=primary_label,
            description=f"Semantic ANN on self_emb ({primary_label}) — finds similar entities",
            index_name=SELF_INDEXES.get(primary_label),
            anchor_id=anchor_id,
            use_self_emb=True,
            top_k=20,
        ))

    # ── Risk rank: anomaly score only ─────────────────────────────────────────

    def _plan_risk_rank(self, spec: QuerySpec, tl: TaskList):
        labels = spec.entity_types or ["GlobalSKU"]
        for i, label in enumerate(labels, start=1):
            tl.tasks.append(QueryTask(
                step=i,
                task_type="anomaly_rank",
                label=label,
                description=f"Rank {label} by anomaly score (1 - cosine similarity)",
                cypher=_cypher_anomaly_rank(label, top_n=50),
                cypher_params={},
            ))

    # ── Anomaly explain: neighbourhood + dual ANN ─────────────────────────────

    def _plan_anomaly_explain(self, spec: QuerySpec, tl: TaskList):
        anchor_label = spec.anchor_label or "GlobalSKU"
        anchor_id    = spec.anchor_entity_id

        step = 1

        # 1-hop neighbourhood to understand context
        if anchor_id:
            tl.tasks.append(QueryTask(
                step=step,
                task_type="cypher_traverse",
                label=anchor_label,
                description=f"1-hop neighbourhood of {anchor_label} {anchor_id}",
                cypher=_cypher_neighbourhood(anchor_label, anchor_id),
                cypher_params={"anchor_id": anchor_id},
            ))
            step += 1

            # reflect ANN: other entities in similar neighbourhood context
            tl.tasks.append(QueryTask(
                step=step,
                task_type="ann_reflect",
                label=anchor_label,
                description="Reflect ANN: entities in similar neighbourhood context",
                index_name=REFLECT_INDEXES.get(anchor_label),
                anchor_id=anchor_id,
                use_reflect_emb=True,
                top_k=15,
            ))
            step += 1

            # self ANN: semantically similar entities
            tl.tasks.append(QueryTask(
                step=step,
                task_type="ann_self",
                label=anchor_label,
                description="Self ANN: semantically similar entities",
                index_name=SELF_INDEXES.get(anchor_label),
                anchor_id=anchor_id,
                use_self_emb=True,
                top_k=15,
            ))

        else:
            # No anchor — fall back to anomaly rank
            tl.tasks.append(QueryTask(
                step=step,
                task_type="anomaly_rank",
                label=anchor_label,
                description=f"No anchor provided — ranking {anchor_label} by anomaly score",
                cypher=_cypher_anomaly_rank(anchor_label, top_n=20),
                cypher_params={},
            ))
