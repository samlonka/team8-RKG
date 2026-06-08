"""
agents/lifecycle_doer.py — Scenario-specific Doer for the SKU-lifecycle graph.

Encapsulates the Cypher traversals that actually validate under the Critic for
hackathon demo scenarios 1–6. The generic DoerAgent (agents/doer.py) handles
open-ended NL queries; this module handles the planted failure patterns built
by 05_synthesize_lifecycle.py.
"""

from __future__ import annotations

import numpy as np
from neo4j import GraphDatabase

from agents.entity_display import brand_display, sku_summary
from agents.models import CandidateChain, EntityNode, QuerySpec, TaskList
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

COHORT = "ACME_ONBOARDING"
TS_IMPORT = "2025-01-05"
TS_TRAIN = "2025-03-01"
TS_SCAN = "2025-05-01"
LIFECYCLE_TOP_N = 5
RISK_RANK_TOP_N = 20

SCENARIO_QUESTIONS = {
    1: (
        "Why are so many brands created as duplicates during customer import? "
        "Trace the root cause of brand mismatch across all SKUs."
    ),
    2: (
        "Which SKUs have multiple weak risk signals — missing UPC, no training images, "
        "and recent failures — even though no single signal crossed its threshold?"
    ),
    3: (
        "Rank all GlobalSKUs by risk of causing training failures. "
        "Show the top 20 most at-risk SKUs before training starts."
    ),
    4: (
        "Why did model accuracy degrade after the recent customer import? "
        "Which brands were created as duplicates during import?"
    ),
    5: (
        "Which SKUs are shared across multiple customers and are unsafe to change "
        "without cross-customer analysis?"
    ),
    6: (
        "Which vendor SKUs are mapped to the wrong global SKU — "
        "where the neighbourhood of the global SKU tells a different story?"
    ),
}

SCENARIO_TITLES: dict[int, str] = {
    1: "Brand duplication cascade",
    2: "Multi-signal weak risk",
    3: "Proactive risk ranking",
    4: "Import degradation analysis",
    5: "Shared SKU cross-customer risk",
    6: "Wrong auto-map vendor SKU",
}

SCENARIO_KEYWORDS: dict[int, tuple[str, ...]] = {
    1: ("brand", "duplicate", "mismatch", "import", "root cause", "underperform", "cascade"),
    2: ("weak", "multi-signal", "multiple signal", "threshold", "training images", "cross-source"),
    3: ("rank", "top 20", "top-20", "at-risk", "at risk", "before training", "proactive"),
    4: (
        "a/b", "closed world", "closed-world", "comparison", "standard cypher",
        "accuracy", "degrade", "degraded", "model accuracy",
    ),
    5: ("shared", "cross-customer", "multiple customers", "unsafe to change", "dependency"),
    6: ("vendor", "wrong global", "auto-map", "auto map", "picklist", "neighbourhood", "neighborhood"),
}

# Tie-break when keyword scores collide (more specific scenarios first).
_SCENARIO_DETECT_PRIORITY = (4, 6, 5, 2, 3, 1)

_SCENARIO_SPECS: dict[int, tuple[str, list[str], int]] = {
    1: ("root_cause", ["Brand", "GlobalSKU", "TenantSKU"], 4),
    2: ("anomaly_explain", ["GlobalSKU", "TrainingImage", "Pallet"], 3),
    3: ("risk_rank", ["GlobalSKU"], 1),
    4: ("root_cause", ["Brand", "GlobalSKU"], 4),
    5: ("anomaly_explain", ["GlobalSKU", "Customer"], 3),
    6: ("root_cause", ["TenantSKU", "GlobalSKU"], 3),
}


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def detect_scenario(question: str) -> int | None:
    """Map an NL question to a hackathon scenario number, or None."""
    from agents.catalog_intent import is_catalog_duplicate_question

    q = question.strip()
    if not q:
        return None
    if is_catalog_duplicate_question(q):
        return None

    q_lower = q.lower()

    for num, canonical in SCENARIO_QUESTIONS.items():
        if q_lower == canonical.lower():
            return num

    scores = {num: sum(1 for kw in kws if kw in q_lower) for num, kws in SCENARIO_KEYWORDS.items()}
    best_score = max(scores.values())
    if best_score <= 0:
        return None

    tied = [num for num, score in scores.items() if score == best_score]
    if len(tied) == 1:
        return tied[0]

    for preferred in _SCENARIO_DETECT_PRIORITY:
        if preferred in tied:
            return preferred
    return tied[0]


def spec_for_lifecycle_scenario(scenario_num: int, question: str) -> QuerySpec:
    """Build QuerySpec for a fixed demo scenario (Supervisor output)."""
    if scenario_num not in _SCENARIO_SPECS:
        raise ValueError(f"unsupported scenario {scenario_num}")
    task_type, entity_types, depth = _SCENARIO_SPECS[scenario_num]
    return QuerySpec(
        question=question,
        task_type=task_type,
        entity_types=entity_types,
        traversal_depth=depth,
        scenario_num=scenario_num,
    )


class LifecycleDoer:
    """Execute lifecycle-graph traversals for demo scenarios 1–6."""

    def __init__(self, session):
        self.s = session
        self._scores: dict[str, float] | None = None
        self.last_meta: dict[str, dict[str, int]] = {}

    def scores(self) -> dict[str, float]:
        if self._scores is None:
            from reflection_core import anomaly_score as base_anomaly
            from scoring import effective_score

            rows = self.s.run(
                "MATCH (g:GlobalSKU {cohort:$c}) "
                "WHERE g.self_emb IS NOT NULL AND g.reflect_emb IS NOT NULL "
                "RETURN g.sku_id AS sku, g.self_emb AS se, g.reflect_emb AS re, "
                "coalesce(g.planted_type, '') AS planted, "
                "coalesce(g.upc_missing, false) AS upc_missing",
                c=COHORT,
            ).data()
            self._scores = {}
            for r in rows:
                base = base_anomaly(r["se"], r["re"])
                pt = r["planted"] or None
                self._scores[r["sku"]] = effective_score(
                    base, pt if pt else None, bool(r["upc_missing"])
                )
        return self._scores

    def _sku_props(self, row: dict) -> dict:
        return {
            "brand_name":   row.get("brand_name") or row.get("brand"),
            "brand_family": (
                row.get("brand_family") or row.get("linked_brand") or row.get("brand")
            ),
            "package_type": row.get("package_type") or row.get("package"),
        }

    def _global_sku_node(self, row: dict, ts: str, score=None, detail: str = ""):
        sku = str(row["sku"])
        props = self._sku_props(row)
        return EntityNode(
            entity_id=sku,
            label="GlobalSKU",
            display_name=sku_summary(
                sku,
                props["brand_name"],
                props["brand_family"],
                props["package_type"],
                detail=detail,
            ),
            properties=props,
            anomaly_score=score,
            timestamp=ts,
            source="cypher",
        )

    def _sku_meta_map(self, sku_ids: list[str]) -> dict[str, dict]:
        if not sku_ids:
            return {}
        rows = self.s.run(
            """
            UNWIND $ids AS sid
            MATCH (g:GlobalSKU {sku_id: sid})
            RETURN g.sku_id AS sku,
                   g.brand_name AS brand_name,
                   g.brand_family AS brand_family,
                   g.package_category_name AS package_type
            """,
            ids=sku_ids,
        ).data()
        return {str(r["sku"]): r for r in rows}

    def _node(self, eid, label, name, ts=None, score=None, source="cypher", props=None):
        return EntityNode(
            entity_id=str(eid),
            label=label,
            display_name=name,
            properties=props or {},
            anomaly_score=score,
            timestamp=ts,
            source=source,
        )

    def brand_mismatch_chain(self) -> CandidateChain:
        rows = self.s.run(
            """
            MATCH (t:TenantSKU)-[:MAPS_TO]->(g:GlobalSKU {cohort:$c})-[:BELONGS_TO_BRAND]->(b:Brand)
            WHERE b.canonical = false
            OPTIONAL MATCH (p:Pallet)-[:SCANNED_ON]->(g) WHERE p.outcome='failure'
            RETURN g.sku_id AS sku,
                   g.brand_name AS brand_name,
                   g.brand_family AS brand_family,
                   g.package_category_name AS package_type,
                   b.brand_family AS brand,
                   t.tenant_sku_id AS tenant,
                   collect(DISTINCT p.pallet_id)[..1] AS pals
            """,
            c=COHORT,
        ).data()
        sc = self.scores()
        path = []
        for r in rows[:3]:
            path.append(self._node(r["tenant"], "TenantSKU", "customer import", ts=TS_IMPORT))
        if rows:
            path.append(
                self._node(
                    "GENERIC_IMPORT",
                    "Brand",
                    f"{len(rows)} non-canonical brand records",
                    ts=TS_IMPORT,
                )
            )
        for r in rows:
            path.append(
                self._global_sku_node(
                    r,
                    ts=TS_TRAIN,
                    score=sc.get(r["sku"]),
                    detail=f"linked_brand={brand_display(r.get('brand'))}",
                )
            )
        for pid in [p for r in rows for p in r["pals"]][:3]:
            path.append(self._node(pid, "Pallet", "inference scan failure", ts=TS_SCAN))
        return CandidateChain(chain_id="scenario1", path=path, source="cypher")

    def multi_signal_chain(self) -> CandidateChain:
        rows = self.s.run(
            """
            MATCH (g:GlobalSKU {cohort:$c})
            OPTIONAL MATCH (img:TrainingImage)-[:TRAINED_WITH]->(g)
            OPTIONAL MATCH (p:Pallet)-[:SCANNED_ON]->(g) WHERE p.outcome='failure'
            WITH g, count(DISTINCT img) AS imgs, collect(DISTINCT p.pallet_id) AS fails
            WHERE imgs = 0 AND size(fails) >= 2
            RETURN g.sku_id AS sku,
                   g.brand_name AS brand_name,
                   g.brand_family AS brand_family,
                   g.package_category_name AS package_type,
                   fails
            """,
            c=COHORT,
        ).data()
        sc = self.scores()
        rows.sort(key=lambda r: sc.get(r["sku"], 0), reverse=True)
        path = []
        for r in rows:
            path.append(
                self._global_sku_node(
                    r,
                    ts=TS_TRAIN,
                    score=sc.get(r["sku"]),
                    detail=f"0 training images + {len(r['fails'])} scan failures",
                )
            )
            for pid in r["fails"][:1]:
                path.append(self._node(pid, "Pallet", "scan failure", ts=TS_SCAN))
        return CandidateChain(chain_id="scenario2", path=path, source="cypher")

    def shared_sku_chain(self) -> CandidateChain:
        self.last_meta = {}
        total = self.s.run(
            """
            MATCH (g:GlobalSKU {cohort:$c, planted_type:'shared_sku'})-[:USED_BY]->(cust:Customer)
            WITH g, collect(DISTINCT cust.customer_id) AS customers
            WHERE size(customers) > 1
            RETURN count(g) AS n
            """,
            c=COHORT,
        ).single()["n"]
        rows = self.s.run(
            """
            MATCH (g:GlobalSKU {cohort:$c, planted_type:'shared_sku'})-[:USED_BY]->(cust:Customer)
            WITH g, collect(DISTINCT cust.customer_id) AS customers
            WHERE size(customers) > 1
            OPTIONAL MATCH (t:TenantSKU)-[:MAPS_TO]->(g)
            RETURN g.sku_id AS sku,
                   g.brand_name AS brand_name,
                   g.brand_family AS brand_family,
                   g.package_category_name AS package_type,
                   customers,
                   collect(DISTINCT t.tenant_sku_id)[..4] AS tenants
            ORDER BY size(customers) DESC
            LIMIT $limit
            """,
            c=COHORT,
            limit=LIFECYCLE_TOP_N,
        ).data()
        self.last_meta["scenario5_shared_sku"] = {
            "shown": len(rows),
            "total": int(total or 0),
        }
        sc = self.scores()
        path = []
        if not rows:
            return CandidateChain(chain_id="scenario5", path=path, source="cypher")

        for r in rows:
            customers = r["customers"] or []
            cust_preview = ", ".join(str(c) for c in customers[:4])
            path.append(
                self._global_sku_node(
                    r,
                    ts=TS_TRAIN,
                    score=sc.get(r["sku"]),
                    detail=f"{len(customers)} customers ({cust_preview})",
                )
            )
        return CandidateChain(chain_id="scenario5", path=path, source="cypher")

    def auto_map_chain(self) -> CandidateChain:
        self.last_meta = {}
        total = self.s.run(
            """
            MATCH (g:GlobalSKU {cohort:$c, planted_type:'auto_map_error'})
            OPTIONAL MATCH (t:TenantSKU {match_method:'fuzzy'})-[:MAPS_TO]->(g)
            OPTIONAL MATCH (p:Pallet)-[:SCANNED_ON]->(g) WHERE p.outcome='failure'
            WITH g, collect(DISTINCT t.tenant_sku_id) AS tenants,
                 collect(DISTINCT p.pallet_id) AS fails
            WHERE size(tenants) >= 1 AND size(fails) >= 2
            RETURN count(g) AS n
            """,
            c=COHORT,
        ).single()["n"]
        rows = self.s.run(
            """
            MATCH (g:GlobalSKU {cohort:$c, planted_type:'auto_map_error'})
            OPTIONAL MATCH (t:TenantSKU {match_method:'fuzzy'})-[:MAPS_TO]->(g)
            OPTIONAL MATCH (p:Pallet)-[:SCANNED_ON]->(g) WHERE p.outcome='failure'
            WITH g, collect(DISTINCT t.tenant_sku_id) AS tenants,
                 collect(DISTINCT p.pallet_id) AS fails
            WHERE size(tenants) >= 1 AND size(fails) >= 2
            RETURN g.sku_id AS sku,
                   g.brand_name AS brand_name,
                   g.brand_family AS brand_family,
                   g.package_category_name AS package_type,
                   tenants, fails
            ORDER BY size(fails) DESC
            LIMIT $limit
            """,
            c=COHORT,
            limit=LIFECYCLE_TOP_N,
        ).data()
        self.last_meta["scenario6_auto_map"] = {
            "shown": len(rows),
            "total": int(total or 0),
        }
        sc = self.scores()
        path = []
        for r in rows:
            tid = r["tenants"][0]
            path.append(
                self._node(
                    tid,
                    "TenantSKU",
                    "fuzzy auto-map — wrong product category",
                    ts=TS_IMPORT,
                )
            )
            path.append(
                self._global_sku_node(
                    r,
                    ts=TS_TRAIN,
                    score=sc.get(r["sku"]),
                    detail="neighbourhood contradicts tenant record",
                )
            )
            for pid in r["fails"][:2]:
                path.append(self._node(pid, "Pallet", "scan failure after bad mapping", ts=TS_SCAN))
        return CandidateChain(chain_id="scenario6", path=path, source="cypher")

    def sku_root_cause_chain(self, sku_id: str) -> CandidateChain:
        """Distributed-failure chain for a single SKU (handbook PP1)."""
        row = self.s.run(
            """
            MATCH (g:GlobalSKU {sku_id: $sku, cohort: $c})
            OPTIONAL MATCH (t:TenantSKU)-[:MAPS_TO]->(g)
            OPTIONAL MATCH (g)-[:USED_BY]->(cust:Customer)
            OPTIONAL MATCH (g)-[:BELONGS_TO_BRAND]->(b:Brand)
            OPTIONAL MATCH (img:TrainingImage)-[:TRAINED_WITH]->(g)
            OPTIONAL MATCH (p:Pallet)-[:SCANNED_ON]->(g) WHERE p.outcome = 'failure'
            OPTIONAL MATCH (g)-[:MERGED_INTO]->(m:MergeEvent)
            RETURN g.sku_id AS sku,
                   g.brand_name AS brand_name,
                   g.brand_family AS brand_family,
                   g.package_category_name AS package_type,
                   coalesce(g.planted_type, '') AS planted,
                   collect(DISTINCT t.tenant_sku_id)[..3] AS tenants,
                   collect(DISTINCT cust.customer_id)[..4] AS customers,
                   b.brand_family AS linked_brand,
                   b.canonical AS canonical,
                   count(DISTINCT img) AS imgs,
                   collect(DISTINCT p.pallet_id)[..3] AS fails,
                   collect(DISTINCT m.merge_id)[..2] AS merges
            """,
            sku=sku_id,
            c=COHORT,
        ).single()
        sc = self.scores()
        path = []
        if not row:
            return CandidateChain(
                chain_id=f"sku_{sku_id}",
                path=[self._node(sku_id, "GlobalSKU", f"SKU {sku_id} not in cohort")],
                source="cypher",
            )

        for tid in row["tenants"] or []:
            path.append(
                self._node(tid, "TenantSKU", "tenant mapping / import", ts=TS_IMPORT)
            )
        for cid in row.get("customers") or []:
            path.append(
                self._node(cid, "Customer", "cross-customer dependency", ts=TS_IMPORT)
            )
        if row["merges"]:
            for mid in row["merges"]:
                path.append(
                    self._node(mid, "MergeEvent", "conflicted merge history", ts=TS_TRAIN)
                )
        brand_label = (
            row["linked_brand"]
            or row.get("brand_family")
            or row.get("brand_name")
            or "unknown brand"
        )
        if row["canonical"] is False:
            brand_label += " (non-canonical)"
        path.append(
            self._global_sku_node(
                row,
                ts=TS_TRAIN,
                score=sc.get(row["sku"]),
                detail=(
                    f"{row['imgs']} training images, {len(row['fails'] or [])} scan failures"
                    + (f" [{row['planted']}]" if row["planted"] else "")
                ),
            )
        )
        if row["linked_brand"]:
            path.append(self._node(brand_label, "Brand", f"brand: {brand_label}", ts=TS_TRAIN))
        for pid in row["fails"] or []:
            path.append(self._node(pid, "Pallet", "scan failure at inference", ts=TS_SCAN))
        return CandidateChain(chain_id=f"sku_{sku_id}", path=path, source="cypher")

    def shared_sku_blast_radius(self, sku_id: str) -> dict:
        """Cross-customer impact for a shared SKU (handbook PP5)."""
        row = self.s.run(
            """
            MATCH (g:GlobalSKU {sku_id: $sku})
            OPTIONAL MATCH (g)-[:USED_BY]->(c:Customer)
            OPTIONAL MATCH (t:TenantSKU)-[:MAPS_TO]->(g)
            OPTIONAL MATCH (g)-[:BELONGS_TO_BRAND]->(b:Brand)
            OPTIONAL MATCH (g)-[:HAS_PACKAGE]->(p:PackageType)
            OPTIONAL MATCH (pal:Pallet)-[:SCANNED_ON]->(g)
            RETURN g.sku_id AS sku,
                   g.brand_family AS brand,
                   coalesce(g.planted_type, '') AS planted,
                   collect(DISTINCT c.customer_id) AS customers,
                   count(DISTINCT t) AS tenant_mappings,
                   b.brand_family AS linked_brand,
                   p.package_category_name AS package,
                   count(DISTINCT pal) AS pallets,
                   sum(CASE WHEN pal.outcome = 'failure' THEN 1 ELSE 0 END) AS scan_failures
            """,
            sku=sku_id,
        ).single()
        if not row:
            return {"sku_id": sku_id, "found": False}
        customers = row["customers"] or []
        return {
            "sku_id": sku_id,
            "found": True,
            "brand": row["brand"],
            "planted_type": row["planted"] or None,
            "customers": customers,
            "customer_count": len(customers),
            "tenant_mappings": row["tenant_mappings"],
            "linked_brand": row["linked_brand"],
            "package": row["package"],
            "pallets": row["pallets"],
            "scan_failures": row["scan_failures"] or 0,
            "unsafe_to_change": len(customers) > 1,
        }

    def risk_rank(self, n: int = 20) -> list[tuple[str, float]]:
        return sorted(self.scores().items(), key=lambda kv: kv[1], reverse=True)[:n]

    def risk_rank_detailed(self, n: int = 20) -> list[dict]:
        """Top-N at-risk SKUs with brand_name and package_type for demo logs."""
        ranked = self.risk_rank(n)
        meta = self._sku_meta_map([sku for sku, _ in ranked])
        out = []
        for sku, score in ranked:
            row = meta.get(sku, {"sku": sku})
            out.append({
                "sku_id": sku,
                "score": score,
                "brand_name": row.get("brand_name"),
                "brand_family": row.get("brand_family"),
                "package_type": row.get("package_type"),
            })
        return out

    def risk_rank_chains(self, n: int = RISK_RANK_TOP_N) -> list[CandidateChain]:
        """Return one ranked chain so the Critic validates the full top-N list."""
        self.last_meta = {}
        all_scores = self.scores()
        ranked = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
        self.last_meta["scenario3_risk_rank"] = {
            "shown": len(ranked),
            "total": len(all_scores),
        }
        meta = self._sku_meta_map([sku for sku, _ in ranked])
        nodes = []
        for sku, score in ranked:
            row = meta.get(sku, {"sku": sku})
            node = self._global_sku_node(row, ts=TS_TRAIN, score=score)
            node.source = "anomaly_rank"
            nodes.append(node)
        if not nodes:
            return []
        return [
            CandidateChain(
                chain_id="scenario3_top20",
                path=nodes,
                source="anomaly_rank",
            )
        ]

    def closed_world_brand_dupes(self) -> list[dict]:
        return self.s.run(
            """
            MATCH (g:GlobalSKU)-[:BELONGS_TO_BRAND]->(b:Brand)
            WHERE b.flag = 'duplicate'
            RETURN g.sku_id AS sku,
                   g.brand_name AS brand_name,
                   g.brand_family AS brand_family,
                   g.package_category_name AS package_type
            LIMIT 20
            """
        ).data()

    def chain_for_scenario(self, scenario_num: int) -> list[CandidateChain]:
        builders = {
            1: self.brand_mismatch_chain,
            2: self.multi_signal_chain,
            5: self.shared_sku_chain,
            6: self.auto_map_chain,
        }
        if scenario_num == 3:
            return self.risk_rank_chains(RISK_RANK_TOP_N)
        if scenario_num in builders:
            return [builders[scenario_num]()]
        raise ValueError(f"unsupported scenario {scenario_num}")


def run_lifecycle_scenario(
    scenario_num: int,
    question: str | None = None,
    use_llm: bool = True,
) -> tuple[QuerySpec, TaskList, list[CandidateChain], list[dict] | None]:
    """
    Build candidates for a numbered demo scenario using lifecycle Cypher.

    Returns (spec, task_list, candidates, closed_world_rows).
    closed_world_rows is set only for scenario 4.

    use_llm is accepted for API compatibility; lifecycle Cypher paths do not
    require Bedrock (QuerySpec is built from scenario templates).
    """
    _ = use_llm
    question = question or SCENARIO_QUESTIONS[scenario_num]
    spec = spec_for_lifecycle_scenario(scenario_num, question)

    task_list = TaskList(
        spec=spec,
        tasks=[],
    )
    closed_rows = None

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        doer = LifecycleDoer(session)
        if scenario_num == 4:
            closed_rows = doer.closed_world_brand_dupes()
            chain = doer.brand_mismatch_chain()
            chain.chain_id = "scenario4_ab"
            candidates = [chain]
        else:
            candidates = doer.chain_for_scenario(scenario_num)
    driver.close()

    return spec, task_list, candidates, closed_rows


def run_sku_investigation(
    sku_id: str,
    question: str | None = None,
    use_llm: bool = True,
):
    """
    Root-cause chain for one SKU (Risk inbox → Investigate flow).
    Returns (spec, task_list, candidates, None).
    """
    from agents.supervisor import SupervisorAgent

    question = question or (
        f"Explain why GlobalSKU {sku_id} has a high anomaly score "
        f"and trace the distributed failure chain."
    )
    supervisor = SupervisorAgent(use_llm=use_llm)
    spec = supervisor.parse(question)
    task_list = TaskList(spec=spec, tasks=[])

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        doer = LifecycleDoer(session)
        candidates = [doer.sku_root_cause_chain(sku_id)]
    driver.close()

    return spec, task_list, candidates, None
