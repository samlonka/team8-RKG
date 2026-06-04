"""
03b_reflection_enhanced.py — Phase 1: Direction-split + edge-property-weighted reflection

Improvements over 03_reflection.py (baseline):
  1. Direction-aware aggregation — incoming (operational) vs outgoing (definitional)
     neighbours are aggregated separately, then combined into one vector.
     Incoming = things that point TO this entity (scan events, training images, tenant maps).
     Outgoing = things this entity points TO (brand, package type, merge events, customers).
  2. Edge-property severity multipliers — neighbour node properties amplify the base
     REL_WEIGHTS scalar when the node itself signals a problem:
       Pallet.outcome='failure'      → ×1.5 on SCANNED_ON weight
       MergeEvent.status='conflicted' → ×2.0 on MERGED_INTO weight
       MergeEvent.rollback_available=False → additional ×1.3
       TenantSKU.match_method='fuzzy' → ×1.5 on MAPS_TO weight

Writes per node:
  reflect_emb_dir  — 768-dim direction-aware reflection vector
  anomaly_dir      — pre-computed 1 - cosine(self_emb, reflect_emb_dir)

Usage:
    python 03b_reflection_enhanced.py [--label GlobalSKU] [--top 20]
    python 03b_reflection_enhanced.py --scores-only
"""

import argparse
import uuid
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    REL_WEIGHTS, EDGE_SEVERITY,
    ANOMALY_HIGH_RISK, ANOMALY_MEDIUM_RISK, EMBEDDING_DIM,
)
from score_log import log_batch_run, ensure_indexes as ensure_score_log_indexes

ALL_LABELS = [
    "GlobalSKU", "VendorSKU", "Brand", "PackageType",
    "Manufacturer", "Supplier", "ProductClass",
]


def _pk(label: str) -> str:
    return {
        "GlobalSKU":     "sku_id",
        "VendorSKU":     "product_id",
        "Brand":         "brand_id",
        "PackageType":   "package_type_id",
        "Manufacturer":  "name",
        "Supplier":      "name",
        "ProductClass":  "name",
        "Customer":      "customer_id",
        "TenantSKU":     "tenant_sku_id",
        "TrainingImage": "image_id",
        "MergeEvent":    "merge_id",
        "Pallet":        "pallet_id",
    }.get(label, "name")


def _severity(outcome: str, status: str, rollback_avail, match_method: str) -> float:
    """Multiplicative severity on top of REL_WEIGHTS from neighbour node properties."""
    mult = 1.0
    if outcome == "failure":
        mult *= EDGE_SEVERITY["outcome:failure"]
    if status == "conflicted":
        mult *= EDGE_SEVERITY["status:conflicted"]
    if rollback_avail is False:
        mult *= EDGE_SEVERITY["rollback_available:False"]
    if match_method == "fuzzy":
        mult *= EDGE_SEVERITY["match_method:fuzzy"]
    return mult


def _fetch_directed(session, label: str, pk: str, eid: str) -> list[tuple]:
    """
    Return (direction, weight, emb_array) tuples for all neighbours.
    Runs two separate Cypher queries to capture direction explicitly.
    """
    signals = []

    # Incoming: nodes that point TO this entity (operational context)
    for row in session.run(
        f"""
        MATCH (n)-[r]->(e:{label} {{{pk}: $eid}})
        WHERE n.self_emb IS NOT NULL
        RETURN type(r) AS rel,
               n.self_emb AS emb,
               COALESCE(n.outcome, '')          AS outcome,
               COALESCE(n.status, '')           AS status,
               COALESCE(n.rollback_available, true) AS ra,
               COALESCE(n.match_method, '')     AS mm
        """,
        eid=eid,
    ):
        w = REL_WEIGHTS.get(row["rel"], REL_WEIGHTS["_DEFAULT"])
        w *= _severity(row["outcome"], row["status"], row["ra"], row["mm"])
        signals.append(("in", w, np.array(row["emb"], dtype=np.float32)))

    # Outgoing: nodes this entity points TO (definitional context)
    for row in session.run(
        f"""
        MATCH (e:{label} {{{pk}: $eid}})-[r]->(n)
        WHERE n.self_emb IS NOT NULL
        RETURN type(r) AS rel,
               n.self_emb AS emb,
               COALESCE(n.outcome, '')          AS outcome,
               COALESCE(n.status, '')           AS status,
               COALESCE(n.rollback_available, true) AS ra,
               COALESCE(n.match_method, '')     AS mm
        """,
        eid=eid,
    ):
        w = REL_WEIGHTS.get(row["rel"], REL_WEIGHTS["_DEFAULT"])
        w *= _severity(row["outcome"], row["status"], row["ra"], row["mm"])
        signals.append(("out", w, np.array(row["emb"], dtype=np.float32)))

    return signals


def _aggregate(signals: list[tuple]) -> list[float] | None:
    """
    Weighted mean per direction, L2-normalize each, then average and re-normalize.
    Separating directions before averaging prevents the operational (incoming) and
    definitional (outgoing) signals from cancelling each other.
    """
    parts = []
    for direction in ("in", "out"):
        weighted = [w * emb for d, w, emb in signals if d == direction]
        if not weighted:
            continue
        v = np.mean(weighted, axis=0)
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            parts.append(v / norm)

    if not parts:
        return None

    combined = np.mean(parts, axis=0)
    norm = np.linalg.norm(combined)
    return (combined / norm).tolist() if norm > 1e-8 else None


def _cosine(a, b) -> float:
    va, vb = np.array(a, np.float32), np.array(b, np.float32)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 1e-8 else 0.0


def batch_compute(session, label: str) -> int:
    pk = _pk(label)
    ids = [r["id"] for r in session.run(
        f"MATCH (n:{label}) WHERE n.self_emb IS NOT NULL RETURN n.{pk} AS id"
    )]
    print(f"  {label}: {len(ids):,} nodes")

    updated = skipped = 0
    for eid in tqdm(ids, desc=f"  {label}", unit="node"):
        signals = _fetch_directed(session, label, pk, eid)
        reflect = _aggregate(signals)
        if reflect is None:
            skipped += 1
            continue

        row = session.run(
            f"MATCH (n:{label} {{{pk}: $eid}}) RETURN n.self_emb AS se",
            eid=eid,
        ).single()
        se = row["se"] if row else None
        anomaly_dir = round(1.0 - _cosine(se, reflect), 4) if se else None

        props: dict = {"reflect_emb_dir": reflect}
        if anomaly_dir is not None:
            props["anomaly_dir"] = anomaly_dir

        session.run(
            f"MATCH (n:{label} {{{pk}: $eid}}) SET n += $props",
            eid=eid, props=props,
        )
        updated += 1

    print(f"    → {updated:,} updated | {skipped:,} skipped (no neighbours)")
    return updated


def rank_anomalies(session, label: str, top_n: int) -> list[dict]:
    pk = _pk(label)
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.anomaly_dir IS NOT NULL
        RETURN n.{pk} AS id, n.anomaly_dir AS score
        ORDER BY score DESC LIMIT $n
        """,
        n=top_n,
    ).data()
    return [
        {
            "id": r["id"],
            "score": r["score"],
            "risk": (
                "HIGH"   if r["score"] >= ANOMALY_HIGH_RISK   else
                "MEDIUM" if r["score"] >= ANOMALY_MEDIUM_RISK else
                "LOW"
            ),
        }
        for r in rows
    ]


def print_table(label: str, ranked: list[dict]):
    print(f"\n── Phase 1 top anomalies: {label} {'─' * 30}")
    print(f"  {'ID':<18} {'Score':<8} Risk")
    print(f"  {'-'*18} {'-'*8} {'-'*6}")
    for r in ranked:
        print(f"  {str(r['id']):<18} {r['score']:<8.4f} {r['risk']}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: direction-split + severity-weighted reflection"
    )
    parser.add_argument("--label", default="ALL",
                        help="Node label to process (default: ALL)")
    parser.add_argument("--top",   type=int, default=20)
    parser.add_argument("--scores-only", action="store_true",
                        help="Skip recomputation, only print current anomaly_dir scores")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    labels = ALL_LABELS if args.label == "ALL" else [args.label]

    with driver.session() as session:
        ensure_score_log_indexes(session)

        # Vector index for reflect_emb_dir — needed for ANN queries in agent pipeline
        session.run(f"""
            CREATE VECTOR INDEX idx_global_sku_reflect_dir IF NOT EXISTS
            FOR (n:GlobalSKU) ON n.reflect_emb_dir
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)

        if not args.scores_only:
            print("\n── Phase 1: Computing direction-split reflect_emb_dir ──────────")
            for label in labels:
                batch_compute(session, label)

        print("\n── Phase 1: Top anomalies (anomaly_dir) ─────────────────────────")
        for label in labels:
            ranked = rank_anomalies(session, label, args.top)
            if ranked:
                print_table(label, ranked)

        # ── Append to score log ───────────────────────────────────────────
        print("\n── Phase 1: Appending to score log ──────────────────────────────")
        run_id = str(uuid.uuid4())
        for label in labels:
            _, n = log_batch_run(session, label, "phase1", run_id=run_id)
            if n:
                print(f"  {label}: {n:,} log entries  run={run_id[:8]}…")

    driver.close()
    print("\nPhase 1 complete.\n")


if __name__ == "__main__":
    main()
