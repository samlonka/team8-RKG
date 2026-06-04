"""
03d_advanced_reflection.py — Three structural improvements to reflect_emb.

Each method is independent and additive — run all three to write the most
complete set of comparison signals before 11_ensemble.py combines them.

──────────────────────────────────────────────────────────────────────────────
Method 1 — Second-order reflection  (anomaly_reflect2)

  Uses neighbours' reflect_emb_attn (the best single-method embedding) as the
  aggregation input instead of self_emb.  The result is a 2-hop neighbourhood
  summary computed by composing two reflections.

  Why it helps: the brand-mismatch cascade in Scenario 1 spans four hops.
  One reflection captures the immediate neighbourhood; two reflections let the
  cascade signal propagate one hop further with no extra training.

  Writes: reflect2_emb  ·  anomaly_reflect2

──────────────────────────────────────────────────────────────────────────────
Method 2 — Temporal decay reflection  (anomaly_temporal)

  Applies exp(−λ × days_since_event) to the standard REL_WEIGHTS aggregation.
  A scan failure last week outweighs one from two years ago.

  λ = TEMPORAL_DECAY_LAMBDA (config.py).  Defaults to 0.01 (~70-day half-life).
  Nodes without a timestamp property receive decay = 1.0 (no discount).

  Uses: creation_date on GlobalSKU/Brand/PackageType;
        scan_timestamp on Pallet (when set by the import pipeline).

  Writes: reflect_emb_temporal  ·  anomaly_temporal

──────────────────────────────────────────────────────────────────────────────
Method 3 — Degree-normalised reflection  (anomaly_degnorm)

  Symmetric GCN normalisation: weight_ij = REL_WEIGHTS[r] / sqrt(d_i × d_j).
  Prevents hub nodes (a Brand connected to 5,000 GlobalSKUs) from dominating
  the reflect_emb of every entity that touches them.

  Uses sum-aggregation (not mean) since the normalisation already accounts for
  the number of neighbours.

  Writes: reflect_emb_degnorm  ·  anomaly_degnorm

Usage:
    python 03d_advanced_reflection.py [--method ALL]
    python 03d_advanced_reflection.py --method second_order
    python 03d_advanced_reflection.py --method temporal
    python 03d_advanced_reflection.py --method degree_norm
    python 03d_advanced_reflection.py --scores-only
"""

from __future__ import annotations

import argparse
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    REL_WEIGHTS, TEMPORAL_DECAY_LAMBDA,
    ANOMALY_HIGH_RISK, ANOMALY_MEDIUM_RISK, EMBEDDING_DIM,
)
from score_log import log_batch_run, ensure_indexes as _ensure_log_indexes

ALL_LABELS = [
    "GlobalSKU", "VendorSKU", "Brand", "PackageType",
    "Manufacturer", "Supplier", "ProductClass",
]

_NOW_DAYS = (datetime.now(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)).days


def _pk(label: str) -> str:
    return {
        "GlobalSKU":    "sku_id",    "VendorSKU":   "product_id",
        "Brand":        "brand_id",  "PackageType":  "package_type_id",
        "Manufacturer": "name",      "Supplier":     "name",
        "ProductClass": "name",
    }.get(label, "name")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def _l2(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


def _write_batch(session, label: str, pk: str, updates: list[dict],
                 emb_field: str, score_field: str):
    """Batch-write embedding + anomaly score to Neo4j nodes."""
    for i in range(0, len(updates), 300):
        chunk = updates[i : i + 300]
        session.run(
            f"""
            UNWIND $rows AS r
            MATCH (n:{label} {{{pk}: r.id}})
            SET n.{emb_field}  = r.emb,
                n.{score_field} = r.score
            """,
            rows=chunk,
        )


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — SECOND-ORDER REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

def batch_second_order(session, label: str) -> int:
    """
    Aggregate neighbours' reflect_emb_attn (falling back to reflect_emb)
    using the same REL_WEIGHTS as the baseline.  Writes reflect2_emb and
    anomaly_reflect2.
    """
    pk = _pk(label)

    # Use attention embedding if available; otherwise plain reflect_emb
    attn_available = session.run(
        f"MATCH (n:{label}) WHERE n.reflect_emb_attn IS NOT NULL RETURN count(n) AS c"
    ).single()["c"] > 0
    nb_emb_field = "reflect_emb_attn" if attn_available else "reflect_emb"

    ids = [r["id"] for r in session.run(
        f"MATCH (n:{label}) WHERE n.self_emb IS NOT NULL RETURN n.{pk} AS id"
    )]
    print(f"  {label}: {len(ids):,} nodes  (neighbour field: {nb_emb_field})")

    updates = []
    skipped = 0
    for eid in tqdm(ids, desc=f"  2nd-order {label}", unit="node"):
        # Fetch entity self_emb
        row = session.run(
            f"MATCH (e:{label} {{{pk}: $eid}}) RETURN e.self_emb AS se",
            eid=eid,
        ).single()
        if not row or not row["se"]:
            skipped += 1
            continue
        self_emb = np.array(row["se"], dtype=np.float32)

        # Fetch neighbours' reflect_emb (2nd-order signal)
        nb_rows = session.run(
            f"""
            MATCH (e:{label} {{{pk}: $eid}})-[r]-(n)
            WHERE n.{nb_emb_field} IS NOT NULL
            RETURN type(r) AS rel, n.{nb_emb_field} AS emb
            """,
            eid=eid,
        ).data()

        if not nb_rows:
            skipped += 1
            continue

        signals = [
            REL_WEIGHTS.get(r["rel"], REL_WEIGHTS["_DEFAULT"])
            * np.array(r["emb"], dtype=np.float32)
            for r in nb_rows
        ]
        vec = _l2(np.mean(signals, axis=0))
        if vec is None:
            skipped += 1
            continue

        score = round(1.0 - _cosine(self_emb / (np.linalg.norm(self_emb) + 1e-8), vec), 4)
        updates.append({"id": eid, "emb": vec.tolist(), "score": score})

    _write_batch(session, label, pk, updates, "reflect2_emb", "anomaly_reflect2")
    print(f"    → {len(updates):,} updated | {skipped:,} skipped")
    return len(updates)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — TEMPORAL DECAY REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

def _ts_to_days(ts_str: str | None) -> int | None:
    """Parse a date string and return days since epoch. Returns None on failure."""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(ts_str).strip()[:26], fmt).replace(tzinfo=timezone.utc)
            return (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).days
        except ValueError:
            continue
    return None


def _decay(ts_str: str | None) -> float:
    """exp(−λ × days_since_event). Returns 1.0 when timestamp is absent."""
    days = _ts_to_days(ts_str)
    if days is None:
        return 1.0
    elapsed = max(0, _NOW_DAYS - days)
    return max(0.05, math.exp(-TEMPORAL_DECAY_LAMBDA * elapsed))


def batch_temporal(session, label: str) -> int:
    """
    Standard REL_WEIGHTS aggregation with per-neighbour temporal decay.
    Fetches creation_date and scan_timestamp from neighbour nodes.
    Writes reflect_emb_temporal and anomaly_temporal.
    """
    pk = _pk(label)
    ids = [r["id"] for r in session.run(
        f"MATCH (n:{label}) WHERE n.self_emb IS NOT NULL RETURN n.{pk} AS id"
    )]
    print(f"  {label}: {len(ids):,} nodes  λ={TEMPORAL_DECAY_LAMBDA}")

    updates = []
    skipped = 0
    for eid in tqdm(ids, desc=f"  temporal {label}", unit="node"):
        row = session.run(
            f"MATCH (e:{label} {{{pk}: $eid}}) RETURN e.self_emb AS se",
            eid=eid,
        ).single()
        if not row or not row["se"]:
            skipped += 1
            continue
        self_emb = np.array(row["se"], dtype=np.float32)

        nb_rows = session.run(
            f"""
            MATCH (e:{label} {{{pk}: $eid}})-[r]-(n)
            WHERE n.self_emb IS NOT NULL
            RETURN type(r) AS rel, n.self_emb AS emb,
                   COALESCE(n.creation_date, n.scan_timestamp, '') AS ts
            """,
            eid=eid,
        ).data()

        if not nb_rows:
            skipped += 1
            continue

        signals = [
            REL_WEIGHTS.get(r["rel"], REL_WEIGHTS["_DEFAULT"])
            * _decay(r["ts"])
            * np.array(r["emb"], dtype=np.float32)
            for r in nb_rows
        ]
        vec = _l2(np.mean(signals, axis=0))
        if vec is None:
            skipped += 1
            continue

        score = round(1.0 - _cosine(self_emb / (np.linalg.norm(self_emb) + 1e-8), vec), 4)
        updates.append({"id": eid, "emb": vec.tolist(), "score": score})

    _write_batch(session, label, pk, updates, "reflect_emb_temporal", "anomaly_temporal")
    print(f"    → {len(updates):,} updated | {skipped:,} skipped")
    return len(updates)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — DEGREE-NORMALISED REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

def batch_degnorm(session, label: str, batch_size: int = 100) -> int:
    """
    weight_ij = REL_WEIGHTS[r] / sqrt(d_i × d_j)

    Prevents high-degree hub nodes (a Brand connected to 5,000 GlobalSKUs)
    from dominating the reflect_emb of every connected entity.
    Uses sum-aggregation (the normalisation already accounts for degree).
    Writes reflect_emb_degnorm and anomaly_degnorm.
    """
    pk   = _pk(label)
    ids  = [r["id"] for r in session.run(
        f"MATCH (n:{label}) WHERE n.self_emb IS NOT NULL RETURN n.{pk} AS id"
    )]
    print(f"  {label}: {len(ids):,} nodes")

    updates = []
    skipped = 0

    for start in tqdm(range(0, len(ids), batch_size), desc=f"  degnorm {label}", unit="batch"):
        chunk = ids[start : start + batch_size]

        # Compute entity degree in a WITH clause, then fetch neighbours.
        # size() with pattern expressions was removed in Neo4j 5 — use COUNT {}.
        rows = session.run(
            f"""
            UNWIND $ids AS eid
            MATCH (e:{label} {{{pk}: eid}})
            WITH e, eid, COUNT {{ (e)-[]-() }} AS e_deg
            OPTIONAL MATCH (e)-[r]-(n)
            WHERE n.self_emb IS NOT NULL
            RETURN eid,
                   e.self_emb                       AS se,
                   e_deg,
                   type(r)                          AS rel,
                   n.self_emb                       AS n_emb,
                   COUNT {{ (n)-[]-() }}            AS n_deg
            """,
            ids=chunk,
        ).data()

        # Group by entity
        by_eid: dict[str, list] = defaultdict(list)
        for r in rows:
            by_eid[r["eid"]].append(r)

        for eid, nbs in by_eid.items():
            se = nbs[0]["se"]
            if not se:
                skipped += 1
                continue
            self_emb = np.array(se, dtype=np.float32)
            e_deg    = max(1, nbs[0]["e_deg"] or 1)

            signals = []
            for nb in nbs:
                if nb["n_emb"] is None or nb["rel"] is None:
                    continue
                n_deg  = max(1, nb["n_deg"] or 1)
                norm_f = 1.0 / math.sqrt(e_deg * n_deg)
                w      = REL_WEIGHTS.get(nb["rel"], REL_WEIGHTS["_DEFAULT"])
                signals.append(w * norm_f * np.array(nb["n_emb"], dtype=np.float32))

            if not signals:
                skipped += 1
                continue

            vec = _l2(np.sum(signals, axis=0))
            if vec is None:
                skipped += 1
                continue

            score = round(1.0 - _cosine(self_emb / (np.linalg.norm(self_emb) + 1e-8), vec), 4)
            updates.append({"id": eid, "emb": vec.tolist(), "score": score})

    _write_batch(session, label, pk, updates, "reflect_emb_degnorm", "anomaly_degnorm")
    print(f"    → {len(updates):,} updated | {skipped:,} skipped")
    return len(updates)


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY TABLE
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_SCORE = {
    "second_order": "anomaly_reflect2",
    "temporal":     "anomaly_temporal",
    "degree_norm":  "anomaly_degnorm",
}

_METHOD_NAME = {
    "second_order": "Second-order",
    "temporal":     "Temporal decay",
    "degree_norm":  "Degree-norm",
}


def print_top(session, label: str, method: str, top_n: int):
    pk       = _pk(label)
    prop     = _METHOD_SCORE[method]
    rows     = session.run(
        f"""
        MATCH (n:{label}) WHERE n.{prop} IS NOT NULL
        RETURN n.{pk} AS id, n.{prop} AS score
        ORDER BY score DESC LIMIT $n
        """,
        n=top_n,
    ).data()
    if not rows:
        return
    name = _METHOD_NAME[method]
    print(f"\n── {name}: {label} {'─'*35}")
    print(f"  {'ID':<18} {'Score':<8} Risk")
    print(f"  {'-'*18} {'-'*8} {'-'*6}")
    for r in rows:
        risk = ("HIGH"   if r["score"] >= ANOMALY_HIGH_RISK   else
                "MEDIUM" if r["score"] >= ANOMALY_MEDIUM_RISK else "LOW")
        print(f"  {str(r['id']):<18} {r['score']:<8.4f} {risk}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

VALID_METHODS = ("second_order", "temporal", "degree_norm", "ALL")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced reflection variants: second-order, temporal, degree-norm"
    )
    parser.add_argument("--method", default="ALL", choices=VALID_METHODS,
                        help="Which variant to compute (default: ALL)")
    parser.add_argument("--label",  default="ALL",
                        help="Node label to process (default: ALL base labels)")
    parser.add_argument("--top",    type=int, default=20)
    parser.add_argument("--scores-only", action="store_true",
                        help="Skip computation, just print current scores")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    labels = ALL_LABELS if args.label == "ALL" else [args.label]
    methods = (
        ["second_order", "temporal", "degree_norm"]
        if args.method == "ALL" else [args.method]
    )

    with driver.session() as session:
        _ensure_log_indexes(session)

        if not args.scores_only:
            run_id = str(uuid.uuid4())
            for method in methods:
                print(f"\n── Computing {_METHOD_NAME[method]} reflection ──────────────────────")
                for label in labels:
                    if method == "second_order":
                        batch_second_order(session, label)
                    elif method == "temporal":
                        batch_temporal(session, label)
                    else:
                        batch_degnorm(session, label)

            # Append to score log
            method_to_log = {
                "second_order": "reflect2",
                "temporal":     "temporal",
                "degree_norm":  "degnorm",
            }
            print("\n── Appending to score log ────────────────────────────────────────")
            for method in methods:
                log_key = method_to_log[method]
                for label in labels:
                    _, n = log_batch_run(session, label, log_key, run_id=run_id)
                    if n:
                        print(f"  {label} / {log_key}: {n:,} entries  run={run_id[:8]}…")

        print("\n── Top anomalies ─────────────────────────────────────────────────")
        for method in methods:
            for label in labels:
                print_top(session, label, method, args.top)

    driver.close()
    print("\nAdvanced reflection complete.\n")


if __name__ == "__main__":
    main()
