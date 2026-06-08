"""
score_log.py — Anomaly score history and drift tracking.

Implements the brief's "Anomaly score log" requirement:
    Entity ID, anomaly score, computed_at, method — for tracking score
    history and measuring drift across batch runs.

Each call to log_batch_run() creates one ScoreLog node per scored entity,
tagged with a shared run_id (UUID). Multiple runs accumulate in Neo4j,
enabling drift analysis: which SKUs are getting worse between runs?

ScoreLog node properties:
    log_id        UUID, unique per entry
    entity_id     primary key value of the entity (e.g. sku_id)
    entity_label  Neo4j label (e.g. GlobalSKU)
    method        baseline | phase1 | phase2 | phase3 | phase4
    score         anomaly score [0, 1]
    computed_at   ISO-8601 UTC timestamp
    run_id        UUID shared across all entries written in one batch

Supported methods and the Neo4j property each reads:
    baseline  — computed from self_emb + reflect_emb (1 - cosine)
    phase1    — anomaly_dir
    phase2    — anomaly_rgcn
    phase3    — triple_anomaly_score
    phase4    — dominant_score

Usage:
    python score_log.py --runs
    python score_log.py --history <entity_id> [--method baseline]
    python score_log.py --drift   [--label GlobalSKU] [--method baseline] [--top 20]
    python score_log.py --purge-before 2025-01-01
"""

from __future__ import annotations

import argparse
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ── method → stored Neo4j property (None = compute from embeddings) ──────────
SCORE_PROPERTY: dict[str, str | None] = {
    "baseline":  None,                   # 03  — computed from self_emb + reflect_emb
    "attention": "anomaly_attn",         # 03c — divergence-weighted soft attention
    "phase1":    "anomaly_dir",          # 03b — direction-split + severity
    "phase2":    "anomaly_rgcn",         # 08  — R-GCN encoder
    "phase3":    "triple_anomaly_score", # 09  — RotatE triple scoring
    "phase4":    "dominant_score",       # 10  — DOMINANT joint attr+struct
    "reflect2":  "anomaly_reflect2",     # 03d — second-order (2-hop) reflection
    "temporal":  "anomaly_temporal",     # 03d — temporal decay weighting
    "degnorm":   "anomaly_degnorm",      # 03d — degree-normalised aggregation
    "lof":       "anomaly_lof",          # 11  — Local Outlier Factor
    "ensemble":  "anomaly_ensemble",     # 11  — logistic regression ensemble
}

PK_MAP = {
    "GlobalSKU":     "sku_id",
    "TenantSKU":     "tenant_sku_id",
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
}


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def ensure_indexes(session) -> None:
    """Create ScoreLog indexes if they don't already exist."""
    session.run(
        "CREATE INDEX idx_scorelog_entity_id IF NOT EXISTS "
        "FOR (n:ScoreLog) ON (n.entity_id)"
    )
    session.run(
        "CREATE INDEX idx_scorelog_run_id IF NOT EXISTS "
        "FOR (n:ScoreLog) ON (n.run_id)"
    )
    session.run(
        "CREATE INDEX idx_scorelog_computed_at IF NOT EXISTS "
        "FOR (n:ScoreLog) ON (n.computed_at)"
    )
    session.run(
        "CREATE CONSTRAINT scorelog_log_id_unique IF NOT EXISTS "
        "FOR (n:ScoreLog) REQUIRE n.log_id IS UNIQUE"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORE FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _cos(a, b) -> float:
    va, vb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 1e-8 else 0.0


def _fetch_scores(session, label: str, method: str) -> dict[str, float]:
    """Fetch {entity_id: score} for all scored entities of this label + method."""
    pk   = PK_MAP.get(label, "name")
    prop = SCORE_PROPERTY.get(method)

    if prop is None:
        # baseline: compute from the two embedding vectors
        rows = session.run(
            f"""
            MATCH (n:{label})
            WHERE n.self_emb IS NOT NULL AND n.reflect_emb IS NOT NULL
            RETURN n.{pk} AS id, n.self_emb AS se, n.reflect_emb AS re
            """
        ).data()
        return {
            str(r["id"]): round(1.0 - _cos(r["se"], r["re"]), 4)
            for r in rows
        }

    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.{prop} IS NOT NULL
        RETURN n.{pk} AS id, n.{prop} AS score
        """
    ).data()
    return {str(r["id"]): round(float(r["score"]), 4) for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_batch_run(
    session,
    label:   str,
    method:  str = "baseline",
    run_id:  str | None = None,
) -> tuple[str, int]:
    """
    Fetch current anomaly scores for all entities of `label` under `method`,
    write one ScoreLog node per entity, and return (run_id, count).

    All entries written in this call share the same run_id, making it
    possible to query "show me all scores from run X."
    """
    if run_id is None:
        run_id = str(uuid.uuid4())
    computed_at = datetime.now(timezone.utc).isoformat()

    scores = _fetch_scores(session, label, method)
    if not scores:
        return run_id, 0

    rows = [
        {
            "log_id":       str(uuid.uuid4()),
            "entity_id":    eid,
            "entity_label": label,
            "method":       method,
            "score":        score,
            "computed_at":  computed_at,
            "run_id":       run_id,
        }
        for eid, score in scores.items()
    ]

    batch = 500
    for i in range(0, len(rows), batch):
        session.run(
            """
            UNWIND $rows AS r
            CREATE (:ScoreLog {
                log_id:       r.log_id,
                entity_id:    r.entity_id,
                entity_label: r.entity_label,
                method:       r.method,
                score:        r.score,
                computed_at:  r.computed_at,
                run_id:       r.run_id
            })
            """,
            rows=rows[i : i + batch],
        )

    return run_id, len(rows)


def log_all_methods(session, label: str) -> str:
    """
    Log all available methods for a label in a single shared run_id.
    Skips methods that have no scored nodes.
    """
    run_id = str(uuid.uuid4())
    total  = 0
    for method in SCORE_PROPERTY:
        _, n = log_batch_run(session, label, method, run_id=run_id)
        if n:
            print(f"    {method:<10} {n:>6,} entries logged")
            total += n
    print(f"  run_id={run_id}  total={total:,} entries")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_history(
    session,
    entity_id: str,
    method:    str | None = None,
    limit:     int = 50,
) -> list[dict]:
    """
    Return score history for a single entity, newest first.
    If method is None, returns entries for all methods.
    """
    method_filter = "AND l.method = $method" if method else ""
    rows = session.run(
        f"""
        MATCH (l:ScoreLog {{entity_id: $eid}})
        WHERE true {method_filter}
        RETURN l.method AS method, l.score AS score,
               l.computed_at AS computed_at, l.run_id AS run_id,
               l.entity_label AS label
        ORDER BY l.computed_at DESC
        LIMIT $lim
        """,
        eid=str(entity_id), method=method, lim=limit,
    ).data()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def drift_report(
    session,
    label:  str  = "GlobalSKU",
    method: str  = "baseline",
    top_n:  int  = 20,
) -> list[dict]:
    """
    Compare each entity's most recent score to its previous score.
    Returns the top_n entities by absolute drift, sorted descending.

    A positive drift means the anomaly score increased (getting worse).
    A negative drift means it improved.
    """
    rows = session.run(
        """
        MATCH (l:ScoreLog {entity_label: $label, method: $method})
        RETURN l.entity_id AS eid, l.score AS score, l.computed_at AS at
        ORDER BY l.entity_id, l.computed_at ASC
        """,
        label=label, method=method,
    ).data()

    # Group by entity, keep chronological order (already sorted)
    by_entity: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        by_entity[r["eid"]].append((r["at"], float(r["score"])))

    drifts = []
    for eid, entries in by_entity.items():
        if len(entries) < 2:
            continue
        prev_at,  prev_score = entries[-2]
        curr_at,  curr_score = entries[-1]
        delta = round(curr_score - prev_score, 4)
        drifts.append({
            "entity_id":  eid,
            "prev_score": round(prev_score, 4),
            "curr_score": round(curr_score, 4),
            "drift":      delta,
            "trend":      "↑ WORSE" if delta > 0 else ("↓ BETTER" if delta < 0 else "→"),
            "prev_at":    prev_at[:19],
            "curr_at":    curr_at[:19],
            "n_runs":     len(entries),
        })

    drifts.sort(key=lambda x: abs(x["drift"]), reverse=True)
    return drifts[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# RUN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def runs_summary(session, label: str | None = None) -> list[dict]:
    """
    Return a summary row per (run_id, method, entity_label), ordered by run time.
    """
    label_filter = "AND l.entity_label = $label" if label else ""
    rows = session.run(
        f"""
        MATCH (l:ScoreLog)
        WHERE true {label_filter}
        WITH l.run_id AS run_id, l.method AS method, l.entity_label AS lbl,
             min(l.computed_at) AS run_started_at,
             count(l)           AS n_entities,
             round(avg(l.score), 4) AS mean_score,
             round(max(l.score), 4) AS max_score,
             round(min(l.score), 4) AS min_score
        RETURN run_id, method, lbl, run_started_at,
               n_entities, mean_score, max_score, min_score
        ORDER BY run_started_at DESC
        """,
        label=label,
    ).data()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# PURGE
# ─────────────────────────────────────────────────────────────────────────────

def purge_before(session, before_date: str) -> int:
    """Delete all ScoreLog entries with computed_at < before_date. Returns count deleted."""
    result = session.run(
        """
        MATCH (l:ScoreLog)
        WHERE l.computed_at < $before
        WITH l, count(l) AS n
        DELETE l
        RETURN count(l) AS deleted
        """,
        before=before_date,
    )
    row = result.single()
    return row["deleted"] if row else 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_runs(rows: list[dict]):
    if not rows:
        print("  No runs logged yet.")
        return
    print(f"\n{'Run started':<22} {'Method':<10} {'Label':<14} "
          f"{'Entities':>8} {'Mean':>7} {'Min':>7} {'Max':>7}")
    print(f"{'-'*22} {'-'*10} {'-'*14} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    for r in rows:
        print(
            f"{str(r['run_started_at'])[:19]:<22} "
            f"{r['method']:<10} "
            f"{r['lbl']:<14} "
            f"{r['n_entities']:>8,} "
            f"{r['mean_score']:>7.4f} "
            f"{r['min_score']:>7.4f} "
            f"{r['max_score']:>7.4f}"
        )


def _print_history(entity_id: str, rows: list[dict]):
    if not rows:
        print(f"  No score log entries for entity '{entity_id}'.")
        return
    print(f"\n  Score history for entity: {entity_id}")
    print(f"  {'Method':<10} {'Score':<8} {'Computed at':<22} {'Run ID'}")
    print(f"  {'-'*10} {'-'*8} {'-'*22} {'-'*36}")
    for r in rows:
        print(
            f"  {r['method']:<10} {r['score']:<8.4f} "
            f"{str(r['computed_at'])[:19]:<22} {r['run_id']}"
        )


def _print_drift(label: str, method: str, rows: list[dict]):
    if not rows:
        print(f"  No drift data — need at least 2 runs for {label}/{method}.")
        return
    print(f"\n── Drift report: {label}  method={method} ─────────────────────────────")
    print(f"  {'Entity ID':<20} {'Prev':>7} {'Curr':>7} {'Drift':>8} {'Trend':<12} Runs")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*8} {'-'*12} {'-'*4}")
    for r in rows:
        print(
            f"  {str(r['entity_id']):<20} "
            f"{r['prev_score']:>7.4f} "
            f"{r['curr_score']:>7.4f} "
            f"{r['drift']:>+8.4f} "
            f"{r['trend']:<12} "
            f"{r['n_runs']:>4}"
        )
    improving = sum(1 for r in rows if r["drift"] < 0)
    worsening = sum(1 for r in rows if r["drift"] > 0)
    print(f"\n  {worsening} entities worsening  |  {improving} improving")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anomaly score log — history and drift")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs",          action="store_true",
                       help="Show summary of all batch runs")
    group.add_argument("--history",       metavar="ENTITY_ID",
                       help="Show score history for one entity")
    group.add_argument("--drift",         action="store_true",
                       help="Show entities with biggest score change between runs")
    group.add_argument("--log-now",       action="store_true",
                       help="Log current scores for all methods into the score log")
    group.add_argument("--purge-before",  metavar="DATE",
                       help="Delete log entries older than DATE (YYYY-MM-DD)")

    parser.add_argument("--label",  default="GlobalSKU",
                        help="Node label (default: GlobalSKU)")
    parser.add_argument("--method", default="baseline",
                        choices=list(SCORE_PROPERTY),
                        help="Scoring method (default: baseline)")
    parser.add_argument("--top",    type=int, default=20,
                        help="Top N results for drift report (default: 20)")

    args = parser.parse_args()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        ensure_indexes(session)

        if args.runs:
            rows = runs_summary(session, label=None if args.label == "ALL" else args.label)
            _print_runs(rows)

        elif args.history:
            rows = get_history(session, args.history,
                               method=None if args.method == "ALL" else args.method)
            _print_history(args.history, rows)

        elif args.drift:
            rows = drift_report(session, label=args.label,
                                method=args.method, top_n=args.top)
            _print_drift(args.label, args.method, rows)

        elif args.log_now:
            print(f"\n── Logging current scores: {args.label} ─────────────────────────")
            log_all_methods(session, args.label)

        elif args.purge_before:
            n = purge_before(session, args.purge_before)
            print(f"  Deleted {n:,} ScoreLog entries before {args.purge_before}")

    driver.close()


if __name__ == "__main__":
    main()
