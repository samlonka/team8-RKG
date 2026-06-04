"""
03_reflection.py — Compute reflect_emb and anomaly scores

Pipeline:
  1. For each entity, fetch all neighbours from Neo4j
  2. Weight neighbours by REL_WEIGHTS × edge severity × divergence (attention)
  3. Aggregate → softmax attention → L2-normalize → reflect_emb
  4. Write reflect_emb back to Neo4j node
  5. Compute and print anomaly score ranking (1 - cosine similarity)

reflect_emb encodes what the neighbourhood collectively implies about an entity —
regardless of what that entity says about itself. Divergence = anomaly signal.

Usage:
    python 03_reflection.py [--label GlobalSKU] [--top 20]
"""

import argparse
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    ANOMALY_HIGH_RISK, ANOMALY_MEDIUM_RISK,
)
from reflection_core import (
    pk as _pk,
    compute_reflect_emb,
    anomaly_score,
    cosine_similarity,
)
from score_log import log_batch_run, ensure_indexes as ensure_score_log_indexes

# ─────────────────────────────────────────────────────────────────────────────
# REFLECTION VECTOR COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_entity_ids(session, label: str) -> list[str]:
    result = session.run(
        f"MATCH (n:{label}) WHERE n.self_emb IS NOT NULL RETURN n.{_pk(label)} AS id"
    )
    return [r["id"] for r in result]


def compute_reflection(entity_id: str, label: str, session) -> list[float] | None:
    """Attention-weighted reflection (see reflection_core)."""
    return compute_reflect_emb(session, entity_id, label)


def write_reflect_emb(session, label: str, pk_field: str, entity_id: str, reflect: list[float]):
    session.run(
        f"MATCH (n:{label} {{{pk_field}: $eid}}) SET n.reflect_emb = $emb",
        eid=entity_id, emb=reflect,
    )


def batch_compute_reflections(session, label: str, batch_size: int = 100):
    """
    Compute and write reflect_emb for all entities of a given label.
    Skips entities with no neighbours (reflect_emb stays null → excluded from scoring).
    """
    pk = _pk(label)
    ids = fetch_all_entity_ids(session, label)
    print(f"  Computing reflect_emb for {len(ids):,} {label} nodes ...")

    updated = 0
    skipped = 0

    for entity_id in tqdm(ids, desc=f"  {label}", unit="node"):
        reflect = compute_reflection(entity_id, label, session)
        if reflect is not None:
            write_reflect_emb(session, label, pk, entity_id, reflect)
            updated += 1
        else:
            skipped += 1

    print(f"    → {updated:,} updated | {skipped:,} skipped (no neighbours)")
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY SCORE QUERY
# ─────────────────────────────────────────────────────────────────────────────

def rank_anomalies(session, label: str, top_n: int = 20) -> list[dict]:
    """
    Fetch all entities with both embeddings, compute anomaly scores in Python,
    return sorted descending list.

    Note: In production this would be a pure Cypher/GDS computation.
    For POC, Python-side scoring is cleaner and avoids GDS licensing.
    """
    pk = _pk(label)

    # Pull label-specific display fields
    label_field = {
        "GlobalSKU":    "n.brand_family AS brand, n.package_category_name AS pkg, n.upc_missing AS upc_missing",
        "VendorSKU":    "n.brand AS brand, n.product_description AS pkg, false AS upc_missing",
        "Brand":        "n.brand_family AS brand, '' AS pkg, false AS upc_missing",
        "PackageType":  "n.package_category_name AS brand, '' AS pkg, false AS upc_missing",
        "Manufacturer": "n.name AS brand, '' AS pkg, false AS upc_missing",
        "Supplier":     "n.name AS brand, '' AS pkg, false AS upc_missing",
        "ProductClass": "n.name AS brand, '' AS pkg, false AS upc_missing",
    }.get(label, "n.name AS brand, '' AS pkg, false AS upc_missing")

    result = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.self_emb IS NOT NULL AND n.reflect_emb IS NOT NULL
        RETURN n.{pk} AS id, {label_field},
               n.self_emb AS self_emb, n.reflect_emb AS reflect_emb
        """
    )

    scored = []
    for row in result:
        score = anomaly_score(row["self_emb"], row["reflect_emb"])
        risk  = (
            "HIGH"   if score >= ANOMALY_HIGH_RISK   else
            "MEDIUM" if score >= ANOMALY_MEDIUM_RISK else
            "LOW"
        )
        scored.append({
            "id":          row["id"],
            "brand":       row.get("brand", ""),
            "pkg":         row.get("pkg", ""),
            "upc_missing": row.get("upc_missing", False),
            "score":       score,
            "risk":        risk,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def print_anomaly_table(label: str, ranked: list[dict]):
    print(f"\n── Top anomalies: {label} ─────────────────────────────────────")
    print(f"  {'ID':<15} {'Brand/Name':<30} {'Package':<20} {'UPC?':<6} {'Score':<7} {'Risk'}")
    print(f"  {'-'*15} {'-'*30} {'-'*20} {'-'*6} {'-'*7} {'-'*6}")
    for r in ranked:
        upc_flag = "MISS" if r.get("upc_missing") else "ok"
        pkg      = str(r.get("pkg", ""))[:20]
        brand    = str(r.get("brand", ""))[:30]
        print(
            f"  {str(r['id']):<15} {brand:<30} {pkg:<20} {upc_flag:<6} {r['score']:<7} {r['risk']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADDITIONAL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def write_baseline_scores(session, label: str) -> int:
    """
    Compute 1 - cosine(self_emb, reflect_emb) for every entity that has both
    vectors and store the result as anomaly_baseline on the node.

    This makes the baseline score readable by score_log without recomputing
    embeddings, and lets 06b_evaluate_methods.py query it like any other method.
    """
    pk = _pk(label)
    result = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.self_emb IS NOT NULL AND n.reflect_emb IS NOT NULL
        RETURN n.{pk} AS id, n.self_emb AS se, n.reflect_emb AS re
        """
    )
    rows = []
    for row in result:
        score = anomaly_score(row["se"], row["re"])
        rows.append({"id": row["id"], "score": score})

    batch = 500
    for i in range(0, len(rows), batch):
        session.run(
            f"""
            UNWIND $rows AS r
            MATCH (n:{label} {{{pk}: r.id}})
            SET n.anomaly_baseline = r.score
            """,
            rows=rows[i : i + batch],
        )
    return len(rows)


def score_distribution(session, label: str) -> dict:
    """
    Return counts by risk band. Useful for benchmarking.
    """
    pk = _pk(label)
    result = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.self_emb IS NOT NULL AND n.reflect_emb IS NOT NULL
        RETURN n.{pk} AS id, n.self_emb AS se, n.reflect_emb AS re
        """
    )

    dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "total": 0}
    for row in result:
        score = anomaly_score(row["se"], row["re"])
        dist["total"] += 1
        if score >= ANOMALY_HIGH_RISK:
            dist["HIGH"] += 1
        elif score >= ANOMALY_MEDIUM_RISK:
            dist["MEDIUM"] += 1
        else:
            dist["LOW"] += 1
    return dist


def missing_upc_anomaly_check(session) -> None:
    """
    Cross-check: GlobalSKUs with missing UPC should skew toward higher anomaly scores.
    This validates that reflect_emb is picking up the evidence-gap signal.
    """
    result = session.run(
        """
        MATCH (n:GlobalSKU)
        WHERE n.self_emb IS NOT NULL AND n.reflect_emb IS NOT NULL
        RETURN n.upc_missing AS upc_missing,
               n.self_emb    AS se,
               n.reflect_emb AS re
        """
    )

    with_upc    = []
    without_upc = []
    for row in result:
        score = anomaly_score(row["se"], row["re"])
        if row["upc_missing"]:
            without_upc.append(score)
        else:
            with_upc.append(score)

    if with_upc and without_upc:
        print(f"\n── UPC missing vs present anomaly scores (GlobalSKU) ────────")
        print(f"  UPC present:  avg={np.mean(with_upc):.4f}  n={len(with_upc)}")
        print(f"  UPC missing:  avg={np.mean(without_upc):.4f}  n={len(without_upc)}")
        delta = np.mean(without_upc) - np.mean(with_upc)
        if delta > 0:
            print(f"  ✓ Missing UPC → +{delta:.4f} avg anomaly score (expected)")
        else:
            print(f"  ⚠ Missing UPC did NOT raise anomaly scores — check neighbour density")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

ALL_LABELS = ["GlobalSKU", "VendorSKU", "Brand", "PackageType",
              "Manufacturer", "Supplier", "ProductClass"]


def main():
    parser = argparse.ArgumentParser(description="Compute reflection vectors and anomaly scores")
    parser.add_argument("--label", default="ALL",
                        help="Node label to process (default: ALL)")
    parser.add_argument("--top",   type=int, default=20,
                        help="Top-N anomalies to print (default: 20)")
    parser.add_argument("--scores-only", action="store_true",
                        help="Skip reflection computation, only print scores")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    labels = ALL_LABELS if args.label == "ALL" else [args.label]

    with driver.session() as session:
        ensure_score_log_indexes(session)

        if not args.scores_only:
            print("\n── Computing reflection vectors ─────────────────────────────")
            for label in labels:
                batch_compute_reflections(session, label)

        print("\n── Anomaly score rankings ───────────────────────────────────")
        for label in labels:
            ranked = rank_anomalies(session, label, top_n=args.top)
            if ranked:
                print_anomaly_table(label, ranked)
                dist = score_distribution(session, label)
                print(f"  Distribution: HIGH={dist['HIGH']} MEDIUM={dist['MEDIUM']} "
                      f"LOW={dist['LOW']} total={dist['total']}")

        # Validation: missing UPC should correlate with higher anomaly
        missing_upc_anomaly_check(session)

        # ── Store anomaly_baseline + append to score log ──────────────────
        print("\n── Writing anomaly_baseline + score log ─────────────────────")
        import uuid as _uuid
        run_id = str(_uuid.uuid4())
        for label in labels:
            n_written = write_baseline_scores(session, label)
            _, n_logged = log_batch_run(session, label, "baseline", run_id=run_id)
            print(f"  {label}: {n_written:,} scores stored | {n_logged:,} log entries  run={run_id[:8]}…")

    driver.close()
    print("\nReflection pipeline complete.\n")


if __name__ == "__main__":
    main()
