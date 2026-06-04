"""
06_scale_evaluate.py — Stratified catalog-scale anomaly evaluation.

Samples 1k–10k GlobalSKUs from the full catalog (preferring SKUs outside the
demo cohort), stratified by upc_missing, and reports score distribution stats.

Usage:
    python 06_scale_evaluate.py [--sample 5000]
"""

from __future__ import annotations

import argparse
import random

import numpy as np
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from reflection_core import anomaly_score

RNG = random.Random(42)
COHORT_TAG = "ACME_ONBOARDING"


def stratified_sample_ids(session, n: int, cohort_tag: str) -> list[str]:
    """Sample up to n SKUs with both embeddings, stratified by upc_missing."""
    rows = session.run(
        """
        MATCH (g:GlobalSKU)
        WHERE g.self_emb IS NOT NULL AND g.reflect_emb IS NOT NULL
              AND (g.cohort IS NULL OR g.cohort <> $tag)
        RETURN g.sku_id AS sku, coalesce(g.upc_missing, false) AS upc_missing
        """,
        tag=cohort_tag,
    ).data()
    if len(rows) < n:
        extra = session.run(
            """
            MATCH (g:GlobalSKU {cohort: $tag})
            WHERE g.self_emb IS NOT NULL AND g.reflect_emb IS NOT NULL
            RETURN g.sku_id AS sku, coalesce(g.upc_missing, false) AS upc_missing
            """,
            tag=cohort_tag,
        ).data()
        seen = {r["sku"] for r in rows}
        for r in extra:
            if r["sku"] not in seen:
                rows.append(r)
                seen.add(r["sku"])

    by_stratum: dict[bool, list[str]] = {True: [], False: []}
    for r in rows:
        by_stratum[bool(r["upc_missing"])].append(r["sku"])

    n_true = len(by_stratum[True])
    n_false = len(by_stratum[False])
    if n_true + n_false == 0:
        return []

    frac_missing = n_true / (n_true + n_false)
    k_missing = max(1, int(round(n * frac_missing))) if n_true else 0
    k_present = n - k_missing
    k_missing = min(k_missing, n_true)
    k_present = min(k_present, n_false)

    RNG.shuffle(by_stratum[True])
    RNG.shuffle(by_stratum[False])
    picked = by_stratum[True][:k_missing] + by_stratum[False][:k_present]
    if len(picked) < n:
        rest = [s for s in by_stratum[True][k_missing:] + by_stratum[False][k_present:] if s not in picked]
        RNG.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def score_sample(session, sku_ids: list[str]) -> list[dict]:
    rows = session.run(
        """
        UNWIND $ids AS sid
        MATCH (g:GlobalSKU {sku_id: sid})
        WHERE g.self_emb IS NOT NULL AND g.reflect_emb IS NOT NULL
        RETURN g.sku_id AS sku,
               g.self_emb AS se,
               g.reflect_emb AS re,
               coalesce(g.upc_missing, false) AS upc_missing,
               coalesce(g.cohort, '') AS cohort
        """,
        ids=sku_ids,
    ).data()
    out = []
    for r in rows:
        base = anomaly_score(r["se"], r["re"])
        out.append(
            {
                "sku": r["sku"],
                "score": base,
                "upc_missing": bool(r["upc_missing"]),
                "in_cohort": r["cohort"] == COHORT_TAG,
            }
        )
    return out


def evaluate_scale(sample_size: int = 5000) -> dict:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        ids = stratified_sample_ids(s, sample_size, COHORT_TAG)
        scored = score_sample(s, ids)
    driver.close()

    if not scored:
        return {"n": 0, "error": "no scored SKUs"}

    scores = [x["score"] for x in scored]
    missing = [x for x in scored if x["upc_missing"]]
    present = [x for x in scored if not x["upc_missing"]]
    high = sum(1 for s in scores if s >= 0.7)

    return {
        "n": len(scored),
        "requested": sample_size,
        "mean": float(np.mean(scores)),
        "p90": float(np.percentile(scores, 90)),
        "p99": float(np.percentile(scores, 99)),
        "high_risk_frac": high / len(scored),
        "upc_missing_mean": float(np.mean([x["score"] for x in missing])) if missing else None,
        "upc_present_mean": float(np.mean([x["score"] for x in present])) if present else None,
        "n_upc_missing": len(missing),
        "n_upc_present": len(present),
        "cohort_in_sample": sum(1 for x in scored if x["in_cohort"]),
    }


def report(sample_size: int = 5000):
    m = evaluate_scale(sample_size)
    print("\n── Catalog-Scale Evaluation (stratified) ───────────────────")
    if m.get("error"):
        print(f"  {m['error']}")
        return m
    print(f"  sampled: {m['n']} SKUs (requested {m['requested']})")
    print(f"  UPC strata: missing={m['n_upc_missing']}  present={m['n_upc_present']}")
    print(f"  mean score={m['mean']:.4f}  p90={m['p90']:.4f}  p99={m['p99']:.4f}")
    print(f"  high-risk (>=0.7): {m['high_risk_frac']:.2%}")
    if m["upc_missing_mean"] is not None and m["upc_present_mean"] is not None:
        delta = m["upc_missing_mean"] - m["upc_present_mean"]
        print(
            f"  UPC missing avg={m['upc_missing_mean']:.4f}  "
            f"present avg={m['upc_present_mean']:.4f}  delta={delta:+.4f}"
        )
    print(f"  demo cohort SKUs in sample: {m['cohort_in_sample']}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5000, help="Sample size (1k–10k)")
    args = ap.parse_args()
    report(min(max(args.sample, 1000), 10000))
