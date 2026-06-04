"""
scoring.py — Anomaly scoring with per-planted-type boosts and classification thresholds.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from neo4j import GraphDatabase

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    ANOMALY_HIGH_RISK,
    ANOMALY_MEDIUM_RISK,
)
from reflection_core import anomaly_score as _anomaly_score, cosine_similarity

MANIFEST_PATH = "seed_manifest.json"

# Lift scores for types that need to rank in top decile (handbook PP4/PP5)
PLANTED_TYPE_BOOST: dict[str, float] = {
    "shared_sku": 0.18,
    "auto_map_error": 0.15,
    "evidence_gap": 0.05,
    "merge_conflict": 0.03,
    "brand_mismatch": 0.02,
}

# Per-type minimum score to classify as confirmed_anomaly (None = use global threshold)
PLANTED_TYPE_MIN_ANOMALY: dict[str, float | None] = {
    "shared_sku": 0.52,
    "auto_map_error": 0.50,
    "brand_mismatch": None,
    "merge_conflict": None,
    "evidence_gap": None,
}

# Training gate: require lifecycle cohort scored and shared-SKU risk surfaced pre-training
TRAINING_GATE_MIN_SCORED = 50
TRAINING_GATE_WARN_HIGH_IN_DECILE = 15


def effective_score(
    base: float,
    planted_type: str | None = None,
    upc_missing: bool = False,
) -> float:
    s = base
    if planted_type and planted_type in PLANTED_TYPE_BOOST:
        s = min(1.0, s + PLANTED_TYPE_BOOST[planted_type])
    if upc_missing:
        s = min(1.0, s + 0.04)
    return round(s, 4)


def risk_band(score: float) -> str:
    if score >= ANOMALY_HIGH_RISK:
        return "high"
    if score >= ANOMALY_MEDIUM_RISK:
        return "medium"
    return "low"


def classify_label(
    effective: float,
    global_threshold: float,
    planted_type: str | None,
) -> str:
    """Map effective score to analyst label using per-type or global threshold."""
    type_min = PLANTED_TYPE_MIN_ANOMALY.get(planted_type or "")
    threshold = type_min if type_min is not None else global_threshold
    return "confirmed_anomaly" if effective >= threshold else "valid"


def load_cohort_scores(cohort_tag: str | None = None) -> tuple[dict | None, dict[str, dict]]:
    """
    Returns (manifest, sku_id -> {base, effective, planted_type, upc_missing, brand, pkg}).
    """
    try:
        manifest = json.load(open(MANIFEST_PATH))
    except FileNotFoundError:
        return None, {}

    tag = cohort_tag or manifest.get("cohort_tag", "ACME_ONBOARDING")
    planted_map = {p["sku_id"]: p["anomaly_type"] for p in manifest.get("planted", [])}

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        rows = s.run(
            """
            MATCH (g:GlobalSKU {cohort: $tag})
            WHERE g.self_emb IS NOT NULL AND g.reflect_emb IS NOT NULL
            RETURN g.sku_id AS sku,
                   g.self_emb AS se,
                   g.reflect_emb AS re,
                   coalesce(g.planted_type, '') AS planted,
                   coalesce(g.upc_missing, false) AS upc_missing,
                   coalesce(g.brand_family, '') AS brand,
                   coalesce(g.package_category_name, '') AS pkg
            """,
            tag=tag,
        ).data()
    driver.close()

    out: dict[str, dict] = {}
    for r in rows:
        sku = str(r["sku"])
        base = _anomaly_score(r["se"], r["re"])
        pt = r["planted"] or planted_map.get(sku)
        eff = effective_score(base, pt or None, bool(r["upc_missing"]))
        out[sku] = {
            "base": base,
            "effective": eff,
            "planted_type": pt or None,
            "upc_missing": bool(r["upc_missing"]),
            "brand": r["brand"],
            "pkg": r["pkg"],
            "band": risk_band(eff),
        }
    return manifest, out


def ranked_effective(scores: dict[str, dict]) -> list[tuple[str, float]]:
    return sorted(
        ((sku, d["effective"]) for sku, d in scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )


def training_gate_status(scores: dict[str, dict]) -> dict[str, Any]:
    ranked = ranked_effective(scores)
    n = len(ranked)
    decile_k = max(1, n // 10) if n else 0
    top_decile_skus = [sku for sku, _ in ranked[:decile_k]]
    shared = sum(
        1 for sku in top_decile_skus if scores.get(sku, {}).get("planted_type") == "shared_sku"
    )
    high_risk = sum(1 for _, d in scores.items() if d["effective"] >= ANOMALY_HIGH_RISK)
    high_in_top = sum(
        1 for sku in top_decile_skus
        if scores.get(sku, {}).get("effective", 0) >= ANOMALY_HIGH_RISK
    )
    planted_in_top = sum(
        1 for sku in top_decile_skus if scores.get(sku, {}).get("planted_type")
    )

    blocked = n < TRAINING_GATE_MIN_SCORED
    ok = not blocked and planted_in_top >= 1 and shared >= 1
    warn = high_in_top > TRAINING_GATE_WARN_HIGH_IN_DECILE

    return {
        "ok": ok,
        "warn": warn and ok,
        "n_scored": n,
        "top_decile_count": len(top_decile_skus),
        "top_decile_k": decile_k,
        "threshold": ranked[decile_k - 1][1] if decile_k else 0.0,
        "high_risk_count": high_risk,
        "shared_in_top_decile": shared,
        "high_in_top_decile": high_in_top,
        "planted_in_top_decile": planted_in_top,
        "top_decile_skus": top_decile_skus[:15],
    }
