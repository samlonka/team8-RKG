"""Neo4j + evaluation data access for the Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import dataclass

import streamlit as st
from neo4j import GraphDatabase

from agents.lifecycle_doer import COHORT, LifecycleDoer
from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    ANOMALY_HIGH_RISK,
    ANOMALY_MEDIUM_RISK,
    REL_WEIGHTS,
)
from scoring import load_cohort_scores, training_gate_status

MANIFEST_PATH = "seed_manifest.json"


def risk_band(score: float) -> str:
    if score >= ANOMALY_HIGH_RISK:
        return "high"
    if score >= ANOMALY_MEDIUM_RISK:
        return "medium"
    return "low"


@dataclass
class SkuRow:
    sku_id: str
    brand_family: str
    package: str
    anomaly_score: float
    base_score: float
    planted_type: str | None
    band: str


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def check_connection() -> tuple[bool, str]:
    try:
        driver = get_driver()
        with driver.session() as s:
            s.run("RETURN 1").single()
        driver.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=120)
def _cached_cohort_score_map() -> dict[str, dict]:
    _, score_map = load_cohort_scores()
    return score_map


@st.cache_data(ttl=120)
def load_cohort_rankings(limit: int = 50) -> list[SkuRow]:
    score_map = _cached_cohort_score_map()
    if not score_map:
        return []

    driver = get_driver()
    meta: dict[str, dict] = {}
    with driver.session() as s:
        for r in s.run(
            """
            MATCH (g:GlobalSKU {cohort: $c})
            WHERE g.self_emb IS NOT NULL AND g.reflect_emb IS NOT NULL
            RETURN g.sku_id AS sku,
                   coalesce(g.brand_family, '') AS brand,
                   coalesce(g.package_category_name, '') AS pkg
            """,
            c=COHORT,
        ).data():
            meta[str(r["sku"])] = r
    driver.close()

    rows: list[SkuRow] = []
    for sku, d in score_map.items():
        m = meta.get(sku, {})
        eff = d["effective"]
        rows.append(
            SkuRow(
                sku_id=sku,
                brand_family=m.get("brand") or "—",
                package=m.get("pkg") or "—",
                anomaly_score=eff,
                base_score=d["base"],
                planted_type=d.get("planted_type"),
                band=risk_band(eff),
            )
        )
    rows.sort(key=lambda x: x.anomaly_score, reverse=True)
    return rows[:limit]


@st.cache_data(ttl=120)
def load_score_histogram() -> list[float]:
    return [r.anomaly_score for r in load_cohort_rankings(300)]


@st.cache_data(ttl=120)
def load_training_gate() -> dict:
    _, score_map = load_cohort_scores()
    if not score_map:
        return {"ok": False, "error": "No cohort scores"}
    return training_gate_status(score_map)


def load_shared_skus() -> list[dict]:
    driver = get_driver()
    with driver.session() as s:
        rows = s.run(
            """
            MATCH (g:GlobalSKU {cohort: $c})-[:USED_BY]->(cust:Customer)
            WITH g, collect(DISTINCT cust.customer_id) AS customers
            WHERE size(customers) > 1
            RETURN g.sku_id AS sku,
                   g.brand_family AS brand,
                   customers,
                   coalesce(g.planted_type, '') AS planted
            ORDER BY size(customers) DESC
            """,
            c=COHORT,
        ).data()
    driver.close()
    return [dict(r) for r in rows]


def load_blast_radius(sku_id: str) -> dict:
    driver = get_driver()
    with driver.session() as s:
        doer = LifecycleDoer(s)
        result = doer.shared_sku_blast_radius(sku_id)
    driver.close()
    return result


def load_auto_map_skus() -> list[dict]:
    driver = get_driver()
    with driver.session() as s:
        rows = s.run(
            """
            MATCH (g:GlobalSKU {cohort: $c, planted_type: 'auto_map_error'})
            OPTIONAL MATCH (t:TenantSKU {match_method: 'fuzzy'})-[:MAPS_TO]->(g)
            OPTIONAL MATCH (p:Pallet)-[:SCANNED_ON]->(g) WHERE p.outcome = 'failure'
            RETURN g.sku_id AS sku,
                   g.brand_family AS brand,
                   count(DISTINCT t) AS fuzzy_tenants,
                   count(DISTINCT p) AS scan_failures
            ORDER BY scan_failures DESC
            """,
            c=COHORT,
        ).data()
    driver.close()
    return [dict(r) for r in rows]


def load_manifest() -> dict | None:
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def run_closed_world() -> list[dict]:
    driver = get_driver()
    with driver.session() as s:
        rows = s.run(
            "MATCH (s:GlobalSKU)-[:BELONGS_TO_BRAND]->(b:Brand) "
            "WHERE b.flag = 'duplicate' "
            "RETURN s.sku_id AS sku_id, b.brand_family AS brand LIMIT 20"
        ).data()
    driver.close()
    return [dict(r) for r in rows]


def cohort_stats() -> dict:
    rankings = load_cohort_rankings(300)
    scores = [r.anomaly_score for r in rankings]
    n = len(scores)
    decile_k = max(1, n // 10) if n else 0
    threshold = rankings[decile_k - 1].anomaly_score if decile_k else 0.0
    gate = load_training_gate()
    return {
        "cohort": COHORT,
        "n_scored": n,
        "median": float(__import__("numpy").median(scores)) if scores else 0.0,
        "threshold": threshold,
        "high_risk": sum(1 for s in scores if s >= ANOMALY_HIGH_RISK),
        "shared_count": len(load_shared_skus()),
        "auto_map_count": len(load_auto_map_skus()),
        "training_gate_ok": gate.get("ok", False),
        "training_gate": gate,
    }


def rel_weights_for_ui() -> dict[str, float]:
    return {k: v for k, v in REL_WEIGHTS.items() if k != "_DEFAULT"}
