"""
api/agent_matcher.py — Agent + Reflexive KG matching pipeline.

Pipeline:
  1. Embed   — encode brand_name + package_type via sentence-transformers
  2. Search  — ANN on Neo4j vector index + brand/package graph lookups
  3. Enrich  — fetch anomaly_attn and reflect_emb from KG per candidate
  4. Score   — composite: ANN sim + graph boost + reflect sim − anomaly penalty
  5. Critic  — validate candidates, produce reasoning (LLM or heuristic)
  6. Route   — merged / updated / insert

Graceful degradation:
  - Neo4j unavailable → falls back to string-matching (api/main.py logic)
  - Bedrock unavailable → uses heuristic reasoning text
  - sentence-transformers unavailable → falls back to string matching
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GLOBAL_SKU_CSV,
    EMBEDDING_MODEL, EMBED_BATCH_SIZE,
    MATCH_AUTO_THRESHOLD, MATCH_REVIEW_THRESHOLD,
    MATCH_ANN_TOP_K,
)
from data.master_loader import load_master_sku_records

# ── thresholds (same as string-matching endpoint) ─────────────────────────────
MERGE_THRESHOLD  = 0.85
UPDATE_THRESHOLD = 0.60

# ── composite score weights ───────────────────────────────────────────────────
W_ANN         = 0.40   # semantic similarity (ANN on self_emb)
W_GRAPH       = 0.30   # brand-block + package-type graph signals
W_REFLECT     = 0.20   # reflect_emb neighborhood-context similarity
W_ANOMALY_PEN = 0.10   # health penalty (high anomaly_attn → lower score)


# ─────────────────────────────────────────────────────────────────────────────
# LAZY SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_driver = None

# brand_name → brand_family lookup (built from master CSV once)
_brand_family_map: dict[str, str] | None = None


def _get_brand_family_map() -> dict[str, str]:
    """Maps brand_name → brand_family from vor_sku_data.csv."""
    global _brand_family_map
    if _brand_family_map is not None:
        return _brand_family_map
    mapping: dict[str, str] = {}
    for rec in load_master_sku_records(ROOT / GLOBAL_SKU_CSV):
        bn = (rec.get("brand_name") or "").strip().upper()
        bf = (rec.get("brand_family") or "").strip().upper()
        if bn and bf and bf not in ("", "UNKNOWN"):
            mapping[bn] = bf
    _brand_family_map = mapping
    return mapping


def _get_model():
    global _model
    if _model is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            device = (
                "mps"  if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available()          else
                "cpu"
            )
            _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
            print(f"[agent_matcher] Loaded embedding model on {device}")
        except Exception as e:
            print(f"[agent_matcher] Could not load embedding model: {e}")
    return _model


def _get_driver():
    global _driver
    if _driver is None:
        try:
            from neo4j import GraphDatabase
            drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            drv.verify_connectivity()
            _driver = drv
            print(f"[agent_matcher] Connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            print(f"[agent_matcher] Neo4j unavailable ({e}) — will fall back to string matching")
    return _driver


def neo4j_available() -> bool:
    return _get_driver() is not None


def embeddings_available() -> bool:
    return _get_model() is not None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EMBED
# ─────────────────────────────────────────────────────────────────────────────

def _embed_query(brand_name: str, package_type: str) -> np.ndarray | None:
    """
    Encode brand + package using the SAME text format as global_sku_to_text()
    in 02_seed_data.py so the query vector aligns with the ANN index:

      "brand {brand_family} package {package_category_name} ..."

    brand_family is looked up from the CSV; falls back to brand_name if not found.
    """
    model = _get_model()
    if model is None:
        return None

    bf_map = _get_brand_family_map()
    brand_family = bf_map.get(brand_name.upper(), brand_name)

    text = f"brand {brand_family} package {package_type}"
    emb = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return emb[0]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MULTI-SIGNAL CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _ann_candidates(session, emb: np.ndarray, top_k: int = MATCH_ANN_TOP_K) -> list[dict]:
    """ANN search on idx_global_sku_self (cosine similarity)."""
    try:
        rows = session.run(
            """
            CALL db.index.vector.queryNodes('idx_global_sku_self', $k, $vec)
            YIELD node AS g, score
            RETURN g.sku_id              AS sku_id,
                   g.brand_name          AS brand_name,
                   g.brand_family        AS brand_family,
                   g.package_category_name AS package_category_name,
                   g.package_name        AS package_name,
                   g.package_type_id     AS package_type_id,
                   g.status              AS status,
                   g.weight              AS weight,
                   g.height              AS height,
                   g.length              AS length,
                   g.width               AS width,
                   g.anomaly_attn        AS anomaly_attn,
                   g.reflect_emb_attn    AS reflect_emb,
                   score                 AS ann_sim
            """,
            k=top_k,
            vec=emb.tolist(),
        ).data()
        for r in rows:
            r["signals"] = ["ann_self"]
            r["ann_sim"]  = float(r.get("ann_sim", 0))
        return rows
    except Exception as e:
        print(f"  [agent] ANN search failed: {e}")
        return []


def _brand_block(session, brand_name: str) -> list[str]:
    """
    GlobalSKU IDs whose brand_name matches the query.

    Searches GlobalSKU.brand_name directly (most reliable) since Brand nodes
    use brand_family which is often 'UNKNOWN' in this dataset.
    """
    try:
        rows = session.run(
            """
            MATCH (g:GlobalSKU)
            WHERE toUpper(g.brand_name) = toUpper($brand)
               OR toUpper(g.brand_name) CONTAINS toUpper($brand)
               OR toUpper($brand) CONTAINS toUpper(g.brand_name)
            RETURN DISTINCT g.sku_id AS sid
            LIMIT 50
            """,
            brand=brand_name,
        ).data()
        return [r["sid"] for r in rows]
    except Exception:
        return []


def _package_block(session, package_type: str) -> dict[str, float]:
    """
    GlobalSKU IDs whose package_category_name matches the query.
    Returns {sku_id: match_quality} where quality is 1.0 (exact) or 0.7 (fuzzy).

    PackageType nodes store only package_type_id (name=None in this dataset),
    so we search GlobalSKU.package_category_name directly.
    """
    try:
        rows = session.run(
            """
            MATCH (g:GlobalSKU)
            WHERE toUpper(g.package_category_name) = toUpper($pkg)
               OR toUpper(g.package_category_name) CONTAINS toUpper($pkg)
               OR toUpper($pkg) CONTAINS toUpper(g.package_category_name)
            RETURN DISTINCT g.sku_id AS sid,
                   CASE WHEN toUpper(g.package_category_name) = toUpper($pkg) THEN 1.0
                        ELSE 0.7 END AS quality
            LIMIT 50
            """,
            pkg=package_type,
        ).data()
        return {r["sid"]: float(r["quality"]) for r in rows}
    except Exception:
        return {}


def _fetch_candidates_by_ids(session, sku_ids: list[str]) -> list[dict]:
    """Fetch full candidate records for a list of SKU IDs."""
    if not sku_ids:
        return []
    try:
        rows = session.run(
            """
            MATCH (g:GlobalSKU) WHERE g.sku_id IN $ids
            RETURN g.sku_id              AS sku_id,
                   g.brand_name          AS brand_name,
                   g.brand_family        AS brand_family,
                   g.package_category_name AS package_category_name,
                   g.package_name        AS package_name,
                   g.package_type_id     AS package_type_id,
                   g.status              AS status,
                   g.weight              AS weight,
                   g.height              AS height,
                   g.length              AS length,
                   g.width               AS width,
                   g.anomaly_attn        AS anomaly_attn,
                   g.reflect_emb_attn    AS reflect_emb
            """,
            ids=sku_ids,
        ).data()
        for r in rows:
            r["ann_sim"] = 0.0
            r["signals"] = []
        return rows
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — REFLEXIVE KG ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray | list, b: np.ndarray | list) -> float:
    """Cosine similarity between two vectors."""
    try:
        av = np.array(a, dtype=float)
        bv = np.array(b, dtype=float)
        na, nb = np.linalg.norm(av), np.linalg.norm(bv)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(av, bv) / (na * nb))
    except Exception:
        return 0.0


def _enrich_with_kg(candidates: list[dict], query_emb: np.ndarray) -> list[dict]:
    """
    For each candidate add:
      reflect_sim  — how similar query_emb is to the candidate's reflect_emb
                     (high = query matches the candidate's KG neighborhood)
      anomaly_attn — already fetched from Neo4j; 0=healthy, 1=anomalous
    """
    for c in candidates:
        reflect_raw = c.get("reflect_emb")
        if reflect_raw is not None:
            c["reflect_sim"] = _cosine(query_emb, reflect_raw)
        else:
            c["reflect_sim"] = 0.0

        # Normalize anomaly_attn to [0, 1]; None → 0 (assume healthy)
        attn = c.get("anomaly_attn")
        c["anomaly_attn"] = float(attn) if attn is not None else 0.0

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — COMPOSITE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _brand_match_score(candidate: dict, brand_name_query: str, brand_ids: set[str]) -> float:
    """
    Brand matching score:
      1.0  exact brand_name match on the GlobalSKU node
      0.75 fuzzy match found via KG brand-block query
      0.0  no brand signal
    """
    cand_brand = (candidate.get("brand_name") or "").upper()
    if cand_brand == brand_name_query.upper():
        return 1.0
    return 0.75 if candidate.get("sku_id") in brand_ids else 0.0


def _pkg_match_score(candidate: dict, pkg_quality: dict[str, float]) -> float:
    """Package match quality: 1.0 exact, 0.7 fuzzy, 0.0 not found."""
    return pkg_quality.get(candidate.get("sku_id", ""), 0.0)


def _composite_score(
    c: dict,
    brand_name_query: str,
    pkg_quality: dict[str, float],
    brand_ids: set[str],
) -> float:
    """
    Composite score weights:
      45% brand match   — primary identity signal
      35% package match — secondary identity signal
      15% ANN sim       — semantic embedding (fallback when brand/pkg unknown)
       5% reflect sim   — KG neighbourhood context
      -10% anomaly      — data-quality health penalty

    Brand + package signals dominate ANN because the ANN index was built with
    brand_family='UNKNOWN' for most SKUs, making it unable to distinguish
    between brands in embedding space.
    """
    brand   = _brand_match_score(c, brand_name_query, brand_ids)
    pkg     = _pkg_match_score(c, pkg_quality)
    ann     = float(c.get("ann_sim", 0.0))
    reflect = float(c.get("reflect_sim", 0.0))
    anomaly = float(c.get("anomaly_attn", 0.0))

    score = (
        0.45 * brand
      + 0.35 * pkg
      + 0.15 * ann
      + 0.05 * reflect
      - 0.10 * anomaly
    )
    return round(max(0.0, min(1.0, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CRITIC (heuristic + optional LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_reasoning(
    brand_name: str, package_type: str, best: dict, score: float,
    status: str, brand_ids: set[str], pkg_quality: dict[str, float],
) -> str:
    sid    = best.get("sku_id", "?")
    bn     = best.get("brand_name") or best.get("brand_family") or "?"
    pkg    = best.get("package_category_name", "?")
    ann    = best.get("ann_sim", 0.0)
    brand_m = _brand_match_score(best, brand_name, brand_ids)
    pkg_m   = _pkg_match_score(best, pkg_quality)
    refl   = best.get("reflect_sim", 0.0)
    anomaly= best.get("anomaly_attn", 0.0)

    signals = []
    if brand_m == 1.0:
        signals.append("brand_exact_match")
    elif brand_m > 0:
        signals.append("brand-block (KG graph edge)")
    if pkg_m > 0:
        signals.append("package-block (KG graph edge)")
    if ann > 0:
        signals.append(f"ANN self_emb similarity={ann:.3f}")
    if refl > 0:
        signals.append(f"reflect_emb neighborhood sim={refl:.3f}")

    health = (
        "healthy (anomaly_attn=0)" if anomaly == 0
        else f"anomaly_attn={anomaly:.3f} — {'high risk' if anomaly >= 0.75 else 'moderate'}"
    )

    action_text = {
        "merged":  "High-confidence match",
        "updated": "Partial match — existing record needs review",
        "insert":  "No sufficient match — new GlobalSKU required",
    }.get(status, status)

    parts = [
        f"{action_text}: GlobalSKU {sid}.",
        f"Brand '{brand_name}' → '{bn}'.",
        f"Package '{package_type}' → '{pkg}'.",
    ]
    if signals:
        parts.append(f"Matching signals: {'; '.join(signals)}.")
    parts.append(f"KG health: {health}.")
    parts.append(
        f"Score breakdown — brand={brand_m:.2f}, pkg={pkg_m:.2f}, "
        f"ANN={ann:.3f}, reflect={refl:.3f}, anomaly_pen={anomaly:.3f} → composite={score:.4f}."
    )
    return " ".join(parts)


def _llm_reasoning(
    brand_name: str, package_type: str, best: dict, score: float,
    status: str, all_candidates: list[dict],
) -> str | None:
    """Call Bedrock LLM to produce a rich reasoning explanation."""
    try:
        from agents.llm import get_llm
        llm = get_llm()
        if not llm.available:
            return None

        top3 = all_candidates[:3]
        cand_text = "\n".join(
            f"  - SKU {c.get('sku_id')}: brand={c.get('brand_name')}, "
            f"package={c.get('package_category_name')}, "
            f"score={c.get('composite_score', 0):.4f}, "
            f"ann={c.get('ann_sim', 0):.3f}, "
            f"reflect={c.get('reflect_sim', 0):.3f}, "
            f"anomaly_attn={c.get('anomaly_attn', 0):.3f}"
            for c in top3
        )

        prompt = f"""You are a SKU data-quality agent for a beverage distribution company.

A new product entry needs to be matched against the Master Global SKU database.

INPUT:
  brand_name   : {brand_name}
  package_type : {package_type}

TOP CANDIDATES FROM REFLEXIVE KNOWLEDGE GRAPH:
{cand_text}

DECISION: status={status}, confidence={score:.4f}

Explain in 2-3 sentences WHY this match decision was made. Mention:
- How well the brand and package aligned
- What the KG graph signals (ANN similarity, reflect neighborhood, anomaly health) indicate
- Whether the decision is confident or needs human review

Be concise and specific. Do not repeat the numbers verbatim."""

        return llm.complete(prompt, max_tokens=256, temperature=0.0)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT MATCH FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def agent_match(
    brand_name: str,
    package_type: str,
    query_dims: dict | None = None,
) -> dict:
    """
    Full agent + Reflexive KG matching pipeline.

    Returns the same schema as the string-matching endpoint so the API
    response is interchangeable, but adds:
      - ann_sim, reflect_sim, anomaly_attn per matched SKU
      - score_breakdown showing contribution of each signal
      - kg_available flag so caller knows which path was taken
    """
    query_dims = query_dims or {}

    # ── Step 1: Embed ─────────────────────────────────────────────────────────
    query_emb = _embed_query(brand_name, package_type)

    # ── Step 2: Candidates from Neo4j ────────────────────────────────────────
    driver = _get_driver()

    if driver is None or query_emb is None:
        # Graceful degradation: fall back to string matching
        print("[agent_matcher] Falling back to string-matching (Neo4j/embeddings unavailable)")
        from api.main import match_sku
        result = match_sku(brand_name, package_type, query_dims)
        result["kg_available"] = False
        result["fallback"] = "string_matching"
        return result

    with driver.session() as session:
        # ANN candidates
        ann_results = _ann_candidates(session, query_emb, top_k=MATCH_ANN_TOP_K)

        # Graph signal: brand block + package block
        brand_ids   = set(_brand_block(session, brand_name))
        pkg_quality = _package_block(session, package_type)  # {sku_id: quality}
        pkg_ids     = set(pkg_quality.keys())

        # Fetch records for graph-signal candidates not already in ANN results.
        # Priority: brand+package (double-signal) first, then single-signal.
        ann_ids = {r["sku_id"] for r in ann_results}
        double_signal = (brand_ids & pkg_ids) - ann_ids
        single_signal = (brand_ids | pkg_ids) - ann_ids - double_signal
        extra_ids = list(double_signal) + list(single_signal)   # no arbitrary cap
        extra_results = _fetch_candidates_by_ids(session, extra_ids)
        for r in extra_results:
            r["signals"] = []
        all_raw = ann_results + extra_results

    if not all_raw:
        return _insert_result(brand_name, package_type, "No candidates found in knowledge graph", True)

    # ── Step 3: Reflexive KG enrichment ──────────────────────────────────────
    candidates = _enrich_with_kg(all_raw, query_emb)

    # ── Step 4: Composite scoring + dedup ────────────────────────────────────
    seen: dict[str, dict] = {}
    for c in candidates:
        sid = c.get("sku_id") or ""
        if not sid:
            continue
        score = _composite_score(c, brand_name, pkg_quality, brand_ids)
        c["composite_score"] = score
        # Annotate signals for traceability
        cand_brand = (c.get("brand_name") or "").upper()
        if cand_brand == brand_name.upper() and "brand_exact" not in c.get("signals", []):
            c.setdefault("signals", []).append("brand_exact")
        elif sid in brand_ids and "brand_block" not in c.get("signals", []):
            c.setdefault("signals", []).append("brand_block")
        q = pkg_quality.get(sid, 0)
        if q == 1.0 and "package_exact" not in c.get("signals", []):
            c.setdefault("signals", []).append("package_exact")
        elif q > 0 and "package_block" not in c.get("signals", []):
            c.setdefault("signals", []).append("package_block")
        if sid not in seen or score > seen[sid]["composite_score"]:
            seen[sid] = c

    ranked = sorted(seen.values(), key=lambda x: x["composite_score"], reverse=True)

    if not ranked:
        return _insert_result(brand_name, package_type, "Scoring produced no valid candidates", True)

    best  = ranked[0]
    score = best["composite_score"]

    # ── Step 5: Critic — reasoning ────────────────────────────────────────────
    if score >= MERGE_THRESHOLD:
        status = "merged"
    elif score >= UPDATE_THRESHOLD:
        status = "updated"
    else:
        status = "insert"

    reasoning = (
        _llm_reasoning(brand_name, package_type, best, score, status, ranked)
        or _heuristic_reasoning(brand_name, package_type, best, score, status, brand_ids, pkg_quality)
    )

    # ── Step 6: Build response ────────────────────────────────────────────────
    matched_skus = [
        {
            "sku_id":               c.get("sku_id", ""),
            "brand_name":           c.get("brand_name") or c.get("brand_family") or "",
            "package_category_name":c.get("package_category_name", ""),
            "package_name":         c.get("package_name", ""),
            "package_type_id":      str(c.get("package_type_id", "")),
            "status":               c.get("status", ""),
            "weight":               _safe_float(c.get("weight")),
            "height":               _safe_float(c.get("height")),
            "length":               _safe_float(c.get("length")),
            "width":                _safe_float(c.get("width")),
            "confidence":           c["composite_score"],
            "score_breakdown": {
                "brand_match":  round(_brand_match_score(c, brand_name, brand_ids), 4),
                "pkg_match":    round(_pkg_match_score(c, pkg_quality), 4),
                "ann_sim":      round(float(c.get("ann_sim", 0)), 4),
                "reflect_sim":  round(float(c.get("reflect_sim", 0)), 4),
                "anomaly_attn": round(float(c.get("anomaly_attn", 0)), 4),
            },
            "signals": c.get("signals", []),
        }
        for c in ranked[:5]
    ]

    return {
        "status":       status,
        "confidence":   score,
        "reasoning":    reasoning,
        "ambiguous":    (
            len(ranked) >= 2
            and abs(ranked[0]["composite_score"] - ranked[1]["composite_score"]) < 0.05
        ),
        "matched_skus": matched_skus,
        "query": {
            "brand_name":   brand_name,
            "package_type": package_type,
        },
        "kg_available": True,
        "pipeline": {
            "ann_candidates":   len(ann_results),
            "brand_block_hits": len(brand_ids),
            "package_block_hits": len(pkg_ids),
            "total_candidates": len(ranked),
        },
    }


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _insert_result(brand: str, package: str, reason: str, kg_available: bool) -> dict:
    return {
        "status":       "insert",
        "confidence":   0.0,
        "reasoning":    (
            f"No sufficient match for brand='{brand}', package_type='{package}'. "
            f"{reason}. A new GlobalSKU should be created."
        ),
        "ambiguous":    False,
        "matched_skus": [],
        "query":        {"brand_name": brand, "package_type": package},
        "kg_available": kg_available,
        "pipeline":     {},
    }
