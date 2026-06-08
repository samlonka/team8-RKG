"""
api/agent_matcher.py — Agent + Reflexive KG matching pipeline.

Pipeline:
  1. Embed   — encode brand_name + package_type via sentence-transformers
  2. Search  — ANN on Neo4j vector index + brand/package graph lookups
  3. Enrich  — fetch anomaly_attn and reflect_emb from KG per candidate
  4. Score   — composite: ANN sim + graph boost + reflect sim − anomaly penalty
  5. Critic  — validate candidates, produce reasoning (Bedrock Opus 4.7, required)
  6. Route   — merged / updated / insert

Graceful degradation:
  - Neo4j unavailable → falls back to string-matching (api/main.py logic)
  - Bedrock unavailable → raises LLMError (no heuristic reasoning fallback)
  - sentence-transformers unavailable → falls back to string matching
"""

from __future__ import annotations

import difflib
import importlib
import math
import re
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
    EMBEDDING_MODEL, EMBED_BATCH_SIZE,
    MATCH_AUTO_THRESHOLD, MATCH_REVIEW_THRESHOLD,
    MATCH_ANN_TOP_K,
)
from data.master_loader import load_master_sku_records
from agents.dim_match import (
    apply_dimension_ranking,
    apply_dim_disambiguation,
    format_dim_comparison,
    format_dims,
    has_query_dims,
    normalize_query_dims,
    dim_boost,
    has_comparable_dims,
)
from agents.product_risk import format_product_risk_for_prompt, product_risk_for_sku

# ── thresholds (same as string-matching endpoint) ─────────────────────────────
MERGE_THRESHOLD  = 0.85
UPDATE_THRESHOLD = 0.60
LLM_TRUST_MIN    = 0.30   # below this composite score the LLM indicator is ignored

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
    """Maps brand_name → brand_family from master catalog (Postgres or CSV)."""
    global _brand_family_map
    if _brand_family_map is not None:
        return _brand_family_map
    mapping: dict[str, str] = {}
    for rec in load_master_sku_records():
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


_BRAND_FUZZY_THRESHOLD   = 0.75
_PACKAGE_FUZZY_THRESHOLD = 0.75


def _norm_brand(s: str) -> str:
    """Normalize a brand string: underscores/hyphens → spaces, uppercase, collapse spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", s)).strip().upper()


def _brand_block(session, brand_name: str) -> list[str]:
    """
    GlobalSKU IDs whose brand_name matches the query.

    Normalizes underscores and hyphens to spaces before comparison so that
    "BANG_BDAY_CAKE_BS" matches a graph value of "BANG BDAY CAKE BS".

    Step 1 — exact case-insensitive match in Neo4j (normalized form).
    Step 2 — if no exact hits, fetch all distinct brand_names and apply
             Python difflib fuzzy matching (ratio ≥ _BRAND_FUZZY_THRESHOLD)
             with both sides normalized.
    """
    brand_q = _norm_brand(brand_name)

    try:
        # Step 1: exact match on normalized brand name
        rows = session.run(
            """
            MATCH (g:GlobalSKU)
            WHERE toUpper(g.brand_name) = toUpper($brand)
            RETURN DISTINCT g.sku_id AS sid
            LIMIT 50
            """,
            brand=brand_q,
        ).data()
        if rows:
            return [r["sid"] for r in rows]

        # Step 2: fuzzy fallback — normalize both sides before comparing.
        all_brands = session.run(
            "MATCH (g:GlobalSKU) WHERE g.brand_name IS NOT NULL "
            "RETURN DISTINCT g.brand_name AS bn"
        ).data()

        matched = [
            r["bn"] for r in all_brands
            if difflib.SequenceMatcher(
                None, brand_q, _norm_brand(r["bn"] or "")
            ).ratio() >= _BRAND_FUZZY_THRESHOLD
        ]
        if not matched:
            return []

        rows = session.run(
            "MATCH (g:GlobalSKU) WHERE g.brand_name IN $brands "
            "RETURN DISTINCT g.sku_id AS sid LIMIT 50",
            brands=matched,
        ).data()
        return [r["sid"] for r in rows]

    except Exception:
        return []


# ── Package-type word-to-number canonicalization ──────────────────────────────
# Converts English number words to digits so that "Sixteen OZ CN One/12" and
# "16OZ CN 1/12" reduce to the same canonical string "16OZCN1/12".
# Applied to BOTH query and graph values before any comparison (bidirectional).

# Space-separated multi-word phrases — must be checked before token-level split.
# Longer phrases listed first to avoid partial substitution ("twenty four" before "four").
_PKG_PHRASE_MAP: list[tuple[str, str]] = [
    ("twenty one", "21"),    ("twenty two", "22"),    ("twenty three", "23"),
    ("twenty four", "24"),   ("twenty five", "25"),   ("twenty six", "26"),
    ("twenty seven", "27"),  ("twenty eight", "28"),  ("twenty nine", "29"),
    ("thirty one", "31"),    ("thirty two", "32"),    ("thirty three", "33"),
    ("thirty four", "34"),   ("thirty five", "35"),   ("thirty six", "36"),
    ("sixty four", "64"),    ("one hundred", "100"),
]

# Single-token words (plain and hyphenated forms).
_PKG_WORD_TO_NUM: dict[str, str] = {
    "zero": "0",     "one": "1",       "two": "2",       "three": "3",
    "four": "4",     "five": "5",      "six": "6",       "seven": "7",
    "eight": "8",    "nine": "9",      "ten": "10",      "eleven": "11",
    "twelve": "12",  "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17","eighteen": "18", "nineteen": "19",
    "twenty": "20",  "thirty": "30",   "forty": "40",    "fifty": "50",
    "sixty": "60",   "seventy": "70",  "eighty": "80",   "ninety": "90",
    "hundred": "100",
    # Hyphenated compound forms
    "twenty-one": "21",   "twenty-two": "22",   "twenty-three": "23",
    "twenty-four": "24",  "twenty-five": "25",  "twenty-six": "26",
    "twenty-seven": "27", "twenty-eight": "28", "twenty-nine": "29",
    "thirty-one": "31",   "thirty-two": "32",   "thirty-three": "33",
    "thirty-four": "34",  "thirty-five": "35",  "thirty-six": "36",
    "sixty-four": "64",   "one-hundred": "100",
}

# Container-type abbreviation synonyms (lowercase keys → canonical form).
# Applied after word-to-number conversion so "can" → "cn" not confused with numbers.
_CONTAINER_SYNONYMS: dict[str, str] = {
    "can":     "cn",
    "cans":    "cn",
    "bottle":  "bt",
    "bottles": "bt",
    "btl":     "bt",
    "plastic": "pl",
    "pet":     "pl",
    "pack":    "pk",
    "pkg":     "pk",
}


def _canonicalize_package(text: str) -> str:
    """
    Reduce a package type string to a canonical form for bidirectional comparison.

    Steps:
      1. Lowercase and phrase pre-scan (space-separated multi-word numbers).
      2. Token-level word → digit conversion; handles "/" compound tokens.
      3. Uppercase and strip all whitespace.

    Examples:
      "Sixteen OZ CN One/12"   → "16OZCN1/12"
      "16OZ CN 1/12"           → "16OZCN1/12"
      "16 OZ CN 1/12"          → "16OZCN1/12"
      "Twenty Four OZ PL 1/6"  → "24OZPL1/6"
      "Thirty Six OZ CN 1/12"  → "36OZCN1/12"
    """
    s = text.lower().strip()

    # Phase 1: replace space-separated multi-word number phrases
    for phrase, digit in _PKG_PHRASE_MAP:
        s = s.replace(phrase, digit)

    # Phase 2: token-level conversion; split "/" compound tokens independently.
    # After number-word conversion, apply container-type synonyms (e.g. "can" → "cn").
    tokens = s.split()
    normalized = []
    for token in tokens:
        if "/" in token:
            parts = [_PKG_WORD_TO_NUM.get(p, p) for p in token.split("/")]
            normalized.append("/".join(parts))
        else:
            t = _PKG_WORD_TO_NUM.get(token, token)
            t = _CONTAINER_SYNONYMS.get(t, t)
            normalized.append(t)

    # Phase 3: uppercase and strip all whitespace
    return "".join(normalized).upper()


def _canonical_sorted(text: str) -> str:
    """
    Like _canonicalize_package but with tokens sorted alphabetically before joining.
    Used as a secondary comparison tier (quality 0.90) that is order-independent:
    "1/12 16OZ CAN" and "16OZ CN 1/12" both reduce to the same sorted string.
    """
    s = text.lower().strip()
    for phrase, digit in _PKG_PHRASE_MAP:
        s = s.replace(phrase, digit)
    tokens = s.split()
    normalized = []
    for token in tokens:
        if "/" in token:
            parts = [_PKG_WORD_TO_NUM.get(p, p) for p in token.split("/")]
            normalized.append("/".join(parts))
        else:
            t = _PKG_WORD_TO_NUM.get(token, token)
            t = _CONTAINER_SYNONYMS.get(t, t)
            normalized.append(t)
    return "".join(sorted(t.upper() for t in normalized))


def _package_block(session, package_type: str) -> dict[str, float]:
    """
    GlobalSKU IDs whose package_category_name matches the query.
    Returns {sku_id: match_quality} where quality is 1.0 (exact) or 0.7 (fuzzy).

    Matching is bidirectional: both query and graph values are reduced to a
    canonical form (digits, uppercase, no whitespace) before any comparison,
    so "Sixteen OZ CN One/12" and "16OZ CN 1/12" resolve to the same value.

    Step 1 — fast Cypher exact match: canonical query vs whitespace-stripped
             graph value. Hits immediately when graph stores numeric form.
    Step 2 — Python canonicalize both sides: handles graph values that store
             English number words, plus fuzzy fallback for near-matches.
    Step 3 — fetch SKU IDs for all matched package names.
    """
    try:
        query_canon = _canonicalize_package(package_type)

        # Step 1: exact match — canonical query vs whitespace-stripped graph value.
        rows = session.run(
            """
            MATCH (g:GlobalSKU)
            WHERE replace(toUpper(g.package_category_name), ' ', '') = $pkg_canon
            RETURN DISTINCT g.sku_id AS sid, 1.0 AS quality
            LIMIT 50
            """,
            pkg_canon=query_canon,
        ).data()
        if rows:
            return {r["sid"]: float(r["quality"]) for r in rows}

        # Step 2: fetch all distinct package names and canonicalize both sides.
        # Three comparison tiers (quality 1.0 → 0.90 → 0.70):
        #   1.0  exact canonical match
        #   0.90 same tokens, different order  ("1/12 16OZ CAN" == "16OZ CN 1/12")
        #   0.70 fuzzy match (difflib ratio ≥ threshold)
        all_pkgs = session.run(
            "MATCH (g:GlobalSKU) WHERE g.package_category_name IS NOT NULL "
            "RETURN DISTINCT g.package_category_name AS pkg"
        ).data()

        query_canon_sorted = _canonical_sorted(package_type)

        pkg_quality: dict[str, float] = {}
        for r in all_pkgs:
            graph_pkg   = r["pkg"] or ""
            graph_canon = _canonicalize_package(graph_pkg)
            if not graph_canon:
                continue
            if query_canon == graph_canon:
                pkg_quality[graph_pkg] = 1.0
            elif _canonical_sorted(graph_pkg) == query_canon_sorted:
                pkg_quality[graph_pkg] = 0.9
            elif difflib.SequenceMatcher(
                None, query_canon, graph_canon
            ).ratio() >= _PACKAGE_FUZZY_THRESHOLD:
                pkg_quality[graph_pkg] = 0.7

        if not pkg_quality:
            return {}

        # Step 3: fetch SKU IDs for all matched package names.
        rows = session.run(
            """
            MATCH (g:GlobalSKU) WHERE g.package_category_name IN $pkgs
            RETURN DISTINCT g.sku_id AS sid, g.package_category_name AS pkg
            LIMIT 50
            """,
            pkgs=list(pkg_quality.keys()),
        ).data()
        return {r["sid"]: pkg_quality.get(r["pkg"], 0.7) for r in rows}

    except Exception:
        return {}


def _fetch_candidates_by_ids(session, sku_ids: list[str]) -> list[dict]:
    """Fetch full candidate records for a list of SKU IDs.
    self_emb is included so _enrich_with_kg can compute ann_sim for these
    candidates (which were found via brand/package graph signals, not ANN).
    """
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
                   g.reflect_emb_attn    AS reflect_emb,
                   g.self_emb            AS self_emb
            """,
            ids=sku_ids,
        ).data()
        for r in rows:
            r["ann_sim"] = 0.0   # placeholder; overwritten by _enrich_with_kg
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
    For each candidate compute / finalize:
      ann_sim      — cosine(query_emb, self_emb) for brand/package-block candidates
                     that were not returned by ANN (their ann_sim is 0.0 placeholder).
                     ANN candidates already carry the correct score from Neo4j.
      reflect_sim  — cosine(query_emb, reflect_emb_attn)
      anomaly_attn — normalise to [0, 1]; None → 0 (assume healthy)
    """
    for c in candidates:
        # Point 1: compute ann_sim for non-ANN candidates using their self_emb
        if c.get("ann_sim", 0.0) == 0.0:
            self_raw = c.get("self_emb")
            if self_raw is not None:
                c["ann_sim"] = _cosine(query_emb, self_raw)

        reflect_raw = c.get("reflect_emb")
        c["reflect_sim"] = _cosine(query_emb, reflect_raw) if reflect_raw is not None else 0.0

        attn = c.get("anomaly_attn")
        c["anomaly_attn"] = float(attn) if attn is not None else 0.0

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — COMPOSITE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _brand_match_score(candidate: dict, brand_name_query: str, brand_ids: set[str]) -> float:
    """
    Brand matching score — continuous, not binary:
      1.00  exact normalized match  (underscores/hyphens treated as spaces)
      0.75  found via KG brand-block query (fuzzy Neo4j lookup)
      0–0.74 partial credit for near-miss brands not in brand_block
             (difflib ratio × 0.74, floor at ratio ≥ 0.50)
      0.00  no signal
    """
    query_norm = _norm_brand(brand_name_query)
    cand_norm  = _norm_brand(candidate.get("brand_name") or "")
    if cand_norm == query_norm:
        return 1.0
    if candidate.get("sku_id") in brand_ids:
        return 0.75
    ratio = difflib.SequenceMatcher(None, query_norm, cand_norm).ratio()
    return round(ratio * 0.74, 4) if ratio >= 0.50 else 0.0


def _pkg_match_score(candidate: dict, pkg_quality: dict[str, float]) -> float:
    """Package match quality: 1.0 exact, 0.7 fuzzy, 0.0 not found."""
    return pkg_quality.get(candidate.get("sku_id", ""), 0.0)


def _composite_score(
    c: dict,
    brand_name_query: str,
    pkg_quality: dict[str, float],
    brand_ids: set[str],
    query_dims: dict | None = None,
) -> float:
    """
    Composite score (clamped to [0, 1]):

      45%  brand match   — continuous 0–1.0 (exact / brand-block / partial difflib)
      35%  package match — 1.0 / 0.90 (sorted-token) / 0.70 (fuzzy) tiers
      15%  ANN sim
       5%  reflect sim
      -10% anomaly       — capped at 0.50
      +5%  multi-signal  — bonus when brand + package + ANN all agree
      +8%  dim match     — when query + candidate share weight/length/width/height
    """
    brand   = _brand_match_score(c, brand_name_query, brand_ids)
    pkg     = _pkg_match_score(c, pkg_quality)
    ann     = float(c.get("ann_sim", 0.0))
    reflect = float(c.get("reflect_sim", 0.0))
    anomaly = min(float(c.get("anomaly_attn", 0.0)), 0.5)

    multi_signal_bonus = 0.05 if (brand > 0 and pkg > 0 and ann > 0) else 0.0
    dim_component = 0.0
    if has_comparable_dims(query_dims, c):
        dim_component = 0.08 * dim_boost(query_dims, c)

    score = (
        0.45 * brand
      + 0.35 * pkg
      + 0.15 * ann
      + 0.05 * reflect
      - 0.10 * anomaly
      + multi_signal_bonus
      + dim_component
    )
    return round(max(0.0, min(1.0, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CRITIC (Bedrock reasoning, required when KG path is used)
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


def _llm_decision(
    brand_name: str, package_type: str, best: dict, score: float,
    score_status: str, all_candidates: list[dict],
    query_dims: dict | None = None,
    dim_applied: bool = False,
    dim_mode: str = "none",
    product_risk: dict | None = None,
) -> dict[str, str]:
    """
    Ask the LLM to analyze the match and return both a recommendation indicator
    and a reasoning explanation.

    Returns {"indicator": "merged"|"updated"|"insert", "reasoning": "..."}.
    Falls back to score_status as indicator if the LLM fails or returns invalid JSON.
    """
    from agents.llm import get_llm, LLMError

    top3 = all_candidates[:3]
    qd = normalize_query_dims(query_dims)
    cand_lines = []
    for c in top3:
        line = (
            f"  - SKU {c.get('sku_id')}: brand={c.get('brand_name')}, "
            f"package={c.get('package_category_name')}, "
            f"composite_score={c.get('composite_score', 0):.4f}, "
            f"ann_sim={c.get('ann_sim', 0):.3f}, "
            f"reflect_sim={c.get('reflect_sim', 0):.3f}, "
            f"anomaly_attn={c.get('anomaly_attn', 0):.3f}"
        )
        if qd:
            line += f", dims={format_dims(c)}"
            if c.get("dim_boost") is not None:
                line += f", dim_boost={c.get('dim_boost'):.3f}"
        cand_lines.append(line)
    cand_text = "\n".join(cand_lines)

    dim_block = ""
    if qd:
        dim_block = (
            f"\nQUERY PHYSICAL DIMENSIONS: {format_dims(qd)}\n"
            f"Best candidate dimension check:\n"
            f"{format_dim_comparison(qd, best)}\n"
        )
        if dim_applied:
            dim_block += (
                "Physical dimensions were used to nudge or break ties between candidates.\n"
            )
        else:
            dim_block += (
                "Note: use dimensions in your reasoning when candidates are close.\n"
            )

    product_block = ""
    if product_risk:
        product_block = f"\n{format_product_risk_for_prompt(product_risk)}\n"

    prompt = f"""You are a SKU data-quality agent for a beverage distribution company.

A new product entry needs to be matched against the Master Global SKU database.

INPUT:
  brand_name   : {brand_name}
  package_type : {package_type}
{dim_block}{product_block}
TOP CANDIDATES FROM REFLEXIVE KNOWLEDGE GRAPH:
{cand_text}

COMPOSITE SCORE: {score:.4f}  (score-based status: {score_status})

Analyze:
- How well the brand and package descriptor align with the top candidate
- What the KG graph signals (ANN similarity, reflect neighbourhood, anomaly health) indicate
- Product ecosystem risk (SKU + neighbors) when provided — cite drivers if product risk exceeds SKU-only anomaly
- When query dimensions are provided, whether the top candidate's weight/length/width/height align
- Whether the match is confident, needs human review, or has no valid match

Based on your analysis, choose ONE indicator:
  "merged"  — confident match, safe to link automatically
  "updated" — partial or uncertain match, needs human review before linking
  "insert"  — no valid match found, a new GlobalSKU should be created

Return a JSON object with exactly two fields:
  "indicator" : one of "merged", "updated", "insert"
  "reasoning" : 2-3 sentences explaining your analysis and decision

Return JSON only. No extra text."""

    try:
        result = get_llm().json(prompt, max_tokens=300)
        indicator = str(result.get("indicator", "")).lower().strip()
        reasoning = str(result.get("reasoning", ""))
        if indicator not in ("merged", "updated", "insert"):
            indicator = score_status
        return {"indicator": indicator, "reasoning": reasoning}
    except (LLMError, Exception) as e:
        reasoning = _heuristic_reasoning(
            brand_name, package_type, best, score, score_status,
            set(), {},
        )
        return {"indicator": score_status, "reasoning": f"[LLM unavailable: {e}] {reasoning}"}


def _combine_status(score: float, llm_indicator: str) -> str:
    """
    Combine the composite score and the LLM indicator into a final status.

    Rules:
      score ≥ MERGE_THRESHOLD (0.85)  → always "merged"  (score is definitive)
      score < LLM_TRUST_MIN   (0.30)  → always "insert"  (ignore LLM; no signal)

    Ambiguous zone [LLM_TRUST_MIN, MERGE_THRESHOLD):
      LLM "merged"  + score ≥ UPDATE_THRESHOLD → "merged"
      LLM "merged"  + score < UPDATE_THRESHOLD → "updated"  (escalate for review)
      LLM "updated"                            → "updated"
      LLM "insert"                             → score-based fallback
    """
    if score >= MERGE_THRESHOLD:
        return "merged"
    if score < LLM_TRUST_MIN:
        return "insert"
    # Ambiguous zone: LLM can influence the final status
    if llm_indicator == "merged":
        return "merged" if score >= UPDATE_THRESHOLD else "updated"
    if llm_indicator == "updated":
        return "updated"
    # LLM says insert — fall back to score threshold
    return "updated" if score >= UPDATE_THRESHOLD else "insert"


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
    query_dims = normalize_query_dims(query_dims)

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
        score = _composite_score(c, brand_name, pkg_quality, brand_ids, query_dims)
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

    dim_applied = False
    dim_mode = "none"
    if has_query_dims(query_dims) and len(ranked) > 1:
        ranked, dim_applied, dim_mode = apply_dimension_ranking(
            ranked, query_dims, score_key="composite_score", tie_threshold=0.05,
        )

    best  = ranked[0]
    score = best["composite_score"]
    best_sku_id = str(best.get("sku_id") or "")

    product_risk = product_risk_for_sku(driver, best_sku_id)
    if product_risk:
        product_risk["brand_name"] = brand_name
        product_risk["package_type"] = package_type

    # ── Step 5: Critic — LLM indicator + reasoning + combined status ─────────
    score_status = (
        "merged"  if score >= MERGE_THRESHOLD  else
        "updated" if score >= UPDATE_THRESHOLD else
        "insert"
    )
    llm_result    = _llm_decision(
        brand_name, package_type, best, score, score_status, ranked,
        query_dims=query_dims, dim_applied=dim_applied, dim_mode=dim_mode,
        product_risk=product_risk,
    )
    llm_indicator = llm_result["indicator"]
    reasoning     = llm_result["reasoning"]
    status        = _combine_status(score, llm_indicator)

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
                "dim_boost":    round(float(c.get("dim_boost", 0)), 4),
            },
            "dim_boost":    c.get("dim_boost"),
            "dim_distance": c.get("dim_distance"),
            "signals": c.get("signals", []),
        }
        for c in ranked[:5]
    ]

    ambiguous = (
        len(ranked) >= 2
        and abs(
            float(ranked[0].get("composite_score_before_dim") or ranked[0]["composite_score"])
            - float(ranked[1].get("composite_score_before_dim") or ranked[1]["composite_score"])
        ) < 0.05
    )

    return {
        "status":        status,
        "score_status":  score_status,    # pure threshold decision before LLM influence
        "llm_indicator": llm_indicator,   # LLM's own recommendation
        "confidence":    score,
        "reasoning":     reasoning,
        "ambiguous":     ambiguous,
        "dim_applied":   dim_applied,
        "dim_mode":      dim_mode,
        "product_risk":  product_risk,
        "matched_skus": matched_skus,
        "query": {
            "brand_name":   brand_name,
            "package_type": package_type,
            "query_dims":   query_dims,
        },
        "kg_available": True,
        "pipeline": {
            "ann_candidates":    len(ann_results),
            "brand_block_hits":  len(brand_ids),
            "package_block_hits": len(pkg_ids),
            "total_candidates":  len(ranked),
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
