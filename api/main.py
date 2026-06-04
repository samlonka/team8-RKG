"""
api/main.py — SKU Matching REST API

POST /match
  Input : {"brand_name": "...", "package_type": "..."}
  Output: status (merged/updated/insert), reasoning, confidence, matched_skus

Matching pipeline:
  1. Exact brand + package match  → merged (high confidence)
  2. Fuzzy brand + package match  → merged or updated depending on confidence
  3. Ambiguous candidates         → resolve using H/W/L dimensions
  4. No match                     → insert (appends draft to Global data)
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import GLOBAL_SKU_CSV
from data.master_loader import load_master_sku_records

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# ── Confidence thresholds ─────────────────────────────────────────────────────
MERGE_THRESHOLD  = 0.85   # ≥ this → merged (clear match)
UPDATE_THRESHOLD = 0.60   # ≥ this → updated (partial match)
                          # < UPDATE_THRESHOLD → insert

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="SKU Matching API", version="1.0.0")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

_SKU_DB: list[dict] | None = None


def _load_sku_db() -> list[dict]:
    """Load master SKU CSV by column name (vor_sku_data.csv)."""
    global _SKU_DB
    if _SKU_DB is not None:
        return _SKU_DB
    _SKU_DB = load_master_sku_records(BASE_DIR / GLOBAL_SKU_CSV)
    return _SKU_DB


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Uppercase, collapse spaces/underscores/dashes, strip punctuation."""
    t = str(text).upper().strip()
    t = re.sub(r"[_\-\s]+", " ", t)
    t = re.sub(r"[^A-Z0-9 ./]", "", t)
    return t.strip()


def _tokens(text: str) -> set[str]:
    return set(_norm(text).split())


def _fuzzy_ratio(a: str, b: str) -> float:
    """difflib sequence matcher similarity [0..1]."""
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _token_overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _numeric_penalty(query: str, candidate: str) -> float:
    """
    Return a multiplier in (0, 1] to penalize mismatched numeric tokens.

    The FIRST number in a package name is the size/volume indicator (e.g. "10 OZ",
    "20 OZ", "1L", "1.5L"). A mismatch on that leading number is a strong signal
    that these are different products, so it gets a heavier penalty than other
    mismatched numbers.

    Examples:
      "10 OZ PL 1/24" vs "20OZ PL 1/24"  → leading "10" ≠ "20"  → 0.50
      "20OZ PL 1/24"  vs "20OZ PL 1/24"  → all match             → 1.0
      "1L PL 1/15"    vs "1L PL 1/12"    → trailing "15" ≠ "12"  → 0.70
    """
    q_nums = re.findall(r"\d+\.?\d*", _norm(query))
    c_nums = set(re.findall(r"\d+\.?\d*", _norm(candidate)))
    if not q_nums:
        return 1.0

    q_set = set(q_nums)
    mismatched = q_set - c_nums
    if not mismatched:
        return 1.0

    # Leading number (size/volume) mismatch → strong penalty
    leading_mismatch = q_nums[0] not in c_nums
    if leading_mismatch:
        # Additional tail mismatch beyond the leading one
        tail_mismatched = len(mismatched - {q_nums[0]}) / max(len(q_set) - 1, 1)
        return max(0.30, 0.50 - 0.15 * tail_mismatched)

    # Only trailing numbers mismatched (e.g. pack count)
    tail_ratio = len(mismatched) / len(q_set)
    return max(0.50, 1.0 - 0.50 * tail_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _brand_score(query: str, sku_brand: str) -> float:
    """Score how well query brand matches the SKU's brand_name."""
    if not query or not sku_brand:
        return 0.0
    qn, sn = _norm(query), _norm(sku_brand)
    if qn == sn:
        return 1.0
    # substring containment
    if qn in sn or sn in qn:
        return 0.92
    seq = _fuzzy_ratio(query, sku_brand)
    tok = _token_overlap(query, sku_brand)
    return max(seq, tok)


def _package_score(query: str, sku: dict) -> float:
    """
    Score how well query package_type matches the SKU's package fields.
    Numeric tokens (oz, count, size) are treated as hard constraints:
    mismatched numbers apply a penalty so "10 OZ" never scores high against "20 OZ".
    """
    if not query:
        return 0.0
    fields = [
        sku.get("package_category_name", ""),
        sku.get("package_name", ""),
        sku.get("short_description", ""),
    ]
    best = 0.0
    qn = _norm(query)
    for field in fields:
        if not field:
            continue
        fn = _norm(field)
        if qn == fn:
            return 1.0
        penalty = _numeric_penalty(query, field)
        if qn in fn or fn in qn:
            score = 0.90 * penalty
        else:
            seq = _fuzzy_ratio(query, field)
            tok = _token_overlap(query, field)
            score = max(seq, tok) * penalty
        best = max(best, score)
    return best


def _combined_score(brand_score: float, package_score: float) -> float:
    """Weighted combination: brand 40%, package 60%."""
    return 0.40 * brand_score + 0.60 * package_score


def _dimensional_similarity(candidate: dict, others: list[dict]) -> float:
    """
    Return a score in [0, 1] reflecting how well this candidate's dimensions
    stand out from the rest. Higher = more distinct / better match.
    Only useful when > 1 candidate exists.
    """
    dims = ["weight", "height", "length", "width"]
    has_any = any(candidate.get(d) is not None for d in dims)
    return 0.5 if not has_any else 0.5  # placeholder — overridden in disambiguate


def _dim_distance(a: dict, b: dict) -> float:
    """Euclidean distance across shared non-null dimensions."""
    dims = ["weight", "height", "length", "width"]
    diffs = []
    for d in dims:
        va, vb = a.get(d), b.get(d)
        if va is not None and vb is not None and va > 0 and vb > 0:
            diffs.append(((va - vb) / max(va, vb)) ** 2)
    if not diffs:
        return 1.0  # unknown — treat as distant
    import math
    return math.sqrt(sum(diffs) / len(diffs))


def _disambiguate(candidates: list[dict], query_dims: dict) -> list[dict]:
    """
    When multiple candidates have similar scores, use physical dimensions to
    break ties. candidates is a list of {sku, brand_score, package_score,
    combined_score, ...}. Returns re-ranked list.
    """
    if len(candidates) <= 1:
        return candidates

    # Only disambiguate if query carries any dimensional hint
    has_dim = any(query_dims.get(d) for d in ["weight", "height", "length", "width"])
    if not has_dim:
        return candidates

    for c in candidates:
        dist = _dim_distance(c["sku"], query_dims)
        # Closer to query dims → higher dim_boost
        c["dim_boost"] = max(0.0, 1.0 - dist)
        # Blend: 70% original combined_score, 30% dim similarity
        c["combined_score"] = 0.70 * c["combined_score"] + 0.30 * c["dim_boost"]

    return sorted(candidates, key=lambda x: x["combined_score"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# CORE MATCH FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def match_sku(
    brand_name: str,
    package_type: str,
    query_dims: dict | None = None,
    top_k: int = 5,
) -> dict:
    """
    Match brand_name + package_type against Global_sku.csv.

    Returns a result dict with: status, confidence, reasoning, matched_skus.
    """
    db = _load_sku_db()
    query_dims = query_dims or {}

    candidates: list[dict] = []

    for sku in db:
        bs = _brand_score(brand_name, sku["brand_name"])
        ps = _package_score(package_type, sku)
        cs = _combined_score(bs, ps)
        if cs >= 0.30:  # pre-filter: discard clearly irrelevant
            candidates.append({
                "sku": sku,
                "brand_score": round(bs, 4),
                "package_score": round(ps, 4),
                "combined_score": round(cs, 4),
                "dim_boost": 0.0,
            })

    if not candidates:
        return _build_insert_result(brand_name, package_type, "No candidates found above minimum threshold")

    # Sort by combined score descending
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    top = candidates[:top_k]

    # Detect ambiguity: top 2 are within 0.05 of each other
    ambiguous = (
        len(top) >= 2
        and abs(top[0]["combined_score"] - top[1]["combined_score"]) < 0.05
    )

    if ambiguous:
        top = _disambiguate(top, query_dims)
        top.sort(key=lambda x: x["combined_score"], reverse=True)

    best = top[0]
    confidence = best["combined_score"]
    best_sku = best["sku"]

    # Build matched_skus list for response
    matched_skus = [
        {
            "sku_id":               c["sku"]["sku_id"],
            "brand_name":           c["sku"]["brand_name"],
            "package_category_name":c["sku"]["package_category_name"],
            "package_name":         c["sku"]["package_name"],
            "package_type_id":      c["sku"]["package_type_id"],
            "status":               c["sku"]["status"],
            "weight":               c["sku"]["weight"],
            "height":               c["sku"]["height"],
            "length":               c["sku"]["length"],
            "width":                c["sku"]["width"],
            "brand_score":          c["brand_score"],
            "package_score":        c["package_score"],
            "confidence":           round(c["combined_score"], 4),
        }
        for c in top
    ]

    # ── Route to status ──────────────────────────────────────────────────────
    if confidence >= MERGE_THRESHOLD:
        status = "merged"
        reasoning = _build_merged_reasoning(brand_name, package_type, best, ambiguous)
    elif confidence >= UPDATE_THRESHOLD:
        status = "updated"
        reasoning = _build_updated_reasoning(brand_name, package_type, best, ambiguous)
    else:
        return _build_insert_result(
            brand_name, package_type,
            f"Best match confidence {confidence:.3f} is below update threshold {UPDATE_THRESHOLD}. "
            f"Closest was SKU {best_sku['sku_id']} "
            f"(brand={best['brand_score']:.2f}, package={best['package_score']:.2f})"
        )

    return {
        "status":       status,
        "confidence":   round(confidence, 4),
        "reasoning":    reasoning,
        "ambiguous":    ambiguous,
        "matched_skus": matched_skus,
        "query": {
            "brand_name":   brand_name,
            "package_type": package_type,
        },
    }


def _build_merged_reasoning(brand: str, package: str, best: dict, ambiguous: bool) -> str:
    sku = best["sku"]
    parts = [
        f"High-confidence match to GlobalSKU {sku['sku_id']}.",
        f"Brand '{brand}' → '{sku['brand_name']}' (score={best['brand_score']:.2f}).",
        f"Package '{package}' → '{sku['package_category_name']}' (score={best['package_score']:.2f}).",
    ]
    if ambiguous and best.get("dim_boost", 0) > 0:
        parts.append(
            f"Ambiguity resolved using physical dimensions "
            f"(dim_boost={best['dim_boost']:.2f})."
        )
    return " ".join(parts)


def _build_updated_reasoning(brand: str, package: str, best: dict, ambiguous: bool) -> str:
    sku = best["sku"]
    parts = [
        f"Partial match to GlobalSKU {sku['sku_id']} — below merge threshold.",
        f"Brand '{brand}' → '{sku['brand_name']}' (score={best['brand_score']:.2f}).",
        f"Package '{package}' → '{sku['package_category_name']}' (score={best['package_score']:.2f}).",
        "Existing GlobalSKU record would need review/update.",
    ]
    if ambiguous and best.get("dim_boost", 0) > 0:
        parts.append(
            f"Disambiguation used physical dimensions (dim_boost={best['dim_boost']:.2f})."
        )
    return " ".join(parts)


def _build_insert_result(brand: str, package: str, reason: str) -> dict:
    return {
        "status":       "insert",
        "confidence":   0.0,
        "reasoning":    (
            f"No sufficient match found for brand='{brand}', package_type='{package}'. "
            f"{reason}. "
            "A new GlobalSKU entry should be created in the Master data."
        ),
        "ambiguous":    False,
        "matched_skus": [],
        "query": {
            "brand_name":   brand,
            "package_type": package,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# API REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    brand_name:   str
    package_type: str
    # Optional dimensional hints for disambiguation
    weight: float | None = None
    height: float | None = None
    length: float | None = None
    width:  float | None = None


class MatchedSKU(BaseModel):
    sku_id:               str
    brand_name:           str
    package_category_name:str
    package_name:         str
    package_type_id:      str
    status:               str
    weight:               float | None
    height:               float | None
    length:               float | None
    width:                float | None
    brand_score:          float
    package_score:        float
    confidence:           float


class MatchResponse(BaseModel):
    status:       str          # merged | updated | insert
    confidence:   float
    reasoning:    str
    ambiguous:    bool
    matched_skus: list[MatchedSKU]
    query:        dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Pre-load SKU database into memory on startup."""
    db = _load_sku_db()
    print(f"[startup] Loaded {len(db):,} GlobalSKUs from {GLOBAL_SKU_CSV}")


@app.post("/match", response_model=MatchResponse)
def match_endpoint(req: MatchRequest):
    """
    Match a brand_name + package_type against the Master Global SKU database.

    Returns:
      - **merged**:  high-confidence match to an existing GlobalSKU
      - **updated**: partial match; existing record needs review
      - **insert**:  no match found; a new GlobalSKU should be created
    """
    if not req.brand_name.strip():
        raise HTTPException(status_code=422, detail="brand_name must not be empty")
    if not req.package_type.strip():
        raise HTTPException(status_code=422, detail="package_type must not be empty")

    query_dims = {
        "weight": req.weight,
        "height": req.height,
        "length": req.length,
        "width":  req.width,
    }

    result = match_sku(
        brand_name=req.brand_name.strip(),
        package_type=req.package_type.strip(),
        query_dims=query_dims,
    )
    return result


@app.post("/match/agent")
def match_agent_endpoint(req: MatchRequest):
    """
    Agent + Reflexive KG matching pipeline.

    Uses sentence-transformer embeddings + Neo4j ANN search + KG graph signals
    (brand-block, package-block) + reflexive embedding health scoring.

    Score breakdown per candidate:
      - **ann_sim**:      cosine similarity of query embedding vs SKU self_emb
      - **graph_boost**:  1.0 if SKU found via brand/package graph edges
      - **reflect_sim**:  cosine similarity vs SKU's reflect_emb (neighborhood context)
      - **anomaly_attn**: KG health penalty — high = anomalous/inconsistent data

    Falls back to string-matching if Neo4j or embeddings are unavailable.
    """
    if not req.brand_name.strip():
        raise HTTPException(status_code=422, detail="brand_name must not be empty")
    if not req.package_type.strip():
        raise HTTPException(status_code=422, detail="package_type must not be empty")

    from api.agent_matcher import agent_match
    query_dims = {
        "weight": req.weight,
        "height": req.height,
        "length": req.length,
        "width":  req.width,
    }
    return agent_match(
        brand_name=req.brand_name.strip(),
        package_type=req.package_type.strip(),
        query_dims=query_dims,
    )


@app.get("/health")
def health():
    db = _load_sku_db()
    from api.agent_matcher import neo4j_available, embeddings_available
    return {
        "status":               "ok",
        "sku_count":            len(db),
        "neo4j_available":      neo4j_available(),
        "embeddings_available": embeddings_available(),
    }


@app.get("/")
def root():
    return {
        "message": "SKU Matching API",
        "endpoints": {
            "POST /match":       "String-matching (fast, no Neo4j required)",
            "POST /match/agent": "Agent + Reflexive KG (embeddings + graph signals + anomaly health)",
            "GET  /health":      "Health check",
            "GET  /docs":        "Interactive Swagger UI",
        },
    }
