"""
agents/dim_match.py — Physical dimension parsing, tie-breaking, and LLM context.

Used by catalog_match (Supervisor → Doer → agent_matcher → Critic) and
legacy string matching in api/main.py.
"""

from __future__ import annotations

import math
import re
from typing import Any

DIM_FIELDS: tuple[str, ...] = ("weight", "height", "length", "width")

# Explicit "Weight: 10.5" / "Case Length (Inches): 12" style fields.
_EXPLICIT_DIM_RES: list[tuple[str, re.Pattern[str]]] = [
    ("weight", re.compile(
        r"(?:unit\s+)?weight\s*(?:\([^)]*\))?\s*:?\s*([\d.]+)\s*(?:lb|lbs|pound|pounds|oz|kg)?",
        re.I,
    )),
    ("height", re.compile(
        r"(?:case\s+)?height\s*(?:\([^)]*\))?\s*:?\s*([\d.]+)\s*(?:in|inches|\"|cm)?",
        re.I,
    )),
    ("length", re.compile(
        r"(?:case\s+)?length\s*(?:\([^)]*\))?\s*:?\s*([\d.]+)\s*(?:in|inches|\"|cm)?",
        re.I,
    )),
    ("width", re.compile(
        r"(?:case\s+)?width\s*(?:\([^)]*\))?\s*:?\s*([\d.]+)\s*(?:in|inches|\"|cm)?",
        re.I,
    )),
]

# L=12 W=8 H=10 or L:12, W:8, H:10
_LWH_COMPACT = re.compile(
    r"\bL\s*[=:]\s*([\d.]+)\s*[,/\s]+\s*W\s*[=:]\s*([\d.]+)\s*[,/\s]+\s*H\s*[=:]\s*([\d.]+)",
    re.I,
)

# 12 x 8 x 10 (inches) — length x width x height
_LWH_X = re.compile(
    r"\b([\d.]+)\s*[x×]\s*([\d.]+)\s*[x×]\s*([\d.]+)\s*(?:in|inches|\"|cm)?\b",
    re.I,
)


def _positive_float(val: Any) -> float | None:
    try:
        v = float(val)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_query_dims(raw: dict[str, Any] | None) -> dict[str, float]:
    """Return only non-zero dimension hints."""
    if not raw:
        return {}
    out: dict[str, float] = {}
    for d in DIM_FIELDS:
        v = _positive_float(raw.get(d))
        if v is not None:
            out[d] = v
    return out


def has_query_dims(query_dims: dict[str, Any] | None) -> bool:
    return bool(normalize_query_dims(query_dims))


def merge_query_dims(*sources: dict[str, Any] | None) -> dict[str, float]:
    """Later sources override earlier ones for each dimension."""
    merged: dict[str, float] = {}
    for src in sources:
        for k, v in normalize_query_dims(src).items():
            merged[k] = v
    return merged


def parse_dimensions_from_text(text: str) -> dict[str, float]:
    """
    Extract weight / height / length / width hints from natural language.
    Returns normalized dict (only positive values).
    """
    if not text or not text.strip():
        return {}

    dims: dict[str, float] = {}

    for field, pattern in _EXPLICIT_DIM_RES:
        m = pattern.search(text)
        if m:
            v = _positive_float(m.group(1))
            if v is not None:
                dims[field] = v

    m = _LWH_COMPACT.search(text)
    if m:
        ln, wd, ht = (_positive_float(m.group(i)) for i in (1, 2, 3))
        if ln is not None:
            dims.setdefault("length", ln)
        if wd is not None:
            dims.setdefault("width", wd)
        if ht is not None:
            dims.setdefault("height", ht)

    m = _LWH_X.search(text)
    if m:
        ln, wd, ht = (_positive_float(m.group(i)) for i in (1, 2, 3))
        if ln is not None:
            dims.setdefault("length", ln)
        if wd is not None:
            dims.setdefault("width", wd)
        if ht is not None:
            dims.setdefault("height", ht)

    return normalize_query_dims(dims)


def candidate_dims(record: dict[str, Any]) -> dict[str, float]:
    """Pull dimension fields from a SKU / candidate dict."""
    return normalize_query_dims({d: record.get(d) for d in DIM_FIELDS})


def dim_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """
    Normalised Euclidean distance across shared non-null dimensions.
    0.0 = perfect match, higher = worse. Returns 1.0 when no comparable dims.
    """
    diffs: list[float] = []
    for d in DIM_FIELDS:
        va = _positive_float(a.get(d))
        vb = _positive_float(b.get(d))
        if va is not None and vb is not None:
            diffs.append(((va - vb) / max(va, vb)) ** 2)
    if not diffs:
        return 1.0
    return math.sqrt(sum(diffs) / len(diffs))


def dim_boost(query_dims: dict[str, Any], candidate: dict[str, Any]) -> float:
    """Similarity in [0, 1] — higher when candidate dims are closer to query."""
    if not has_query_dims(query_dims):
        return 0.0
    dist = dim_distance(candidate, query_dims)
    return max(0.0, 1.0 - dist)


def is_ambiguous_scores(scores: list[float], tie_threshold: float = 0.05) -> bool:
    if len(scores) < 2:
        return False
    return abs(scores[0] - scores[1]) < tie_threshold


def overlapping_fields(
    query_dims: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> list[str]:
    """Dimension fields present and positive on both query and candidate."""
    qd = normalize_query_dims(query_dims)
    cd = candidate_dims(candidate or {})
    return [f for f in DIM_FIELDS if f in qd and f in cd]


def has_comparable_dims(
    query_dims: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> bool:
    return bool(overlapping_fields(query_dims, candidate))


def dims_from_vendor_row(row: dict[str, Any]) -> dict[str, float]:
    """Map tenant/vendor ingest columns to master-catalog dimension keys."""
    return normalize_query_dims({
        "weight": row.get("unit_weight") or row.get("weight"),
        "length": row.get("case_length") or row.get("length"),
        "width":  row.get("case_width") or row.get("width"),
        "height": row.get("case_height") or row.get("height"),
    })


def enrich_candidates_with_dims(
    candidates: list[dict[str, Any]],
    query_dims: dict[str, Any] | None,
) -> bool:
    """
    Annotate each candidate with dim_boost / dim_distance when query dims exist.
    Returns True if any candidate had comparable dimensions.
    """
    qd = normalize_query_dims(query_dims)
    if not qd:
        return False

    any_comparable = False
    for c in candidates:
        cd = candidate_dims(c)
        overlap = overlapping_fields(qd, cd)
        if overlap:
            any_comparable = True
            sub_q = {k: qd[k] for k in overlap}
            sub_c = {k: cd[k] for k in overlap}
            dist = dim_distance(sub_c, sub_q)
            boost = max(0.0, 1.0 - dist)
        else:
            dist = 1.0
            boost = 0.0
        c["dim_distance"] = round(dist, 4)
        c["dim_boost"] = round(boost, 4)
        c["dim_overlap_fields"] = overlap
    return any_comparable


def apply_dimension_ranking(
    candidates: list[dict[str, Any]],
    query_dims: dict[str, Any] | None,
    *,
    score_key: str = "composite_score",
    tie_threshold: float = 0.05,
) -> tuple[list[dict[str, Any]], bool, str]:
    """
    Use physical dimensions whenever query hints overlap candidate data.

    - Always annotates dim_boost / dim_distance when query dims exist.
    - tie_break: top scores within tie_threshold → 70% score + 30% dim_boost
    - nudge: comparable dims but not tied → 88% score + 12% dim_boost
    - annotate_only: query dims but no candidate overlap → metrics only, no re-rank

    Returns (candidates, applied, mode).
    """
    if len(candidates) <= 1:
        return candidates, False, "none"

    comparable = enrich_candidates_with_dims(candidates, query_dims)
    qd = normalize_query_dims(query_dims)
    if not qd:
        return candidates, False, "none"

    if not comparable:
        return candidates, False, "annotate_only"

    scores = [float(c.get(score_key) or 0) for c in candidates]
    ambiguous = is_ambiguous_scores(scores, tie_threshold)
    if ambiguous:
        blend_original, blend_dim, mode = 0.70, 0.30, "tie_break"
    else:
        blend_original, blend_dim, mode = 0.88, 0.12, "nudge"

    for c in candidates:
        if not c.get("dim_overlap_fields"):
            continue
        base = float(c.get(score_key) or 0)
        c[f"{score_key}_before_dim"] = base
        boost = float(c.get("dim_boost") or 0)
        c[score_key] = round(blend_original * base + blend_dim * boost, 4)

    candidates.sort(key=lambda x: float(x.get(score_key) or 0), reverse=True)
    return candidates, True, mode


def apply_dim_disambiguation(
    candidates: list[dict[str, Any]],
    query_dims: dict[str, Any] | None,
    *,
    score_key: str = "composite_score",
    tie_threshold: float = 0.05,
    blend_original: float = 0.70,
    blend_dim: float = 0.30,
    force_when_dims: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Back-compat wrapper. Prefer apply_dimension_ranking for new code.

    When force_when_dims=True, applies nudge blending whenever comparable dims exist.
    """
    if force_when_dims:
        ranked, applied, _ = apply_dimension_ranking(
            candidates, query_dims, score_key=score_key, tie_threshold=tie_threshold,
        )
        return ranked, applied

    qd = normalize_query_dims(query_dims)
    if not qd or len(candidates) <= 1:
        return candidates, False

    scores = [float(c.get(score_key) or 0) for c in candidates]
    ambiguous = is_ambiguous_scores(scores, tie_threshold)
    if not ambiguous:
        enrich_candidates_with_dims(candidates, query_dims)
        return candidates, False

    for c in candidates:
        cd = candidate_dims(c)
        dist = dim_distance(cd, qd)
        boost = max(0.0, 1.0 - dist)
        c["dim_distance"] = round(dist, 4)
        c["dim_boost"] = round(boost, 4)
        base = float(c.get(score_key) or 0)
        c[f"{score_key}_before_dim"] = base
        c[score_key] = round(blend_original * base + blend_dim * boost, 4)

    candidates.sort(key=lambda x: float(x.get(score_key) or 0), reverse=True)
    return candidates, True


def format_dims(dims: dict[str, Any] | None, *, prefix: str = "") -> str:
    """Human-readable dimension string for logs and LLM prompts."""
    d = normalize_query_dims(dims)
    if not d:
        return f"{prefix}(none)" if prefix else "(none)"
    parts = [f"{k}={d[k]:g}" for k in DIM_FIELDS if k in d]
    body = ", ".join(parts)
    return f"{prefix}{body}" if prefix else body


def format_dim_comparison(
    query_dims: dict[str, Any] | None,
    candidate_dims_map: dict[str, Any] | None,
) -> str:
    """Side-by-side comparison for LLM reasoning."""
    qd = normalize_query_dims(query_dims)
    cd = normalize_query_dims(candidate_dims_map)
    if not qd:
        return "No query dimension hints provided."
    if not cd:
        return f"Query dims: {format_dims(qd)}. Candidate has no stored dimensions."

    lines = [f"Query: {format_dims(qd)}", f"Candidate: {format_dims(cd)}"]
    for field in DIM_FIELDS:
        if field in qd and field in cd:
            dist_pct = abs(qd[field] - cd[field]) / max(qd[field], cd[field]) * 100
            lines.append(f"  {field}: {dist_pct:.1f}% relative difference")
        elif field in qd:
            lines.append(f"  {field}: missing on candidate")
    boost = dim_boost(qd, cd)
    lines.append(f"Dimension similarity score: {boost:.3f}")
    return "\n".join(lines)
