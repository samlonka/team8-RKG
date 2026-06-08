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
import json
import queue
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import GLOBAL_SKU_CSV
from data.master_loader import resolve_master_sku_records

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# ── Confidence thresholds ─────────────────────────────────────────────────────
MERGE_THRESHOLD  = 0.85   # ≥ this → merged (clear match)
UPDATE_THRESHOLD = 0.60   # ≥ this → updated (partial match)
                          # < UPDATE_THRESHOLD → insert

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Reflexive KG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

_SKU_DB: list[dict] | None = None
_SKU_DB_SOURCE: str = ""


def _load_sku_db() -> list[dict]:
    """Load master GlobalSKUs — PostgreSQL master_data preferred, CSV fallback."""
    global _SKU_DB, _SKU_DB_SOURCE
    if _SKU_DB is not None:
        return _SKU_DB
    _SKU_DB, _SKU_DB_SOURCE = resolve_master_sku_records()
    return _SKU_DB


def master_sku_source() -> str:
    """Human-readable label for where master SKUs were loaded from."""
    if _SKU_DB_SOURCE:
        return _SKU_DB_SOURCE
    return GLOBAL_SKU_CSV


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
    from agents.dim_match import candidate_dims
    dims = ["weight", "height", "length", "width"]
    has_any = any(candidate_dims(candidate).get(d) for d in dims)
    return 0.5 if not has_any else 0.5  # placeholder — overridden in disambiguate


def _dim_distance(a: dict, b: dict) -> float:
    from agents.dim_match import dim_distance
    return dim_distance(a, b)


def _rank_string_match_candidates(
    candidates: list[dict],
    query_dims: dict,
    *,
    score_key: str = "combined_score",
) -> tuple[list[dict], bool, str]:
    """Apply dimension nudge/tie-break to string-match candidate rows."""
    from agents.dim_match import apply_dimension_ranking, has_query_dims, normalize_query_dims

    if len(candidates) <= 1 or not has_query_dims(query_dims):
        return candidates, False, "none"

    wrapped = []
    for c in candidates:
        sku = c["sku"]
        wrapped.append({
            **{k: sku.get(k) for k in ("weight", "height", "length", "width")},
            score_key: c[score_key],
            "_orig": c,
        })

    reranked, applied, mode = apply_dimension_ranking(
        wrapped, normalize_query_dims(query_dims), score_key=score_key,
    )
    if not applied:
        return candidates, False, mode

    out: list[dict] = []
    for r in reranked:
        orig = r["_orig"]
        orig[score_key] = r[score_key]
        orig["dim_boost"] = r.get("dim_boost", 0.0)
        orig["dim_distance"] = r.get("dim_distance")
        orig["dim_mode"] = mode
        out.append(orig)
    return out, applied, mode


def _disambiguate(candidates: list[dict], query_dims: dict) -> list[dict]:
    """Re-rank string-match candidates using physical dimensions when available."""
    ranked, _, _ = _rank_string_match_candidates(candidates, query_dims)
    return ranked


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
    from agents.dim_match import has_query_dims, normalize_query_dims
    query_dims = normalize_query_dims(query_dims)

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

    dim_applied = False
    dim_mode = "none"
    if has_query_dims(query_dims) and len(top) > 1:
        top, dim_applied, dim_mode = _rank_string_match_candidates(top, query_dims)

    ambiguous = (
        len(top) >= 2
        and abs(
            float(top[0].get("combined_score_before_dim") or top[0]["combined_score"])
            - float(top[1].get("combined_score_before_dim") or top[1]["combined_score"])
        ) < 0.05
    )

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
            "dim_boost":            c.get("dim_boost"),
        }
        for c in top
    ]

    # ── Route to status ──────────────────────────────────────────────────────
    if confidence >= MERGE_THRESHOLD:
        status = "merged"
        reasoning = _build_merged_reasoning(
            brand_name, package_type, best, ambiguous, dim_applied, dim_mode,
        )
    elif confidence >= UPDATE_THRESHOLD:
        status = "updated"
        reasoning = _build_updated_reasoning(
            brand_name, package_type, best, ambiguous, dim_applied, dim_mode,
        )
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
        "dim_applied":  dim_applied,
        "dim_mode":     dim_mode,
        "matched_skus": matched_skus,
        "query": {
            "brand_name":   brand_name,
            "package_type": package_type,
            "query_dims":   query_dims,
        },
    }


def _dim_reasoning_suffix(best: dict, dim_applied: bool, dim_mode: str) -> str:
    boost = best.get("dim_boost") or 0
    if not dim_applied or boost <= 0:
        return ""
    if dim_mode == "tie_break":
        return (
            f" Ambiguity resolved using physical dimensions "
            f"(dim_boost={boost:.2f})."
        )
    return f" Physical dimensions supported the match (dim_boost={boost:.2f}, mode={dim_mode})."


def _build_merged_reasoning(
    brand: str, package: str, best: dict, ambiguous: bool,
    dim_applied: bool = False, dim_mode: str = "none",
) -> str:
    sku = best["sku"]
    parts = [
        f"High-confidence match to GlobalSKU {sku['sku_id']}.",
        f"Brand '{brand}' → '{sku['brand_name']}' (score={best['brand_score']:.2f}).",
        f"Package '{package}' → '{sku['package_category_name']}' (score={best['package_score']:.2f}).",
    ]
    suffix = _dim_reasoning_suffix(best, dim_applied, dim_mode)
    if suffix:
        parts.append(suffix.strip())
    return " ".join(parts)


def _build_updated_reasoning(
    brand: str, package: str, best: dict, ambiguous: bool,
    dim_applied: bool = False, dim_mode: str = "none",
) -> str:
    sku = best["sku"]
    parts = [
        f"Partial match to GlobalSKU {sku['sku_id']} — below merge threshold.",
        f"Brand '{brand}' → '{sku['brand_name']}' (score={best['brand_score']:.2f}).",
        f"Package '{package}' → '{sku['package_category_name']}' (score={best['package_score']:.2f}).",
        "Existing GlobalSKU record would need review/update.",
    ]
    suffix = _dim_reasoning_suffix(best, dim_applied, dim_mode)
    if suffix:
        parts.append(suffix.strip())
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


class QueryRequest(BaseModel):
    """Handbook §5 — natural-language investigation question."""
    question:   str = Field(..., min_length=3, description="Plain-English SKU lifecycle question")
    anchor_sku: str | None = Field(None, description="Optional GlobalSKU sku_id to anchor root-cause trace")
    scenario:   int | None = Field(
        None,
        ge=1,
        le=6,
        description="Force demo scenario 1–6 (skips keyword detection)",
    )
    weight: float | None = Field(None, description="Optional unit/case weight for catalog disambiguation")
    height: float | None = Field(None, description="Optional case height (inches) for catalog disambiguation")
    length: float | None = Field(None, description="Optional case length (inches) for catalog disambiguation")
    width:  float | None = Field(None, description="Optional case width (inches) for catalog disambiguation")


class QueryResponse(BaseModel):
    question:            str
    latency_seconds:     float
    summary:             str
    task_type:           str | None = None
    scenario:            int | None = None
    best_confidence:     float | None = None
    best_classification: str | None = None
    best_reasoning:      str | None = None
    doer_summary:        str | None = None
    best_chain:          dict | None = None
    validated_chains:    list[dict] = []
    candidate_summaries: list[dict] = []
    planner_rationale:   str = ""
    spec:                dict = {}
    tasks:               list[dict] = []
    catalog_query:       dict | None = None
    match_result:        dict | None = None
    duplicate_report:    dict | None = None
    closed_world_rows:   list[dict] | None = None
    reflexive_finding:   str | None = None
    display_limits:      list[dict] = []
    pipeline_events:     list[dict] = []


class IngestJobResponse(BaseModel):
    job_id:     str
    status:     str
    message:    str


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
    """Pre-load master SKU catalog into memory on startup."""
    db = _load_sku_db()
    print(f"[startup] Loaded {len(db):,} GlobalSKUs from {master_sku_source()}")


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


@app.post("/query/ask", response_model=QueryResponse)
def query_ask_endpoint(req: QueryRequest):
    """
    **API 1 — Natural-language agent query (handbook §5).**

    Routes by intent:
    - **catalog_match** — master catalog lookup (brand + package → GlobalSKU)
    - **lifecycle scenarios 1–6** — demo Cypher chains (optional `scenario` param)
    - **root_cause / risk_rank / anomaly_explain** — full graph investigation

    Examples:
    - "Is the product available in the master list: AQUA WATER 28OZ PL 1/15"
    - "Why did model accuracy degrade after the recent customer import?"
    - "Rank all GlobalSKUs by risk of causing training failures."
    """
    try:
        from api.query_service import run_nl_query
        payload = run_nl_query(
            req.question,
            anchor_sku=req.anchor_sku,
            scenario=req.scenario,
            query_dims={
                "weight": req.weight,
                "height": req.height,
                "length": req.length,
                "width": req.width,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        question=payload["question"],
        latency_seconds=payload["latency_seconds"],
        summary=payload["summary"],
        task_type=payload.get("task_type"),
        scenario=payload.get("scenario"),
        best_confidence=payload.get("best_confidence"),
        best_classification=payload.get("best_classification"),
        best_reasoning=payload.get("best_reasoning"),
        doer_summary=payload.get("doer_summary"),
        best_chain=payload.get("best_chain"),
        validated_chains=payload.get("validated_chains", []),
        candidate_summaries=payload.get("candidate_summaries", []),
        planner_rationale=payload.get("planner_rationale", ""),
        spec=payload.get("spec", {}),
        tasks=payload.get("tasks", []),
        catalog_query=payload.get("catalog_query"),
        match_result=payload.get("match_result"),
        duplicate_report=payload.get("duplicate_report"),
        closed_world_rows=payload.get("closed_world_rows"),
        reflexive_finding=payload.get("reflexive_finding"),
        display_limits=payload.get("display_limits", []),
        pipeline_events=payload.get("pipeline_events", []),
    )


@app.post("/query/ask/stream")
def query_ask_stream_endpoint(req: QueryRequest):
    """
    Stream live agent pipeline progress (SSE), then the final JSON result.

    Events use phases: supervisor → planner → doer → critic → complete.
    """
    event_queue: queue.Queue = queue.Queue()
    result_holder: dict[str, Any] = {}
    error_holder: list[Exception] = []

    def run_pipeline() -> None:
        try:
            from api.query_service import run_nl_query
            result_holder["payload"] = run_nl_query(
                req.question,
                anchor_sku=req.anchor_sku,
                scenario=req.scenario,
                query_dims={
                    "weight": req.weight,
                    "height": req.height,
                    "length": req.length,
                    "width": req.width,
                },
                on_event=lambda ev: event_queue.put(ev.to_dict()),
            )
        except Exception as exc:
            error_holder.append(exc)
        finally:
            event_queue.put(None)

    threading.Thread(target=run_pipeline, daemon=True).start()

    def generate():
        while True:
            item = event_queue.get()
            if item is None:
                if error_holder:
                    yield (
                        "data: "
                        + json.dumps({
                            "phase": "error",
                            "status": "error",
                            "title": "Something went wrong",
                            "detail": str(error_holder[0]),
                        })
                        + "\n\n"
                    )
                else:
                    yield (
                        "data: "
                        + json.dumps({
                            "phase": "complete",
                            "status": "done",
                            "title": "Answer ready",
                            "result": result_holder.get("payload", {}),
                        })
                        + "\n\n"
                    )
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tenant/ingest", response_model=IngestJobResponse)
async def tenant_ingest_endpoint(
    file: UploadFile = File(..., description="Tenant SKU Excel (SKU_Export.xlsx format)"),
    skip_validation: bool = False,
):
    """
    **API 2 — Async tenant Excel ingest.**

    Per row: normalize brand/package/UPC metadata → match against GlobalSKU master
    (vor_sku_data.csv graph) → **AUTO_MATCH** (merge), **REVIEW_QUEUE**, or **CREATE_NEW**.

    Poll ``GET /tenant/ingest/{job_id}`` until ``status`` is ``completed`` or ``failed``.
    Download annotated Excel via ``GET /tenant/ingest/{job_id}/download``.
    """
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=422, detail="Upload must be an Excel file (.xlsx)")

    from api.ingest_jobs import save_upload_and_create_job, start_job_async

    suffix = Path(file.filename).suffix or ".xlsx"
    job = save_upload_and_create_job(await file.read(), suffix=suffix)
    start_job_async(job.job_id, skip_validation=skip_validation)

    return IngestJobResponse(
        job_id=job.job_id,
        status="pending",
        message=(
            "Tenant ingest started. Poll GET /tenant/ingest/{job_id} for status; "
            "download results when completed."
        ),
    )


@app.get("/tenant/ingest/{job_id}")
def tenant_ingest_status(job_id: str):
    """Poll async tenant ingest job status and per-row reasoning."""
    from api.ingest_jobs import get_job, job_to_response

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job_to_response(job)


@app.get("/tenant/ingest/{job_id}/download")
def tenant_ingest_download(job_id: str):
    """Download the output Excel with RKG_Action / RKG_Reasoning columns appended."""
    from api.ingest_jobs import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != "completed" or not job.output_path:
        raise HTTPException(
            status_code=409,
            detail=f"Job status is '{job.status}'; output not ready",
        )
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file missing on disk")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


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
    from data.postgres_store import check_connection as pg_check, postgres_configured, table_counts, pg_session

    pg_ok, pg_msg = pg_check()
    pg_counts: dict[str, int] = {}
    if pg_ok:
        try:
            with pg_session() as conn:
                pg_counts = table_counts(conn)
        except Exception:
            pg_ok = False
            pg_msg = "Connected but could not read table counts"

    from config import POSTGRES_DB, POSTGRES_HOST

    return {
        "status":               "ok",
        "sku_count":            len(db),
        "master_sku_source":    master_sku_source(),
        "neo4j_available":      neo4j_available(),
        "embeddings_available": embeddings_available(),
        "postgres_configured":  postgres_configured(),
        "postgres_available":   pg_ok,
        "postgres_message":     pg_msg,
        "postgres_host":        POSTGRES_HOST or None,
        "postgres_db":          POSTGRES_DB or None,
        "postgres_counts":      pg_counts,
    }


@app.get("/")
def root():
    return {
        "message": "Reflexive KG API",
        "data_sources": {
            "master_global_skus": master_sku_source(),
            "seed_tenant_list":   "data/SKU_Export.xlsx",
        },
        "endpoints": {
            "POST /query/ask":              "NL question → Supervisor → Planner → Doer → Critic",
            "POST /query/ask/stream":       "Same pipeline with live SSE progress events",
            "POST /tenant/ingest":          "Upload tenant Excel → async merge/review/insert",
            "GET  /tenant/ingest/{job_id}": "Poll ingest job status + per-row reasoning",
            "GET  /tenant/ingest/{job_id}/download": "Download annotated output Excel",
            "POST /match":                  "String-matching (fast, no Neo4j required)",
            "POST /match/agent":            "Single SKU agent match (brand + package)",
            "GET  /health":                 "Health check",
            "GET  /docs":                   "Interactive Swagger UI",
        },
    }
