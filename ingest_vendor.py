"""
ingest_vendor.py — Incremental vendor SKU ingestion pipeline.

Processes a new vendor Excel file against the existing Global SKU graph:

  Step 1  Normalize  — parse and clean incoming rows (reuses 02_seed_data.py)
  Step 2  Delta      — classify each row:
                         NEW          product_id not yet in graph
                         UNCHANGED    product_id exists, key fields identical
                         FIELD_UPDATE product_id exists, fields changed
                         UPC_CONFLICT existing row has a different MAPS_TO target
  Step 3  Embed      — generate self_emb for NEW and FIELD_UPDATE rows only
  Step 4  Match      — multi-signal candidate generation per new/updated row:
                         a. Exact UPC      retail_upc / case_upc exact string match
                         b. Fuzzy UPC      strip leading zeros, handle GTIN variants
                         c. ANN self_emb   Neo4j vector index top-K cosine search
                         d. Brand block    GlobalSKUs sharing a fuzzy-matched brand
  Step 5  Route      — confidence-gated decision:
                         ≥ 0.90  AUTO_MATCH    create MAPS_TO edge immediately
                         ≥ 0.65  REVIEW_QUEUE  write MatchCandidate for human approval
                         < 0.65  CREATE_NEW    write GlobalSKUDraft for analyst review
  Step 6  Validate   — re-run attention reflection on affected GlobalSKUs;
                         alert if anomaly_attn rises by ≥ MATCH_ANOMALY_ALERT_DELTA

New Neo4j node types:
  MatchCandidate  — pending match awaiting human approval
  GlobalSKUDraft  — proposed new GlobalSKU derived from unmatched vendor row
  IngestionRun    — audit record for every ingest run

Usage:
    python ingest_vendor.py data/new_client.xlsx     # ingest a vendor file
    python ingest_vendor.py --review-queue           # show items awaiting review
    python ingest_vendor.py --approve <product_id>   # approve top match candidate
    python ingest_vendor.py --reject  <product_id>   # reject all candidates → draft
    python ingest_vendor.py --report                 # show last ingestion run summary
"""

from __future__ import annotations

import argparse
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import importlib

import numpy as np
import torch
from tqdm import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBEDDING_MODEL, EMBED_BATCH_SIZE,
    MATCH_AUTO_THRESHOLD, MATCH_REVIEW_THRESHOLD,
    MATCH_ANN_TOP_K, MATCH_ANOMALY_ALERT_DELTA,
)

# Modules whose names start with a digit cannot be imported with 'from X import'
_seed  = importlib.import_module("02_seed_data")
load_vendor_sku    = _seed.load_vendor_sku
vendor_sku_to_text = _seed.vendor_sku_to_text


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    global_sku_id: str
    confidence:    float
    signals:       list[str]      # e.g. ['exact_upc', 'ann', 'brand_block']
    emb_similarity: float = 0.0
    global_brand:  str = ""
    global_category: str = ""
    global_upc:    str = ""


@dataclass
class Decision:
    product_id:    str
    action:        str            # AUTO_MATCH | REVIEW_QUEUE | CREATE_NEW | UNCHANGED
    best:          Candidate | None = None
    all_candidates: list[Candidate] = field(default_factory=list)
    delta_status:  str = "NEW"    # NEW | FIELD_UPDATE | UNCHANGED | UPC_CONFLICT


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_schema(session) -> None:
    """Create indexes for new node types introduced by the ingestion pipeline."""
    ddl = [
        "CREATE INDEX idx_match_candidate_vendor IF NOT EXISTS "
        "FOR (n:MatchCandidate) ON (n.vendor_sku_id)",

        "CREATE INDEX idx_match_candidate_status IF NOT EXISTS "
        "FOR (n:MatchCandidate) ON (n.status)",

        "CREATE INDEX idx_global_sku_draft IF NOT EXISTS "
        "FOR (n:GlobalSKUDraft) ON (n.draft_id)",

        "CREATE INDEX idx_ingestion_run IF NOT EXISTS "
        "FOR (n:IngestionRun) ON (n.run_id)",

        "CREATE INDEX idx_ingestion_run_at IF NOT EXISTS "
        "FOR (n:IngestionRun) ON (n.started_at)",
    ]
    for stmt in ddl:
        session.run(stmt)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — NORMALIZE  (wraps 02_seed_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(path: str):
    """Load and normalize a vendor Excel file. Returns a DataFrame."""
    return load_vendor_sku(path)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DELTA DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_FINGERPRINT_COLS = [
    "brand", "supplier", "product_class", "product_description",
    "retail_upc", "case_upc", "units_per_case", "unit_weight",
]


def _fingerprint(row: dict) -> str:
    """Hash the key fields of a vendor row for change detection."""
    parts = "|".join(str(row.get(c, "")) for c in _FINGERPRINT_COLS)
    return hashlib.md5(parts.encode()).hexdigest()


def detect_delta(session, df) -> tuple[dict, dict]:
    """
    Compare incoming rows against existing VendorSKU nodes.

    Returns:
        classified  : {product_id: 'NEW'|'UNCHANGED'|'FIELD_UPDATE'|'UPC_CONFLICT'}
        existing_fp : {product_id: fingerprint} for rows already in graph
    """
    existing = session.run(
        """
        MATCH (v:VendorSKU)
        RETURN v.product_id    AS pid,
               v._fingerprint  AS fp,
               v.retail_upc    AS rupc,
               v.case_upc      AS cupc
        """
    ).data()

    existing_map: dict[str, dict] = {
        r["pid"]: {"fp": r["fp"], "retail_upc": r["rupc"], "case_upc": r["cupc"]}
        for r in existing
        if r["pid"]
    }

    classified: dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row["product_id"])
        fp  = _fingerprint(row.to_dict())

        if pid not in existing_map:
            classified[pid] = "NEW"
            continue

        prev = existing_map[pid]
        if prev["fp"] == fp:
            classified[pid] = "UNCHANGED"
            continue

        # Check if UPC changed → re-run matching
        new_rupc  = str(row.get("retail_upc") or "")
        prev_rupc = str(prev.get("retail_upc") or "")
        if new_rupc and prev_rupc and new_rupc != prev_rupc:
            classified[pid] = "UPC_CONFLICT"
        else:
            classified[pid] = "FIELD_UPDATE"

    counts = {k: sum(1 for v in classified.values() if v == k)
              for k in ("NEW", "UNCHANGED", "FIELD_UPDATE", "UPC_CONFLICT")}
    print(f"  Delta: NEW={counts['NEW']} UNCHANGED={counts['UNCHANGED']} "
          f"FIELD_UPDATE={counts['FIELD_UPDATE']} UPC_CONFLICT={counts['UPC_CONFLICT']}")
    return classified, existing_map


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EMBED
# ─────────────────────────────────────────────────────────────────────────────

def embed_rows(df, classified: dict[str, str], model: SentenceTransformer) -> dict[str, np.ndarray]:
    """
    Generate self_emb only for rows that need matching (NEW / FIELD_UPDATE / UPC_CONFLICT).
    Returns {product_id: embedding_array}.
    """
    active = df[df["product_id"].astype(str).isin(
        {pid for pid, status in classified.items()
         if status in ("NEW", "FIELD_UPDATE", "UPC_CONFLICT")}
    )]

    if active.empty:
        return {}

    texts = [vendor_sku_to_text(row.to_dict()) for _, row in active.iterrows()]
    embs  = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return {
        str(row["product_id"]): embs[i]
        for i, (_, row) in enumerate(active.iterrows())
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MULTI-SIGNAL CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_upc(upc: str | None) -> str:
    """Strip non-digits and leading zeros for fuzzy comparison."""
    if not upc:
        return ""
    import re
    digits = re.sub(r"[^0-9]", "", str(upc))
    return digits.lstrip("0")


def _vendor_query_upcs(row: dict) -> list[str]:
    """Normalized UPCs from a vendor row for master matching."""
    from data.master_loader import normalize_upc
    out = []
    for field in ("retail_upc", "case_upc", "eaches_upc"):
        u = normalize_upc(row.get(field))
        if u:
            out.append(u)
    return list(dict.fromkeys(out))


def _exact_upc_candidates(session, row: dict) -> list[Candidate]:
    """Exact match on any master UPC alias (primary + each/case/unit/package)."""
    upcs = _vendor_query_upcs(row)
    if not upcs:
        return []

    rows = session.run(
        """
        MATCH (g:GlobalSKU)
        WHERE any(u IN $upcs WHERE u IN coalesce(g.upc_aliases, []))
           OR g.upc IN $upcs
           OR g.each_upc IN $upcs
           OR g.case_upc IN $upcs
           OR g.unit_upc IN $upcs
           OR g.package_upc IN $upcs
        RETURN g.sku_id AS sid, g.brand_family AS brand,
               g.product_category AS cat, g.upc AS upc
        LIMIT 5
        """,
        upcs=upcs,
    ).data()

    return [
        Candidate(
            global_sku_id=r["sid"],
            confidence=1.0,
            signals=["exact_upc"],
            emb_similarity=1.0,
            global_brand=r["brand"] or "",
            global_category=r["cat"] or "",
            global_upc=r["upc"] or "",
        )
        for r in rows
    ]


def _fuzzy_upc_candidates(session, row: dict) -> list[Candidate]:
    """
    GTIN-variant matching: strip leading zeros and compare digit strings.
    Catches '0012345678' vs '12345678' mismatches common in vendor exports.
    """
    norm_upcs = list(dict.fromkeys(_vendor_query_upcs(row)))
    if not norm_upcs:
        return []

    rows = session.run(
        """
        MATCH (g:GlobalSKU)
        WHERE any(u IN $upcs WHERE
            any(alias IN coalesce(g.upc_aliases, []) WHERE
                alias CONTAINS u OR u CONTAINS alias
            )
            OR (g.upc IS NOT NULL AND (
                replace(replace(replace(g.upc, '0', ''), ' ', ''), '-', '') CONTAINS u
                OR u CONTAINS replace(replace(replace(g.upc, '0', ''), ' ', ''), '-', '')
            ))
        )
        RETURN g.sku_id AS sid, g.brand_family AS brand,
               g.product_category AS cat, g.upc AS upc
        LIMIT 5
        """,
        upcs=norm_upcs,
    ).data()

    return [
        Candidate(
            global_sku_id=r["sid"],
            confidence=0.92,
            signals=["fuzzy_upc"],
            emb_similarity=0.92,
            global_brand=r["brand"] or "",
            global_category=r["cat"] or "",
            global_upc=r["upc"] or "",
        )
        for r in rows
    ]


def _ann_candidates(session, emb: np.ndarray) -> list[Candidate]:
    """
    ANN search on idx_global_sku_self using the vendor SKU's embedding.
    Returns top-K GlobalSKUs by cosine similarity.
    """
    rows = session.run(
        """
        CALL db.index.vector.queryNodes('idx_global_sku_self', $k, $vec)
        YIELD node AS g, score
        RETURN g.sku_id AS sid, g.brand_family AS brand,
               g.product_category AS cat, g.upc AS upc, score
        """,
        k=MATCH_ANN_TOP_K,
        vec=emb.tolist(),
    ).data()

    candidates = []
    for r in rows:
        sim = float(r["score"])
        conf = (
            0.88 if sim >= 0.95 else
            0.78 if sim >= 0.90 else
            0.68 if sim >= 0.85 else
            0.58 if sim >= 0.80 else
            0.0
        )
        if conf > 0:
            candidates.append(Candidate(
                global_sku_id=r["sid"],
                confidence=conf,
                signals=["ann"],
                emb_similarity=sim,
                global_brand=r["brand"] or "",
                global_category=r["cat"] or "",
                global_upc=r["upc"] or "",
            ))
    return candidates


def _brand_block_candidates(session, row: dict) -> list[str]:
    """
    Return GlobalSKU IDs that share a fuzzy-matched brand with this vendor row.
    Used to boost confidence when combined with ANN similarity.
    """
    vendor_brand = str(row.get("brand", "")).strip().upper()
    if not vendor_brand or vendor_brand == "UNKNOWN":
        return []

    rows = session.run(
        """
        MATCH (b:Brand)
        WHERE toUpper(b.brand_family) CONTAINS $brand
           OR $brand CONTAINS toUpper(b.brand_family)
        MATCH (g:GlobalSKU)-[:BELONGS_TO_BRAND]->(b)
        RETURN DISTINCT g.sku_id AS sid
        LIMIT 30
        """,
        brand=vendor_brand,
    ).data()
    return [r["sid"] for r in rows]


def _numeric_match(row: dict, global_node: dict) -> bool:
    """True if units_per_case and/or weight are within 15% of each other."""
    def _close(a, b, tol=0.15):
        try:
            fa, fb = float(a), float(b)
            return fa > 0 and fb > 0 and abs(fa - fb) / max(fa, fb) <= tol
        except (TypeError, ValueError):
            return False

    return (
        _close(row.get("units_per_case"), global_node.get("units_per_case"))
        or _close(row.get("unit_weight"),  global_node.get("weight"))
    )


def find_candidates(session, row: dict, emb: np.ndarray) -> list[Candidate]:
    """
    Run all four matching signals and merge into a deduplicated candidate list.
    Candidates for the same GlobalSKU are merged with the highest confidence
    and union of signals.
    """
    raw: list[Candidate] = []
    raw += _exact_upc_candidates(session, row)
    raw += _fuzzy_upc_candidates(session, row)
    raw += _ann_candidates(session, emb)

    brand_ids = set(_brand_block_candidates(session, row))

    # Merge by global_sku_id
    merged: dict[str, Candidate] = {}
    for c in raw:
        sid = c.global_sku_id
        if sid in merged:
            existing = merged[sid]
            # Merge signals, keep highest confidence
            existing.signals = list(set(existing.signals) | set(c.signals))
            existing.confidence = max(existing.confidence, c.confidence)
            existing.emb_similarity = max(existing.emb_similarity, c.emb_similarity)
        else:
            merged[sid] = c

    # Apply brand-block boost
    for sid, c in merged.items():
        if sid in brand_ids and "ann" in c.signals:
            c.signals.append("brand_block")
            c.confidence = min(c.confidence + 0.08, 0.95)

    # Fetch numeric fields for top candidates and apply numeric boost
    if merged:
        top_sids = sorted(merged, key=lambda s: merged[s].confidence, reverse=True)[:5]
        g_rows = session.run(
            """
            MATCH (g:GlobalSKU) WHERE g.sku_id IN $sids
            RETURN g.sku_id AS sid, g.units_per_case AS upc, g.weight AS wt
            """,
            sids=top_sids,
        ).data()
        for gr in g_rows:
            sid = gr["sid"]
            if sid in merged and _numeric_match(row, gr):
                merged[sid].signals.append("numeric")
                merged[sid].confidence = min(merged[sid].confidence + 0.04, 0.95)

    return sorted(merged.values(), key=lambda c: c.confidence, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CONFIDENCE SCORING AND ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def route(candidates: list[Candidate], delta_status: str) -> Decision:
    """
    Route a vendor row to AUTO_MATCH, REVIEW_QUEUE, or CREATE_NEW based on
    the best candidate's confidence.
    """
    # UPC_CONFLICT always needs review regardless of confidence
    force_review = (delta_status == "UPC_CONFLICT")

    if not candidates:
        return Decision(
            product_id="",
            action="CREATE_NEW",
            best=None,
            all_candidates=[],
            delta_status=delta_status,
        )

    best = candidates[0]

    if best.confidence >= MATCH_AUTO_THRESHOLD and not force_review:
        action = "AUTO_MATCH"
    elif best.confidence >= MATCH_REVIEW_THRESHOLD or force_review:
        action = "REVIEW_QUEUE"
    else:
        action = "CREATE_NEW"

    return Decision(
        product_id="",     # filled by caller
        action=action,
        best=best,
        all_candidates=candidates,
        delta_status=delta_status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5B — EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_vendor_sku(session, row: dict, emb: np.ndarray, run_id: str):
    """Write (or update) a VendorSKU node with its self_emb and fingerprint."""
    fp = _fingerprint(row)
    session.run(
        """
        MERGE (v:VendorSKU {product_id: $pid})
        SET v.product_description = $desc,
            v.brand               = $brand,
            v.supplier            = $supplier,
            v.product_class       = $cls,
            v.warehouse           = $wh,
            v.units_per_case      = $upc,
            v.unit_weight         = $wt,
            v.case_length         = $cl,
            v.case_width          = $cw,
            v.case_height         = $ch,
            v.case_upc            = $cupc,
            v.retail_upc          = $rupc,
            v.eaches_upc          = $eupc,
            v.pkg_qty             = $pq,
            v.pkg_size            = $ps,
            v.pkg_unit            = $pu,
            v.pkg_container       = $pc,
            v.self_emb            = $emb,
            v._fingerprint        = $fp,
            v._last_ingested_run  = $run_id,
            v._last_ingested_at   = $ts
        """,
        pid=str(row["product_id"]),
        desc=row.get("product_description", ""),
        brand=row.get("brand", ""),
        supplier=row.get("supplier", ""),
        cls=row.get("product_class", ""),
        wh=row.get("warehouse", ""),
        upc=float(row.get("units_per_case", 0)),
        wt=float(row.get("unit_weight", 0)),
        cl=float(row.get("case_length", 0)),
        cw=float(row.get("case_width", 0)),
        ch=float(row.get("case_height", 0)),
        cupc=row.get("case_upc"),
        rupc=row.get("retail_upc"),
        eupc=row.get("eaches_upc"),
        pq=int(row.get("pkg_qty", 0)),
        ps=float(row.get("pkg_size", 0)),
        pu=row.get("pkg_unit", ""),
        pc=row.get("pkg_container", ""),
        emb=emb.tolist(),
        fp=fp,
        run_id=run_id,
        ts=datetime.now(timezone.utc).isoformat(),
    )


def execute_auto_match(session, row: dict, emb: np.ndarray,
                       match: Candidate, run_id: str) -> str:
    """
    Upsert the VendorSKU, create (or update) the MAPS_TO edge to the matched
    GlobalSKU. Store confidence and signals on the edge for traceability.
    Returns the matched GlobalSKU ID.
    """
    _upsert_vendor_sku(session, row, emb, run_id)

    session.run(
        """
        MATCH (v:VendorSKU  {product_id: $pid})
        MATCH (g:GlobalSKU  {sku_id:     $sid})
        MERGE (v)-[e:MAPS_TO]->(g)
        SET e.match_method   = $method,
            e.confidence     = $conf,
            e.signals        = $signals,
            e.matched_at     = $ts,
            e.ingestion_run  = $run_id
        """,
        pid=str(row["product_id"]),
        sid=match.global_sku_id,
        method="|".join(match.signals),
        conf=match.confidence,
        signals=match.signals,
        ts=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )
    return match.global_sku_id


def execute_review_queue(session, row: dict, emb: np.ndarray,
                         candidates: list[Candidate], run_id: str):
    """
    Upsert the VendorSKU. Write up to 3 MatchCandidate nodes (one per top
    candidate) so the analyst can pick the right one.
    """
    _upsert_vendor_sku(session, row, emb, run_id)
    pid  = str(row["product_id"])
    now  = datetime.now(timezone.utc).isoformat()

    for c in candidates[:3]:
        mc_id = str(uuid.uuid4())
        session.run(
            """
            MATCH (v:VendorSKU {product_id: $pid})
            MATCH (g:GlobalSKU {sku_id: $sid})
            CREATE (mc:MatchCandidate {
                mc_id:          $mc_id,
                vendor_sku_id:  $pid,
                global_sku_id:  $sid,
                confidence:     $conf,
                signals:        $signals,
                emb_similarity: $sim,
                global_brand:   $brand,
                global_category:$cat,
                status:         'PENDING',
                created_at:     $ts,
                run_id:         $run_id
            })
            MERGE (v)-[:HAS_CANDIDATE]->(mc)
            MERGE (mc)-[:CANDIDATE_FOR]->(g)
            """,
            pid=pid, sid=c.global_sku_id, mc_id=mc_id,
            conf=c.confidence, signals=c.signals, sim=c.emb_similarity,
            brand=c.global_brand, cat=c.global_category,
            ts=now, run_id=run_id,
        )


def execute_create_new(session, row: dict, emb: np.ndarray, run_id: str) -> str:
    """
    Upsert the VendorSKU and create a GlobalSKUDraft that an analyst can
    review and promote to a real GlobalSKU when ready.
    Returns the draft_id.
    """
    _upsert_vendor_sku(session, row, emb, run_id)
    draft_id = str(uuid.uuid4())
    session.run(
        """
        MATCH (v:VendorSKU {product_id: $pid})
        CREATE (d:GlobalSKUDraft {
            draft_id:            $did,
            source_vendor_sku_id: $pid,
            brand_family:        $brand,
            product_class:       $cls,
            units_per_case:      $upc,
            weight:              $wt,
            retail_upc:          $rupc,
            case_upc:            $cupc,
            product_description: $desc,
            supplier:            $supplier,
            self_emb:            $emb,
            status:              'DRAFT',
            created_at:          $ts,
            run_id:              $run_id
        })
        MERGE (v)-[:PROPOSED_AS]->(d)
        """,
        pid=str(row["product_id"]),
        did=draft_id,
        brand=row.get("brand", ""),
        cls=row.get("product_class", ""),
        upc=float(row.get("units_per_case", 0)),
        wt=float(row.get("unit_weight", 0)),
        rupc=row.get("retail_upc"),
        cupc=row.get("case_upc"),
        desc=row.get("product_description", ""),
        supplier=row.get("supplier", ""),
        emb=emb.tolist(),
        ts=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )
    return draft_id


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — POST-MERGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def post_merge_validation(session, affected_sku_ids: list[str]) -> list[dict]:
    """
    Re-run divergence-weighted attention reflection on every GlobalSKU that
    received a new or updated MAPS_TO edge in this run.

    Returns a list of alert dicts for SKUs whose anomaly_attn rose by ≥
    MATCH_ANOMALY_ALERT_DELTA — meaning the new mapping introduced a
    data-quality contradiction.
    """
    if not affected_sku_ids:
        return []

    _attn = importlib.import_module("03c_reflection_attention")
    compute_attention_reflection = _attn.compute_attention_reflection

    alerts = []
    print(f"  Validating {len(affected_sku_ids)} affected GlobalSKUs ...")

    for sku_id in tqdm(affected_sku_ids, desc="  post-merge", unit="SKU"):
        # Fetch previous anomaly_attn (may be None if first time)
        row = session.run(
            "MATCH (g:GlobalSKU {sku_id: $sid}) "
            "RETURN g.anomaly_attn AS prev",
            sid=sku_id,
        ).single()
        prev_score = row["prev"] if row and row["prev"] is not None else None

        # Recompute
        reflect, new_score, diag = compute_attention_reflection(
            sku_id, "GlobalSKU", "sku_id", session
        )
        if reflect is None:
            continue

        # Write updated embedding and score
        session.run(
            "MATCH (g:GlobalSKU {sku_id: $sid}) "
            "SET g.reflect_emb_attn = $emb, g.anomaly_attn = $score",
            sid=sku_id, emb=reflect, score=new_score,
        )

        if prev_score is not None:
            delta = new_score - prev_score
            if delta >= MATCH_ANOMALY_ALERT_DELTA:
                alerts.append({
                    "sku_id":     sku_id,
                    "prev_score": prev_score,
                    "new_score":  new_score,
                    "delta":      round(delta, 4),
                    "top_contributors": diag.get("top_neighbours", []),
                })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# REVIEW QUEUE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def show_review_queue(session):
    """Print all pending MatchCandidate items."""
    rows = session.run(
        """
        MATCH (v:VendorSKU)-[:HAS_CANDIDATE]->(mc:MatchCandidate {status:'PENDING'})
              -[:CANDIDATE_FOR]->(g:GlobalSKU)
        RETURN v.product_id    AS vendor_id,
               v.brand         AS vendor_brand,
               g.sku_id        AS global_id,
               g.brand_family  AS global_brand,
               mc.confidence   AS conf,
               mc.signals      AS signals,
               mc.created_at   AS created_at
        ORDER BY mc.confidence DESC
        """
    ).data()

    if not rows:
        print("  No pending review items.")
        return

    print(f"\n── Review Queue ({len(rows)} pending) ──────────────────────────────────")
    print(f"  {'Vendor ID':<14} {'Vendor Brand':<22} {'Global ID':<12} "
          f"{'Global Brand':<22} {'Conf':>6} Signals")
    print(f"  {'-'*14} {'-'*22} {'-'*12} {'-'*22} {'-'*6} {'-'*20}")
    for r in rows:
        signals = ", ".join(r["signals"]) if r["signals"] else "—"
        print(
            f"  {str(r['vendor_id']):<14} "
            f"{str(r['vendor_brand']):<22} "
            f"{str(r['global_id']):<12} "
            f"{str(r['global_brand']):<22} "
            f"{r['conf']:>6.3f} {signals}"
        )


def approve_match(session, vendor_sku_id: str):
    """
    Approve the highest-confidence pending candidate for a vendor SKU.
    Creates the MAPS_TO edge and marks all candidates for this vendor as APPROVED/REJECTED.
    """
    rows = session.run(
        """
        MATCH (v:VendorSKU {product_id: $pid})-[:HAS_CANDIDATE]->
              (mc:MatchCandidate {status:'PENDING'})-[:CANDIDATE_FOR]->(g:GlobalSKU)
        RETURN mc.mc_id AS mc_id, g.sku_id AS sid, mc.confidence AS conf,
               mc.signals AS signals
        ORDER BY conf DESC LIMIT 1
        """,
        pid=vendor_sku_id,
    ).data()

    if not rows:
        print(f"  No pending candidates for vendor SKU '{vendor_sku_id}'.")
        return

    best = rows[0]
    now  = datetime.now(timezone.utc).isoformat()

    session.run(
        """
        MATCH (v:VendorSKU  {product_id: $pid})
        MATCH (g:GlobalSKU  {sku_id:     $sid})
        MERGE (v)-[e:MAPS_TO]->(g)
        SET e.match_method  = $method,
            e.confidence    = $conf,
            e.signals       = $signals,
            e.matched_at    = $ts,
            e.approved_by   = 'analyst'
        """,
        pid=vendor_sku_id, sid=best["sid"],
        method="|".join(best["signals"] or []),
        conf=best["conf"], signals=best["signals"],
        ts=now,
    )

    # Mark all candidates for this vendor as resolved
    session.run(
        """
        MATCH (:VendorSKU {product_id: $pid})-[:HAS_CANDIDATE]->(mc:MatchCandidate)
        SET mc.status     = CASE WHEN mc.mc_id = $best_id THEN 'APPROVED' ELSE 'REJECTED' END,
            mc.resolved_at = $ts
        """,
        pid=vendor_sku_id, best_id=best["mc_id"], ts=now,
    )

    print(f"  Approved: VendorSKU '{vendor_sku_id}' → GlobalSKU '{best['sid']}' "
          f"(confidence={best['conf']:.3f})")


def reject_match(session, vendor_sku_id: str):
    """
    Reject all candidates for a vendor SKU and promote it to a GlobalSKUDraft.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Mark all candidates as rejected
    session.run(
        """
        MATCH (:VendorSKU {product_id: $pid})-[:HAS_CANDIDATE]->(mc:MatchCandidate)
        SET mc.status = 'REJECTED', mc.resolved_at = $ts
        """,
        pid=vendor_sku_id, ts=now,
    )

    # Create a GlobalSKUDraft if one doesn't already exist
    row = session.run(
        "MATCH (v:VendorSKU {product_id: $pid}) "
        "RETURN v.product_description AS desc, v.brand AS brand, "
        "       v.product_class AS cls, v.units_per_case AS upc, "
        "       v.unit_weight AS wt, v.retail_upc AS rupc, "
        "       v.case_upc AS cupc, v.supplier AS sup, v.self_emb AS emb",
        pid=vendor_sku_id,
    ).single()

    if not row:
        print(f"  VendorSKU '{vendor_sku_id}' not found.")
        return

    draft_id = str(uuid.uuid4())
    session.run(
        """
        MATCH (v:VendorSKU {product_id: $pid})
        MERGE (v)-[:PROPOSED_AS]->(d:GlobalSKUDraft {source_vendor_sku_id: $pid})
        SET d.draft_id            = $did,
            d.brand_family        = $brand,
            d.product_class       = $cls,
            d.units_per_case      = $upc,
            d.weight              = $wt,
            d.retail_upc          = $rupc,
            d.case_upc            = $cupc,
            d.product_description = $desc,
            d.supplier            = $sup,
            d.self_emb            = $emb,
            d.status              = 'DRAFT',
            d.created_at          = $ts
        """,
        pid=vendor_sku_id, did=draft_id,
        brand=row["brand"], cls=row["cls"],
        upc=row["upc"], wt=row["wt"],
        rupc=row["rupc"], cupc=row["cupc"],
        desc=row["desc"], sup=row["sup"],
        emb=row["emb"], ts=now,
    )
    print(f"  Rejected all candidates for '{vendor_sku_id}' → created GlobalSKUDraft {draft_id[:8]}…")


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION RUN RECORD
# ─────────────────────────────────────────────────────────────────────────────

def _write_run_record(session, run_id: str, vendor_file: str,
                      started_at: str, counts: dict):
    completed_at = datetime.now(timezone.utc).isoformat()
    session.run(
        """
        CREATE (:IngestionRun {
            run_id:         $run_id,
            vendor_file:    $vf,
            started_at:     $start,
            completed_at:   $end,
            n_total:        $total,
            n_new:          $new,
            n_unchanged:    $unch,
            n_updated:      $upd,
            n_auto_matched: $auto,
            n_review_queued:$review,
            n_create_new:   $create,
            n_alerts:       $alerts
        })
        """,
        run_id=run_id, vf=vendor_file,
        start=started_at, end=completed_at,
        total=counts.get("total", 0),
        new=counts.get("NEW", 0),
        unch=counts.get("UNCHANGED", 0),
        upd=counts.get("FIELD_UPDATE", 0) + counts.get("UPC_CONFLICT", 0),
        auto=counts.get("AUTO_MATCH", 0),
        review=counts.get("REVIEW_QUEUE", 0),
        create=counts.get("CREATE_NEW", 0),
        alerts=counts.get("alerts", 0),
    )


def show_last_report(session):
    """Print the most recent IngestionRun summary."""
    row = session.run(
        """
        MATCH (r:IngestionRun)
        RETURN r ORDER BY r.started_at DESC LIMIT 1
        """
    ).single()

    if not row:
        print("  No ingestion runs found.")
        return

    r = dict(row["r"])
    print(f"\n── Last Ingestion Run ────────────────────────────────────────────")
    print(f"  run_id      : {r.get('run_id', '—')}")
    print(f"  file        : {r.get('vendor_file', '—')}")
    print(f"  started     : {str(r.get('started_at', ''))[:19]}")
    print(f"  completed   : {str(r.get('completed_at', ''))[:19]}")
    print(f"  total rows  : {r.get('n_total', 0):,}")
    print(f"  unchanged   : {r.get('n_unchanged', 0):,}")
    print(f"  updated     : {r.get('n_updated', 0):,}")
    print(f"  AUTO_MATCH  : {r.get('n_auto_matched', 0):,}")
    print(f"  REVIEW_QUEUE: {r.get('n_review_queued', 0):,}")
    print(f"  CREATE_NEW  : {r.get('n_create_new', 0):,}")
    print(f"  alerts      : {r.get('n_alerts', 0):,}  (anomaly spike ≥ {MATCH_ANOMALY_ALERT_DELTA})")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_ingest_report(decisions: list[Decision], alerts: list[dict], vendor_file: str):
    counts = {k: 0 for k in ("AUTO_MATCH", "REVIEW_QUEUE", "CREATE_NEW", "UNCHANGED")}
    for d in decisions:
        counts[d.action] = counts.get(d.action, 0) + 1

    print(f"\n{'═'*64}")
    print(f"  INGESTION COMPLETE — {vendor_file}")
    print(f"{'═'*64}")
    print(f"  Total rows processed : {len(decisions):,}")
    print(f"  UNCHANGED            : {counts['UNCHANGED']:,}  (skipped)")
    print(f"  AUTO_MATCH           : {counts['AUTO_MATCH']:,}  (MAPS_TO edge created)")
    print(f"  REVIEW_QUEUE         : {counts['REVIEW_QUEUE']:,}  (MatchCandidate written)")
    print(f"  CREATE_NEW           : {counts['CREATE_NEW']:,}  (GlobalSKUDraft written)")

    if alerts:
        print(f"\n  ⚠  {len(alerts)} anomaly alerts — new mappings raised anomaly_attn ≥ "
              f"+{MATCH_ANOMALY_ALERT_DELTA}:")
        for a in alerts[:10]:
            print(f"    GlobalSKU {a['sku_id']:<12}  "
                  f"{a['prev_score']:.3f} → {a['new_score']:.3f}  "
                  f"(Δ +{a['delta']:.3f})")
            if a["top_contributors"]:
                top = a["top_contributors"][0]
                print(f"      driven by [{top['rel_type']}] "
                      f"attn={top['attention']:.3f} div={top['divergence']:.3f}")
    else:
        print(f"\n  No anomaly alerts — all new mappings are consistent with existing graph.")

    if counts["REVIEW_QUEUE"]:
        print(f"\n  Review pending items with:")
        print(f"    python ingest_vendor.py --review-queue")
        print(f"    python ingest_vendor.py --approve <product_id>")

    print(f"{'═'*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Incremental vendor SKU ingestion pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("vendor_file",    nargs="?", help="Vendor Excel file to ingest")
    group.add_argument("--review-queue", action="store_true", help="Show pending review items")
    group.add_argument("--approve",      metavar="PRODUCT_ID", help="Approve top match candidate")
    group.add_argument("--reject",       metavar="PRODUCT_ID", help="Reject all candidates → draft")
    group.add_argument("--report",       action="store_true",  help="Show last ingestion run summary")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip post-merge anomaly validation (faster for large files)")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        _ensure_schema(session)

        # ── Read-only modes ──────────────────────────────────────────────────
        if args.review_queue:
            show_review_queue(session)
            driver.close()
            return

        if args.approve:
            approve_match(session, args.approve)
            driver.close()
            return

        if args.reject:
            reject_match(session, args.reject)
            driver.close()
            return

        if args.report:
            show_last_report(session)
            driver.close()
            return

        # ── Ingestion run ────────────────────────────────────────────────────
        vendor_file  = args.vendor_file
        run_id       = str(uuid.uuid4())
        started_at   = datetime.now(timezone.utc).isoformat()

        print(f"\n── Ingesting: {vendor_file} ─────────────────────────────────────")
        print(f"  run_id = {run_id[:8]}…")

        # Step 1 — Normalize
        print("\n── Step 1: Normalizing ──────────────────────────────────────────")
        df = normalize(vendor_file)

        # Step 2 — Delta detection
        print("\n── Step 2: Delta detection ──────────────────────────────────────")
        classified, _ = detect_delta(session, df)

        # Step 3 — Embed new/updated rows
        print("\n── Step 3: Embedding new/updated rows ───────────────────────────")
        device = (
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()           else
            "cpu"
        )
        model      = SentenceTransformer(EMBEDDING_MODEL, device=device)
        embeddings = embed_rows(df, classified, model)
        print(f"  Embedded {len(embeddings):,} rows")

        # Steps 4+5 — Match and route
        print("\n── Steps 4–5: Matching and routing ──────────────────────────────")
        decisions:      list[Decision] = []
        affected_skus:  list[str]      = []
        counts = {"total": len(df), "AUTO_MATCH": 0, "REVIEW_QUEUE": 0,
                  "CREATE_NEW": 0, "UNCHANGED": 0}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="  routing", unit="row"):
            pid    = str(row["product_id"])
            status = classified.get(pid, "NEW")
            rd     = row.to_dict()

            # UNCHANGED rows: update node properties only (no matching needed)
            if status == "UNCHANGED":
                counts["UNCHANGED"] += 1
                decisions.append(Decision(pid, "UNCHANGED", delta_status="UNCHANGED"))
                continue

            emb = embeddings.get(pid)
            if emb is None:
                counts["UNCHANGED"] += 1
                decisions.append(Decision(pid, "UNCHANGED", delta_status=status))
                continue

            candidates = find_candidates(session, rd, emb)
            dec        = route(candidates, status)
            dec.product_id = pid

            if dec.action == "AUTO_MATCH":
                global_id = execute_auto_match(session, rd, emb, dec.best, run_id)
                affected_skus.append(global_id)
                counts["AUTO_MATCH"] += 1

            elif dec.action == "REVIEW_QUEUE":
                execute_review_queue(session, rd, emb, dec.all_candidates, run_id)
                counts["REVIEW_QUEUE"] += 1

            else:  # CREATE_NEW
                execute_create_new(session, rd, emb, run_id)
                counts["CREATE_NEW"] += 1

            decisions.append(dec)

        # Step 6 — Post-merge validation
        alerts: list[dict] = []
        if not args.skip_validation and affected_skus:
            print("\n── Step 6: Post-merge anomaly validation ────────────────────────")
            alerts = post_merge_validation(session, list(set(affected_skus)))
            counts["alerts"] = len(alerts)
            if alerts:
                print(f"  ⚠  {len(alerts)} GlobalSKUs flagged")
            else:
                print(f"  ✓  No anomaly spikes detected")

        # Write run record
        _write_run_record(session, run_id, vendor_file, started_at, counts)

    driver.close()
    print_ingest_report(decisions, alerts, vendor_file)


if __name__ == "__main__":
    main()
