"""
03c_reflection_attention.py — Divergence-weighted soft attention reflection

The baseline (03_reflection.py) scales each neighbour by a fixed scalar
REL_WEIGHTS[rel_type] and takes the mean.  Every MERGED_INTO neighbour gets
weight 3.0 regardless of whether it actually contradicts the central entity.

This script replaces that scalar with a per-neighbour attention weight that is
proportional to how much the neighbour diverges from the entity:

    raw_score_i  =  REL_WEIGHTS[rel_type_i]
                    ×  (1 − cosine(entity_emb, neighbour_emb_i))

    attention_i  =  softmax( raw_scores / τ )      # τ = ATTN_TEMPERATURE

    reflect_emb_attn  =  Σ  attention_i × neighbour_emb_i   (L2-normalised)

Why this produces better anomaly signal:

  Healthy entity   — all neighbours are semantically coherent, divergence
                     scores are small, softmax is nearly uniform, reflect_emb
                     stays close to self_emb  →  low anomaly_attn score.

  Anomalous entity — one or two neighbours contradict the entity (wrong brand,
                     conflicted merge event, scan-failure pallet).  Those
                     neighbours get the highest divergence scores, dominate
                     after softmax, and pull reflect_emb_attn away from
                     self_emb  →  high anomaly_attn score.

No training.  No new dependencies.  Pure softmax over existing embeddings.

Temperature τ (ATTN_TEMPERATURE in config.py):
  τ = 1.0  standard softmax, good default
  τ < 1.0  sharpens attention — most-divergent neighbour dominates
  τ > 1.0  softens attention — approaches uniform weighted mean

Writes per node:
  reflect_emb_attn  — 768-dim attention-weighted reflection vector
  anomaly_attn      — 1 − cosine(self_emb, reflect_emb_attn)

Usage:
    python 03c_reflection_attention.py [--label GlobalSKU] [--top 20]
    python 03c_reflection_attention.py --temp 0.5   # sharper attention
    python 03c_reflection_attention.py --scores-only
"""

import argparse
import uuid

import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    REL_WEIGHTS, ATTN_TEMPERATURE,
    ANOMALY_HIGH_RISK, ANOMALY_MEDIUM_RISK, EMBEDDING_DIM,
)
from score_log import log_batch_run, ensure_indexes as ensure_score_log_indexes

ALL_LABELS = [
    "GlobalSKU", "VendorSKU", "Brand", "PackageType",
    "Manufacturer", "Supplier", "ProductClass",
]


def _pk(label: str) -> str:
    return {
        "GlobalSKU":     "sku_id",
        "VendorSKU":     "product_id",
        "Brand":         "brand_id",
        "PackageType":   "package_type_id",
        "Manufacturer":  "name",
        "Supplier":      "name",
        "ProductClass":  "name",
    }.get(label, "name")


# ─────────────────────────────────────────────────────────────────────────────
# CORE ATTENTION COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    # Subtract max for numerical stability before exp.
    z = x / temperature
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def compute_attention_reflection(
    entity_id: str,
    label: str,
    pk: str,
    session,
    temperature: float = 1.0,
) -> tuple[list[float] | None, float | None, dict]:
    """
    Return (reflect_emb_attn, anomaly_attn, diagnostics).

    diagnostics contains per-neighbour attention weights for interpretability —
    useful for showing judges exactly which neighbour drove the anomaly.
    """
    # ── Fetch the entity's own embedding ────────────────────────────────────
    row = session.run(
        f"MATCH (e:{label} {{{pk}: $eid}}) RETURN e.self_emb AS emb",
        eid=entity_id,
    ).single()
    if not row or row["emb"] is None:
        return None, None, {}

    entity_emb = np.array(row["emb"], dtype=np.float32)
    entity_norm = np.linalg.norm(entity_emb)
    if entity_norm < 1e-8:
        return None, None, {}
    entity_emb_n = entity_emb / entity_norm  # unit vector for cosine

    # ── Fetch all neighbours ─────────────────────────────────────────────────
    rows = session.run(
        f"""
        MATCH (e:{label} {{{pk}: $eid}})-[r]-(n)
        WHERE n.self_emb IS NOT NULL
        RETURN type(r) AS rel_type, n.self_emb AS emb
        """,
        eid=entity_id,
    ).data()

    if not rows:
        return None, None, {}

    # ── Compute raw attention scores ─────────────────────────────────────────
    #
    # raw_score_i = REL_WEIGHTS[rel_type] × (1 − cosine(entity, neighbour))
    #
    # The (1 − cosine) term is divergence in [0, 2].  Multiplying by
    # REL_WEIGHTS preserves the domain signal that MERGED_INTO neighbours
    # should matter more than USED_BY ones, while the divergence factor
    # ensures only the *actually contradicting* neighbours get high attention.
    raw_scores = []
    neighbour_embs = []
    rel_types = []

    for r in rows:
        if r["emb"] is None:
            continue
        n_emb = np.array(r["emb"], dtype=np.float32)
        rel   = r["rel_type"]

        cos_sim   = _cosine(entity_emb_n, n_emb / (np.linalg.norm(n_emb) + 1e-8))
        divergence = 1.0 - cos_sim   # [0, 2]; higher = more surprising

        base_w    = REL_WEIGHTS.get(rel, REL_WEIGHTS["_DEFAULT"])
        raw_score = base_w * divergence

        raw_scores.append(raw_score)
        neighbour_embs.append(n_emb)
        rel_types.append(rel)

    if not raw_scores:
        return None, None, {}

    raw_arr = np.array(raw_scores, dtype=np.float32)

    # ── Softmax normalisation ────────────────────────────────────────────────
    attention = _softmax(raw_arr, temperature)   # sums to 1

    # ── Attention-weighted sum of neighbour embeddings ───────────────────────
    emb_matrix = np.stack(neighbour_embs)        # (N, 768)
    reflect    = attention @ emb_matrix           # (768,)

    norm = np.linalg.norm(reflect)
    if norm < 1e-8:
        return None, None, {}

    reflect_n = reflect / norm
    anomaly   = round(1.0 - _cosine(entity_emb_n, reflect_n), 4)

    # ── Diagnostics: top-3 neighbours by attention weight ───────────────────
    top_idx = np.argsort(attention)[::-1][:3]
    diagnostics = {
        "top_neighbours": [
            {
                "rel_type":   rel_types[i],
                "attention":  round(float(attention[i]), 4),
                "divergence": round(float(1.0 - _cosine(
                    entity_emb_n,
                    neighbour_embs[i] / (np.linalg.norm(neighbour_embs[i]) + 1e-8)
                )), 4),
                "raw_score":  round(float(raw_scores[i]), 4),
            }
            for i in top_idx
        ],
        "n_neighbours":  len(raw_scores),
        "attention_entropy": round(float(-np.sum(attention * np.log(attention + 1e-12))), 4),
    }

    return reflect_n.tolist(), anomaly, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTE
# ─────────────────────────────────────────────────────────────────────────────

def batch_compute(session, label: str, temperature: float) -> int:
    pk  = _pk(label)
    ids = [
        r["id"] for r in session.run(
            f"MATCH (n:{label}) WHERE n.self_emb IS NOT NULL RETURN n.{pk} AS id"
        )
    ]
    print(f"  {label}: {len(ids):,} nodes  (τ={temperature})")

    updated = skipped = 0
    for eid in tqdm(ids, desc=f"  {label}", unit="node"):
        reflect, anomaly, _ = compute_attention_reflection(
            eid, label, pk, session, temperature
        )
        if reflect is None:
            skipped += 1
            continue

        session.run(
            f"""
            MATCH (n:{label} {{{pk}: $eid}})
            SET n.reflect_emb_attn = $emb,
                n.anomaly_attn     = $score
            """,
            eid=eid, emb=reflect, score=anomaly,
        )
        updated += 1

    print(f"    → {updated:,} updated | {skipped:,} skipped (no neighbours)")
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def rank_anomalies(session, label: str, top_n: int) -> list[dict]:
    pk   = _pk(label)
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.anomaly_attn IS NOT NULL
        RETURN n.{pk} AS id, n.anomaly_attn AS score
        ORDER BY score DESC LIMIT $n
        """,
        n=top_n,
    ).data()
    return [
        {
            "id":   r["id"],
            "score": r["score"],
            "risk": (
                "HIGH"   if r["score"] >= ANOMALY_HIGH_RISK   else
                "MEDIUM" if r["score"] >= ANOMALY_MEDIUM_RISK else
                "LOW"
            ),
        }
        for r in rows
    ]


def print_table(label: str, ranked: list[dict], temperature: float):
    print(f"\n── Attention anomalies: {label}  (τ={temperature}) {'─'*25}")
    print(f"  {'ID':<18} {'Score':<8} Risk")
    print(f"  {'-'*18} {'-'*8} {'-'*6}")
    for r in ranked:
        print(f"  {str(r['id']):<18} {r['score']:<8.4f} {r['risk']}")


def explain_entity(session, entity_id: str, label: str, temperature: float):
    """
    Print a human-readable attention breakdown for a single entity.
    Shows which neighbours drove the anomaly score and by how much.
    """
    pk = _pk(label)
    _, anomaly, diag = compute_attention_reflection(
        entity_id, label, pk, session, temperature
    )
    if anomaly is None:
        print(f"  {entity_id}: no neighbours with embeddings found")
        return

    print(f"\n── Attention explanation: {entity_id} ({'─'*35})")
    print(f"  anomaly_attn = {anomaly:.4f}   "
          f"n_neighbours = {diag['n_neighbours']}   "
          f"entropy = {diag['attention_entropy']:.3f}")
    print(f"\n  Top attention contributors:")
    print(f"  {'Rel type':<22} {'Attention':>9} {'Divergence':>10} {'Raw score':>9}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*9}")
    for nb in diag["top_neighbours"]:
        print(
            f"  {nb['rel_type']:<22} "
            f"{nb['attention']:>9.4f} "
            f"{nb['divergence']:>10.4f} "
            f"{nb['raw_score']:>9.4f}"
        )
    print(
        f"\n  Entropy note: {diag['attention_entropy']:.3f}  "
        f"{'(focused — few neighbours dominate)' if diag['attention_entropy'] < 1.0 else '(diffuse — attention spread across neighbours)'}"
    )


def compare_with_baseline(session, label: str, top_n: int = 10):
    """
    Side-by-side: anomaly_baseline vs anomaly_attn for the top entities.
    Highlights where attention finds anomalies the baseline misses (or ranks higher).
    """
    pk   = _pk(label)
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.anomaly_attn IS NOT NULL AND n.anomaly_baseline IS NOT NULL
        RETURN n.{pk} AS id,
               n.anomaly_baseline AS base,
               n.anomaly_attn     AS attn
        ORDER BY attn DESC LIMIT $n
        """,
        n=top_n,
    ).data()

    if not rows:
        print(f"  No entities with both scores for {label}.")
        return

    print(f"\n── Baseline vs Attention: {label} (top {top_n} by attention) {'─'*15}")
    print(f"  {'ID':<18} {'Baseline':>9} {'Attention':>9} {'Delta':>8}")
    print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*8}")
    for r in rows:
        delta = r["attn"] - r["base"]
        flag  = " ← attention finds more" if delta > 0.05 else ""
        print(
            f"  {str(r['id']):<18} "
            f"{r['base']:>9.4f} "
            f"{r['attn']:>9.4f} "
            f"{delta:>+8.4f}{flag}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Divergence-weighted soft attention reflection (no training)"
    )
    parser.add_argument("--label",       default="ALL",
                        help="Node label to process (default: ALL base labels)")
    parser.add_argument("--top",         type=int, default=20)
    parser.add_argument("--temp",        type=float, default=ATTN_TEMPERATURE,
                        help=f"Softmax temperature (default: {ATTN_TEMPERATURE})")
    parser.add_argument("--scores-only", action="store_true",
                        help="Skip recomputation, just print current anomaly_attn scores")
    parser.add_argument("--explain",     metavar="ENTITY_ID",
                        help="Print attention breakdown for a single entity")
    parser.add_argument("--compare",     action="store_true",
                        help="Show side-by-side baseline vs attention scores")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    labels = ALL_LABELS if args.label == "ALL" else [args.label]

    with driver.session() as session:
        ensure_score_log_indexes(session)

        # Vector index so the agent pipeline can do ANN on reflect_emb_attn
        session.run(f"""
            CREATE VECTOR INDEX idx_global_sku_reflect_attn IF NOT EXISTS
            FOR (n:GlobalSKU) ON n.reflect_emb_attn
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)

        # ── Explain mode ─────────────────────────────────────────────────────
        if args.explain:
            label = labels[0] if len(labels) == 1 else "GlobalSKU"
            explain_entity(session, args.explain, label, args.temp)
            driver.close()
            return

        # ── Compute ───────────────────────────────────────────────────────────
        if not args.scores_only:
            print(f"\n── Computing reflect_emb_attn  (τ={args.temp}) ─────────────────")
            for label in labels:
                batch_compute(session, label, args.temp)

        # ── Report ────────────────────────────────────────────────────────────
        print(f"\n── Top anomalies (anomaly_attn, τ={args.temp}) ─────────────────────")
        for label in labels:
            ranked = rank_anomalies(session, label, args.top)
            if ranked:
                print_table(label, ranked, args.temp)

        if args.compare:
            print("\n── Baseline vs Attention comparison ─────────────────────────────")
            for label in labels:
                compare_with_baseline(session, label, top_n=args.top)

        # ── Score log ─────────────────────────────────────────────────────────
        print("\n── Appending to score log ───────────────────────────────────────")
        run_id = str(uuid.uuid4())
        for label in labels:
            _, n = log_batch_run(session, label, "attention", run_id=run_id)
            if n:
                print(f"  {label}: {n:,} log entries  run={run_id[:8]}…")

    driver.close()
    print("\nAttention reflection complete.\n")


if __name__ == "__main__":
    main()
