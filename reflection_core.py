"""
reflection_core.py — Shared reflection vector computation (attention + edge severity).

Default production path: divergence-weighted softmax attention over neighbours,
with REL_WEIGHTS and EDGE_SEVERITY multipliers. Writes to reflect_emb on nodes.
"""

from __future__ import annotations

import numpy as np
from neo4j import GraphDatabase

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    REL_WEIGHTS,
    EDGE_SEVERITY,
    ATTN_TEMPERATURE,
)

PK = {
    "GlobalSKU": "sku_id",
    "TenantSKU": "tenant_sku_id",
    "Brand": "brand_id",
    "PackageType": "package_type_id",
    "Manufacturer": "name",
    "Supplier": "name",
    "ProductClass": "name",
    "Customer": "customer_id",
    "TenantSKU": "tenant_sku_id",
    "TrainingImage": "image_id",
    "MergeEvent": "merge_id",
    "Pallet": "pallet_id",
}


def pk(label: str) -> str:
    return PK.get(label, "name")


def cosine_similarity(a, b) -> float:
    va = np.asarray(a, np.float32)
    vb = np.asarray(b, np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 1e-8 else 0.0


def anomaly_score(self_emb, reflect_emb) -> float:
    return round(1.0 - cosine_similarity(self_emb, reflect_emb), 4)


def _severity_mult(outcome: str, status: str, rollback_avail, match_method: str) -> float:
    mult = 1.0
    if outcome == "failure":
        mult *= EDGE_SEVERITY["outcome:failure"]
    if status == "conflicted":
        mult *= EDGE_SEVERITY["status:conflicted"]
    if rollback_avail is False:
        mult *= EDGE_SEVERITY["rollback_available:False"]
    if match_method == "fuzzy":
        mult *= EDGE_SEVERITY["match_method:fuzzy"]
    return mult


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / max(temperature, 1e-6)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def _fetch_neighbours(session, label: str, pk: str, entity_id: str) -> list[dict]:
    return session.run(
        f"""
        MATCH (e:{label} {{{pk}: $eid}})-[r]-(n)
        WHERE n.self_emb IS NOT NULL
        RETURN type(r) AS rel_type,
               n.self_emb AS emb,
               COALESCE(n.outcome, '') AS outcome,
               COALESCE(n.status, '') AS status,
               n.rollback_available AS rollback_available,
               COALESCE(n.match_method, '') AS match_method
        """,
        eid=entity_id,
    ).data()


def compute_reflect_emb(
    session,
    entity_id: str,
    label: str,
    temperature: float = ATTN_TEMPERATURE,
) -> list[float] | None:
    """
    Attention-weighted reflection: REL_WEIGHTS × edge_severity × neighbour divergence.
    """
    pk_field = pk(label)
    row = session.run(
        f"MATCH (e:{label} {{{pk_field}: $eid}}) RETURN e.self_emb AS emb",
        eid=entity_id,
    ).single()
    if not row or row["emb"] is None:
        return None

    entity_emb = np.array(row["emb"], dtype=np.float32)
    norm_e = np.linalg.norm(entity_emb)
    if norm_e < 1e-8:
        return None
    entity_n = entity_emb / norm_e

    rows = _fetch_neighbours(session, label, pk_field, entity_id)
    if not rows:
        return None

    raw_scores, neighbour_embs = [], []
    for r in rows:
        if r["emb"] is None:
            continue
        n_emb = np.array(r["emb"], dtype=np.float32)
        n_norm = np.linalg.norm(n_emb)
        if n_norm < 1e-8:
            continue
        n_unit = n_emb / n_norm
        divergence = 1.0 - float(np.dot(entity_n, n_unit))
        base_w = REL_WEIGHTS.get(r["rel_type"], REL_WEIGHTS["_DEFAULT"])
        sev = _severity_mult(
            r["outcome"], r["status"], r["rollback_available"], r["match_method"]
        )
        raw_scores.append(base_w * sev * divergence)
        neighbour_embs.append(n_emb)

    if not raw_scores:
        return None

    attention = _softmax(np.array(raw_scores, dtype=np.float32), temperature)
    reflect = attention @ np.stack(neighbour_embs)
    norm_r = np.linalg.norm(reflect)
    if norm_r < 1e-8:
        return None
    return (reflect / norm_r).tolist()


def write_reflect_emb(session, label: str, entity_id: str, reflect: list[float]):
    pk_field = pk(label)
    session.run(
        f"MATCH (n:{label} {{{pk_field}: $eid}}) SET n.reflect_emb = $emb",
        eid=entity_id,
        emb=reflect,
    )


def batch_compute_label(session, label: str, temperature: float = ATTN_TEMPERATURE) -> int:
    pk_field = pk(label)
    ids = [
        r["id"]
        for r in session.run(
            f"MATCH (n:{label}) WHERE n.self_emb IS NOT NULL RETURN n.{pk_field} AS id"
        )
    ]
    updated = 0
    for eid in ids:
        reflect = compute_reflect_emb(session, eid, label, temperature)
        if reflect is not None:
            write_reflect_emb(session, label, eid, reflect)
            updated += 1
    return updated


def recompute_cohort_skus(session, sku_ids: list[str], batch: int = 200) -> int:
    """Recompute reflect_emb for a list of GlobalSKU ids."""
    updated = 0
    for i in range(0, len(sku_ids), batch):
        for sid in sku_ids[i : i + batch]:
            reflect = compute_reflect_emb(session, sid, "GlobalSKU")
            if reflect is not None:
                write_reflect_emb(session, "GlobalSKU", sid, reflect)
                updated += 1
    return updated
