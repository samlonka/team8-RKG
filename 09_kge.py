"""
09_kge.py — Phase 3: KGE triple scoring with RotatE

Unlike Phases 1, 2, and 4 which produce node-level anomaly scores, RotatE
produces edge-level scores: each individual triple (head, relation, tail) gets
a plausibility score. This makes anomalies actionable — the analyst sees
"this specific BELONGS_TO_BRAND edge is implausible" rather than just
"this node is anomalous."

RotatE (Sun et al., 2019) models relations as rotations in complex space:
    score(h, r, t) = -||h ∘ r - t||²
Higher score (less negative) = more plausible triple.

After training:
  - Each relationship in Neo4j gets  triple_score  (normalised [0,1], higher = more normal)
  - Each node gets  triple_anomaly_score = 1 - min(triple_score of its edges)
    (the most implausible edge determines the node's triple-level vulnerability)

Requirements:
    pip install pykeen

Usage:
    python 09_kge.py [--epochs 100]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from neo4j import GraphDatabase

try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
except ImportError as e:
    raise SystemExit(
        "pykeen is required for Phase 3. Install with: pip install pykeen"
    ) from e

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    KGE_MODEL, KGE_EMBEDDING_DIM, KGE_EPOCHS, KGE_LR,
    ANOMALY_HIGH_RISK, ANOMALY_MEDIUM_RISK,
)

ALL_LABELS = [
    "GlobalSKU", "TenantSKU", "Brand", "PackageType",
    "Manufacturer", "Supplier", "ProductClass",
    "Customer", "TenantSKU", "TrainingImage", "MergeEvent", "Pallet",
]

PK_MAP = {
    "GlobalSKU":     "sku_id",
    "TenantSKU":     "tenant_sku_id",
    "Brand":         "brand_id",
    "PackageType":   "package_type_id",
    "Manufacturer":  "name",
    "Supplier":      "name",
    "ProductClass":  "name",
    "Customer":      "customer_id",
    "TenantSKU":     "tenant_sku_id",
    "TrainingImage": "image_id",
    "MergeEvent":    "merge_id",
    "Pallet":        "pallet_id",
}


# ─────────────────────────────────────────────────────────────────────────────
# TRIPLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_triples(session) -> list[tuple[str, str, str]]:
    """
    Pull all edges as (head_entity_key, relation, tail_entity_key) string triples.
    Entity keys are '<Label>:<id>' strings for global uniqueness across node types.
    """
    triples: list[tuple[str, str, str]] = []

    for label in ALL_LABELS:
        pk = PK_MAP[label]
        try:
            rows = session.run(
                f"""
                MATCH (a:{label})-[r]->(b)
                WHERE a.self_emb IS NOT NULL AND b.self_emb IS NOT NULL
                RETURN $lbl + ':' + toString(a.{pk}) AS head,
                       type(r) AS rel,
                       labels(b)[0] + ':' + toString(
                         CASE labels(b)[0]
                           WHEN 'GlobalSKU'     THEN b.sku_id
                           WHEN 'TenantSKU'     THEN b.tenant_sku_id
                           WHEN 'Brand'         THEN b.brand_id
                           WHEN 'PackageType'   THEN b.package_type_id
                           WHEN 'Customer'      THEN b.customer_id
                           WHEN 'TenantSKU'     THEN b.tenant_sku_id
                           WHEN 'TrainingImage' THEN b.image_id
                           WHEN 'MergeEvent'    THEN b.merge_id
                           WHEN 'Pallet'        THEN b.pallet_id
                           ELSE b.name
                         END
                       ) AS tail
                """,
                lbl=label,
            ).data()
            triples.extend((r["head"], r["rel"], r["tail"]) for r in rows)
        except Exception:
            pass

    print(f"  {len(triples):,} triples across {len({t[1] for t in triples})} relation types")
    return triples


# ─────────────────────────────────────────────────────────────────────────────
# KGE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_kge(triples: list[tuple[str, str, str]], epochs: int, lr: float):
    """Train RotatE via PyKEEN pipeline and return (model, full TriplesFactory).

    PyKEEN ≥1.11 requires both training and testing factories — passing only
    training is no longer accepted.  We split 80/20, train on the larger split,
    and score all triples from the full factory afterwards.
    """
    triple_array = np.array(triples, dtype=str)
    tf = TriplesFactory.from_labeled_triples(
        triples=triple_array,
        create_inverse_triples=True,   # adds reverse relations for better coverage
    )
    training, testing = tf.split([0.8, 0.2], random_state=42)

    print(f"\n  Training {KGE_MODEL} ...")
    print(f"    entities={tf.num_entities:,}  relations={tf.num_relations}  "
          f"triples={tf.num_triples:,}  (train={training.num_triples:,} / "
          f"test={testing.num_triples:,})")

    result = pipeline(
        training=training,
        testing=testing,
        model=KGE_MODEL,
        model_kwargs=dict(embedding_dim=KGE_EMBEDDING_DIM),
        training_loop="slcwa",        # stochastic local closed-world assumption
        training_kwargs=dict(num_epochs=epochs),
        optimizer="adam",
        optimizer_kwargs=dict(lr=lr),
        random_seed=42,
        use_tqdm=True,
    )
    return result.model, tf


# ─────────────────────────────────────────────────────────────────────────────
# SCORING & WRITE-BACK
# ─────────────────────────────────────────────────────────────────────────────

def score_and_write(session, model, tf: TriplesFactory,
                    triples: list[tuple[str, str, str]]):
    """
    Score all triples, normalise to [0,1], then write:
      - triple_score on each edge  (1 = very plausible / normal)
      - triple_anomaly_score on each node (1 - min triple_score of adjacent edges)
    """
    triple_array = np.array(triples, dtype=str)

    # Map labeled triples to integer IDs used by the model
    mapped = tf.map_triples(triple_array)   # (T, 3) int64

    model.eval()
    with torch.no_grad():
        raw_scores = model.score_hrt(
            torch.tensor(mapped, dtype=torch.long)
        ).squeeze(-1).cpu().numpy()

    # RotatE raw score = -distance² → more negative = worse.
    # Normalise so that 1.0 = most plausible, 0.0 = least plausible.
    lo, hi = raw_scores.min(), raw_scores.max()
    span = hi - lo if hi > lo else 1.0
    norm_scores = (raw_scores - lo) / span   # [0, 1], higher = more normal

    print(f"\n  Scored {len(norm_scores):,} triples — "
          f"min={norm_scores.min():.3f} max={norm_scores.max():.3f} "
          f"mean={norm_scores.mean():.3f}")

    # Per-node mean triple score (mean-aggregation over all adjacent edges).
    # min-aggregation was tried but amplifies random low-scoring edges on healthy
    # nodes, raising their triple_anomaly_score unfairly (healthy=0.793 vs 0.793).
    # Mean-aggregation gives a smoother, more discriminative signal.
    node_sum:   dict[str, float] = {}
    node_count: dict[str, int]   = {}

    edge_updates: list[dict] = []
    for i, (head, rel, tail) in enumerate(triples):
        score = float(norm_scores[i])
        edge_updates.append({"head": head, "rel": rel, "tail": tail, "score": score})
        node_sum[head]   = node_sum.get(head, 0.0)   + score
        node_count[head] = node_count.get(head, 0)   + 1
        node_sum[tail]   = node_sum.get(tail, 0.0)   + score
        node_count[tail] = node_count.get(tail, 0)   + 1

    # ── Write triple_score to edges (batched by relation type) ───────────────
    print("  Writing triple_score to edges ...")
    by_rel: dict[str, list] = defaultdict(list)
    for upd in edge_updates:
        # Only write for actual (non-inverse) relation types
        if not upd["rel"].endswith("_inverse"):
            by_rel[upd["rel"]].append(upd)

    for rel, rows in tqdm(by_rel.items(), desc="  edges", unit="rel"):
        # We don't know head/tail labels at this point, so we use the entity key
        # format 'Label:id' to do generic Cypher matching.
        for row in rows:
            head_parts = row["head"].split(":", 1)
            tail_parts = row["tail"].split(":", 1)
            if len(head_parts) < 2 or len(tail_parts) < 2:
                continue
            h_label, h_id = head_parts[0], head_parts[1]
            t_label, t_id = tail_parts[0], tail_parts[1]
            h_pk = PK_MAP.get(h_label, "name")
            t_pk = PK_MAP.get(t_label, "name")
            try:
                session.run(
                    f"""
                    MATCH (a:{h_label} {{{h_pk}: $hid}})-[r:{rel}]->(b:{t_label} {{{t_pk}: $tid}})
                    SET r.triple_score = $score
                    """,
                    hid=h_id, tid=t_id, score=row["score"],
                )
            except Exception:
                pass

    # ── Write triple_anomaly_score to nodes ──────────────────────────────────
    print("  Writing triple_anomaly_score to nodes ...")
    by_label: dict[str, list] = defaultdict(list)
    for entity_key in node_sum:
        parts = entity_key.split(":", 1)
        if len(parts) < 2:
            continue
        label, eid = parts[0], parts[1]
        mean_score = node_sum[entity_key] / max(node_count[entity_key], 1)
        by_label[label].append({
            "eid":   eid,
            "score": round(1.0 - mean_score, 4),  # invert: higher = more anomalous
        })

    for label, rows in by_label.items():
        pk = PK_MAP.get(label, "name")
        batch = 200
        for i in range(0, len(rows), batch):
            chunk = rows[i : i + batch]
            session.run(
                f"""
                UNWIND $rows AS r
                MATCH (n:{label} {{{pk}: r.eid}})
                SET n.triple_anomaly_score = r.score
                """,
                rows=chunk,
            )
        print(f"    {label}: {len(rows):,} nodes")


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_top_anomalies(session, label: str, top_n: int):
    pk = PK_MAP.get(label, "name")
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.triple_anomaly_score IS NOT NULL
        RETURN n.{pk} AS id, n.triple_anomaly_score AS score
        ORDER BY score DESC LIMIT $n
        """,
        n=top_n,
    ).data()
    if not rows:
        return
    print(f"\n── Phase 3 top anomalies: {label} {'─' * 30}")
    print(f"  {'ID':<18} {'Score':<8} Risk")
    print(f"  {'-'*18} {'-'*8} {'-'*6}")
    for r in rows:
        risk = (
            "HIGH"   if r["score"] >= ANOMALY_HIGH_RISK   else
            "MEDIUM" if r["score"] >= ANOMALY_MEDIUM_RISK else
            "LOW"
        )
        print(f"  {str(r['id']):<18} {r['score']:<8.4f} {risk}")

    # Also show the most anomalous edges (triple_score < 0.3)
    edge_rows = session.run(
        f"""
        MATCH (a:{label})-[r]->(b)
        WHERE r.triple_score IS NOT NULL AND r.triple_score < 0.3
        RETURN a.{pk} AS head, type(r) AS rel, labels(b)[0] AS tail_label,
               r.triple_score AS score
        ORDER BY score ASC LIMIT 10
        """
    ).data()
    if edge_rows:
        print(f"\n  Low-plausibility edges for {label} (triple_score < 0.30):")
        for r in edge_rows:
            print(f"    {str(r['head']):<18} -[{r['rel']}]→ [{r['tail_label']}]  "
                  f"triple_score={r['score']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: KGE triple scoring with RotatE")
    parser.add_argument("--epochs", type=int, default=KGE_EPOCHS)
    parser.add_argument("--top",    type=int, default=20)
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("\n── Phase 3: Building triples ────────────────────────────────────")
        triples = fetch_triples(session)
        if not triples:
            print("  No triples found. Run 02_seed_data.py + 05_synthesize_lifecycle.py first.")
            return

        model, tf = train_kge(triples, epochs=args.epochs, lr=KGE_LR)

        print("\n── Phase 3: Scoring triples + writing to Neo4j ─────────────────")
        score_and_write(session, model, tf, triples)

        print("\n── Phase 3: Top anomalies (triple_anomaly_score) ────────────────")
        for label in ["GlobalSKU", "TenantSKU", "Brand"]:
            print_top_anomalies(session, label, args.top)

    driver.close()
    print("\nPhase 3 complete.\n")


if __name__ == "__main__":
    main()
