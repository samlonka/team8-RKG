"""
08_rgcn.py — Phase 2: R-GCN relational encoder

Replaces the hand-crafted scalar REL_WEIGHTS with learned per-relation weight
matrices. Each relation type r gets its own matrix W_r (implemented via basis
decomposition to share parameters). After training, the R-GCN output for each
node is its neighbourhood-contextualized embedding — the learned equivalent of
reflect_emb.

Architecture (2-layer R-GCN with basis decomposition):
  Input:  self_emb (EMBEDDING_DIM)
  Layer 1: RGCNConv(EMBEDDING_DIM → RGCN_HIDDEN_DIM, num_bases=RGCN_NUM_BASES) + ReLU + Dropout
  Layer 2: RGCNConv(RGCN_HIDDEN_DIM → RGCN_OUT_DIM, num_bases=RGCN_NUM_BASES)
  Output: reflect_emb_rgcn (RGCN_OUT_DIM = EMBEDDING_DIM, for direct cosine comparison)

Training: self-supervised link prediction (reconstruct held-out edges).
  Decoder: inner product  score(u,v) = z_u · z_v

Writes per node:
  reflect_emb_rgcn — RGCN_OUT_DIM-dim learned neighbourhood embedding
  anomaly_rgcn     — 1 - cosine(self_emb, reflect_emb_rgcn)

Requirements:
    pip install torch_geometric

Usage:
    python 08_rgcn.py [--epochs 50] [--label GlobalSKU]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from neo4j import GraphDatabase

try:
    from torch_geometric.nn import RGCNConv
    from torch_geometric.utils import negative_sampling
except ImportError as e:
    raise SystemExit(
        "torch_geometric is required for Phase 2. "
        "Install it with: pip install torch_geometric"
    ) from e

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBEDDING_DIM,
    RGCN_HIDDEN_DIM, RGCN_OUT_DIM, RGCN_NUM_BASES,
    RGCN_EPOCHS, RGCN_LR, RGCN_DROPOUT,
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
# GRAPH DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_graph(session) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Pull all nodes with self_emb and all edges from Neo4j.

    Returns:
        node_id_map  : {(label, entity_id): global_int_idx}
        x            : (N, EMBEDDING_DIM) float tensor — node features
        edge_index   : (2, E) long tensor — source / target global indices
        edge_type    : (E,)  long tensor  — relation type index per edge
        rel2int      : {rel_type_str: int}
    """
    print("  Fetching nodes ...")
    node_rows = []
    for label in ALL_LABELS:
        pk = PK_MAP[label]
        try:
            rows = session.run(
                f"MATCH (n:{label}) WHERE n.self_emb IS NOT NULL "
                f"RETURN n.{pk} AS id, n.self_emb AS emb"
            ).data()
            for r in rows:
                node_rows.append((label, r["id"], r["emb"]))
        except Exception:
            pass  # label may not exist in this graph instance

    # Build global node index
    node_id_map: dict[tuple, int] = {}
    feat_list: list[list[float]] = []
    for label, eid, emb in node_rows:
        key = (label, eid)
        if key not in node_id_map:
            node_id_map[key] = len(node_id_map)
            feat_list.append(emb)

    if not feat_list:
        raise RuntimeError("No nodes with self_emb found. Run 02_seed_data.py first.")

    x = torch.tensor(feat_list, dtype=torch.float32)
    print(f"    {x.shape[0]:,} nodes, {x.shape[1]}-dim features")

    # ── reverse lookup: global_idx → (label, entity_id) ──
    idx2key: dict[int, tuple] = {v: k for k, v in node_id_map.items()}

    print("  Fetching edges ...")
    # Cypher to pull all edges between any pair of known labels
    label_list = "|".join(ALL_LABELS)
    edge_rows = session.run(
        f"""
        MATCH (a)-[r]->(b)
        WHERE a.self_emb IS NOT NULL AND b.self_emb IS NOT NULL
        RETURN labels(a)[0] AS al, id(a) AS aid_neo,
               labels(b)[0] AS bl, id(b) AS bid_neo,
               type(r) AS rel
        LIMIT 500000
        """
    ).data()

    # We need entity property IDs, not Neo4j internal IDs — re-query with pk
    # More efficient: fetch (label, pk_value, label, pk_value, rel_type) directly
    edge_rows2 = []
    for label in ALL_LABELS:
        pk = PK_MAP[label]
        try:
            rows = session.run(
                f"""
                MATCH (a:{label})-[r]->(b)
                WHERE a.self_emb IS NOT NULL AND b.self_emb IS NOT NULL
                RETURN $lbl AS al, a.{pk} AS aid, labels(b)[0] AS bl, type(r) AS rel,
                       CASE labels(b)[0]
                         WHEN 'GlobalSKU'    THEN b.sku_id
                         WHEN 'TenantSKU'    THEN b.tenant_sku_id
                         WHEN 'Brand'        THEN b.brand_id
                         WHEN 'PackageType'  THEN b.package_type_id
                         WHEN 'Customer'     THEN b.customer_id
                         WHEN 'TenantSKU'    THEN b.tenant_sku_id
                         WHEN 'TrainingImage'THEN b.image_id
                         WHEN 'MergeEvent'   THEN b.merge_id
                         WHEN 'Pallet'       THEN b.pallet_id
                         ELSE b.name
                       END AS bid
                """,
                lbl=label,
            ).data()
            edge_rows2.extend(rows)
        except Exception:
            pass

    rel2int: dict[str, int] = {}
    src_list: list[int] = []
    dst_list: list[int] = []
    etype_list: list[int] = []

    for row in edge_rows2:
        src_key = (row["al"], row["aid"])
        dst_key = (row["bl"], row["bid"])
        if src_key not in node_id_map or dst_key not in node_id_map:
            continue
        if row["rel"] not in rel2int:
            rel2int[row["rel"]] = len(rel2int)
        src_list.append(node_id_map[src_key])
        dst_list.append(node_id_map[dst_key])
        etype_list.append(rel2int[row["rel"]])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type  = torch.tensor(etype_list, dtype=torch.long)
    print(f"    {edge_index.shape[1]:,} edges | {len(rel2int)} relation types")
    return node_id_map, x, edge_index, edge_type, rel2int


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RGCNEncoder(nn.Module):
    """
    2-layer R-GCN encoder with basis decomposition.
    Output dimension matches EMBEDDING_DIM for direct cosine comparison with self_emb.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_relations: int, num_bases: int, dropout: float):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index, edge_type))
        h = self.drop(h)
        return self.conv2(h, edge_index, edge_type)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Inner-product link prediction score."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(model: RGCNEncoder, x: torch.Tensor,
          edge_index: torch.Tensor, edge_type: torch.Tensor,
          epochs: int, lr: float, device: torch.device) -> RGCNEncoder:

    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_type  = edge_type.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_nodes = x.size(0)

    print(f"\n  Training R-GCN ({epochs} epochs, device={device}) ...")
    for epoch in tqdm(range(1, epochs + 1), desc="  R-GCN", unit="ep"):
        model.train()
        optimizer.zero_grad()

        z = model(x, edge_index, edge_type)

        neg_edge = negative_sampling(
            edge_index, num_nodes=num_nodes,
            num_neg_samples=edge_index.size(1),
            method="sparse",
        ).to(device)

        pos_score = model.decode(z, edge_index)
        neg_score = model.decode(z, neg_edge)

        loss = (
            -F.logsigmoid(pos_score).mean()
            - F.logsigmoid(-neg_score).mean()
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(f"    epoch {epoch:3d}  loss={loss.item():.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING EXTRACTION & WRITE-BACK
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(model: RGCNEncoder, x: torch.Tensor,
                       edge_index: torch.Tensor, edge_type: torch.Tensor,
                       device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model(x.to(device), edge_index.to(device), edge_type.to(device))
        z = F.normalize(z, p=2, dim=-1)   # L2-normalize for cosine comparison
    return z.cpu().numpy()


def write_back(session, node_id_map: dict, x_np: np.ndarray, z_np: np.ndarray,
               target_labels: list[str]):
    """
    Write reflect_emb_rgcn and anomaly_rgcn for target_labels.
    Groups writes by label for batch efficiency.
    """
    by_label: dict[str, list[dict]] = defaultdict(list)

    for (label, eid), idx in node_id_map.items():
        if label not in target_labels:
            continue
        se  = x_np[idx]
        re  = z_np[idx]
        cos = float(np.dot(se, re) / (np.linalg.norm(se) * np.linalg.norm(re) + 1e-8))
        by_label[label].append({
            "eid":   eid,
            "emb":   re.tolist(),
            "score": round(1.0 - cos, 4),
        })

    for label, rows in by_label.items():
        pk = PK_MAP[label]
        batch_size = 200
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            session.run(
                f"""
                UNWIND $rows AS r
                MATCH (n:{label} {{{pk}: r.eid}})
                SET n.reflect_emb_rgcn = r.emb,
                    n.anomaly_rgcn     = r.score
                """,
                rows=chunk,
            )
        print(f"    {label}: {len(rows):,} nodes written")


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_top_anomalies(session, label: str, top_n: int):
    pk = PK_MAP.get(label, "name")
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.anomaly_rgcn IS NOT NULL
        RETURN n.{pk} AS id, n.anomaly_rgcn AS score
        ORDER BY score DESC LIMIT $n
        """,
        n=top_n,
    ).data()
    if not rows:
        return
    print(f"\n── Phase 2 top anomalies: {label} {'─' * 30}")
    print(f"  {'ID':<18} {'Score':<8} Risk")
    print(f"  {'-'*18} {'-'*8} {'-'*6}")
    for r in rows:
        risk = (
            "HIGH"   if r["score"] >= ANOMALY_HIGH_RISK   else
            "MEDIUM" if r["score"] >= ANOMALY_MEDIUM_RISK else
            "LOW"
        )
        print(f"  {str(r['id']):<18} {r['score']:<8.4f} {risk}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: R-GCN reflection encoder")
    parser.add_argument("--epochs", type=int, default=RGCN_EPOCHS)
    parser.add_argument("--lr",     type=float, default=RGCN_LR)
    parser.add_argument("--label",  default="ALL",
                        help="Labels to write results for (default: ALL base labels)")
    parser.add_argument("--top",    type=int, default=20)
    args = parser.parse_args()

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )

    write_labels = (
        ["GlobalSKU", "TenantSKU", "Brand", "PackageType",
         "Manufacturer", "Supplier", "ProductClass"]
        if args.label == "ALL" else [args.label]
    )

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # Vector index for ANN queries in agent pipeline
        session.run(f"""
            CREATE VECTOR INDEX idx_global_sku_reflect_rgcn IF NOT EXISTS
            FOR (n:GlobalSKU) ON n.reflect_emb_rgcn
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {RGCN_OUT_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)

        print("\n── Phase 2: Loading graph data ──────────────────────────────────")
        node_id_map, x, edge_index, edge_type, rel2int = fetch_graph(session)

        num_relations = len(rel2int)
        model = RGCNEncoder(
            in_dim=EMBEDDING_DIM,
            hidden_dim=RGCN_HIDDEN_DIM,
            out_dim=RGCN_OUT_DIM,
            num_relations=num_relations,
            num_bases=min(RGCN_NUM_BASES, num_relations),
            dropout=RGCN_DROPOUT,
        )
        print(f"  R-GCN: {EMBEDDING_DIM}→{RGCN_HIDDEN_DIM}→{RGCN_OUT_DIM} | "
              f"{num_relations} relations | {RGCN_NUM_BASES} bases")

        model = train(model, x, edge_index, edge_type, args.epochs, args.lr, device)

        print("\n  Extracting embeddings ...")
        x_np = x.numpy()
        z_np = extract_embeddings(model, x, edge_index, edge_type, device)

        print("\n── Phase 2: Writing reflect_emb_rgcn + anomaly_rgcn ─────────────")
        write_back(session, node_id_map, x_np, z_np, write_labels)

        print("\n── Phase 2: Top anomalies (anomaly_rgcn) ────────────────────────")
        for label in write_labels:
            print_top_anomalies(session, label, args.top)

    driver.close()
    print("\nPhase 2 complete.\n")


if __name__ == "__main__":
    main()
