"""
10_dominant.py — Phase 4: DOMINANT joint attribute + structure reconstruction

DOMINANT (Ding et al., 2019) is the only method in this pipeline that detects
STRUCTURAL anomalies — nodes whose connectivity pattern is inconsistent with
what their embedding would predict. Phases 1–3 are purely semantic.

Model:
  GCN encoder:  self_emb (768) → hidden (256) → bottleneck (64)
  Attr decoder: bottleneck (64) → 768  (reconstructs self_emb)
  Struct decoder: inner product  Z @ Z.T  (reconstructs adjacency)

Anomaly score per node:
  attr_err[i]   = ||self_emb[i] - reconstructed_attr[i]||²  (per-node MSE)
  struct_err[i] = BCE over sampled positive + negative edges adjacent to node i
  dominant_score[i] = alpha * norm(attr_err[i]) + (1 - alpha) * norm(struct_err[i])

The structural term catches topology-based anomalies invisible to embedding
distance methods — e.g. a shared_sku that spans 5 customers is structurally
unusual even if all its embeddings are coherent.

Full N×N adjacency reconstruction is avoided by computing the structural loss
over sampled edges (positive + equal-count random negatives) — O(E) memory.

Writes per node:
  dominant_score — [0,1], higher = more anomalous (blended attr + struct)
  dominant_attr_score  — attribute component only
  dominant_struct_score — structural component only

Requirements:
    pip install torch  (already in requirements.txt)

Usage:
    python 10_dominant.py [--label GlobalSKU] [--epochs 100] [--alpha 0.5]
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

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBEDDING_DIM,
    DOMINANT_ALPHA, DOMINANT_HIDDEN_DIM, DOMINANT_BOTTLENECK,
    DOMINANT_EPOCHS, DOMINANT_LR,
    ANOMALY_HIGH_RISK, ANOMALY_MEDIUM_RISK,
)

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

ALL_LABELS = list(PK_MAP.keys())


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_subgraph(session, focus_label: str):
    """
    Load the subgraph centred on focus_label:
      - all nodes of focus_label that have self_emb
      - all direct neighbours (any label) with self_emb
      - all edges incident to focus_label nodes

    Returns:
        node_idx   : {(label, eid): int}
        x          : (N, EMBEDDING_DIM) float tensor
        pos_edges  : (2, E) long tensor — directed edges (src, dst)
        focus_mask : boolean mask of shape (N,) — True for focus_label nodes
    """
    pk = PK_MAP[focus_label]

    # Focus nodes
    focus_rows = session.run(
        f"MATCH (n:{focus_label}) WHERE n.self_emb IS NOT NULL "
        f"RETURN n.{pk} AS id, n.self_emb AS emb"
    ).data()
    print(f"  focus={focus_label}: {len(focus_rows):,} nodes")

    node_idx: dict[tuple, int] = {}
    feat_list: list[list[float]] = []

    for r in focus_rows:
        key = (focus_label, r["id"])
        if key not in node_idx:
            node_idx[key] = len(node_idx)
            feat_list.append(r["emb"])

    # Neighbour nodes (all labels)
    neighbour_rows = session.run(
        f"""
        MATCH (f:{focus_label})-[r]-(n)
        WHERE f.self_emb IS NOT NULL AND n.self_emb IS NOT NULL
        RETURN labels(n)[0] AS lbl,
               CASE labels(n)[0]
                 WHEN 'GlobalSKU'     THEN n.sku_id
                 WHEN 'TenantSKU'     THEN n.tenant_sku_id
                 WHEN 'Brand'         THEN n.brand_id
                 WHEN 'PackageType'   THEN n.package_type_id
                 WHEN 'Customer'      THEN n.customer_id
                 WHEN 'TenantSKU'     THEN n.tenant_sku_id
                 WHEN 'TrainingImage' THEN n.image_id
                 WHEN 'MergeEvent'    THEN n.merge_id
                 WHEN 'Pallet'        THEN n.pallet_id
                 ELSE n.name
               END AS nid,
               n.self_emb AS emb
        """
    ).data()

    for r in neighbour_rows:
        key = (r["lbl"], r["nid"])
        if key not in node_idx:
            node_idx[key] = len(node_idx)
            feat_list.append(r["emb"])

    x = torch.tensor(feat_list, dtype=torch.float32)
    focus_mask = torch.zeros(len(node_idx), dtype=torch.bool)
    for (lbl, _), idx in node_idx.items():
        if lbl == focus_label:
            focus_mask[idx] = True

    # Edges incident to focus nodes
    edge_rows = session.run(
        f"""
        MATCH (f:{focus_label})-[r]->(n)
        WHERE f.self_emb IS NOT NULL AND n.self_emb IS NOT NULL
        RETURN f.{pk} AS fid, labels(n)[0] AS nl,
               CASE labels(n)[0]
                 WHEN 'GlobalSKU'     THEN n.sku_id
                 WHEN 'TenantSKU'     THEN n.tenant_sku_id
                 WHEN 'Brand'         THEN n.brand_id
                 WHEN 'PackageType'   THEN n.package_type_id
                 WHEN 'Customer'      THEN n.customer_id
                 WHEN 'TenantSKU'     THEN n.tenant_sku_id
                 WHEN 'TrainingImage' THEN n.image_id
                 WHEN 'MergeEvent'    THEN n.merge_id
                 WHEN 'Pallet'        THEN n.pallet_id
                 ELSE n.name
               END AS nid
        UNION
        MATCH (n)-[r]->(f:{focus_label})
        WHERE f.self_emb IS NOT NULL AND n.self_emb IS NOT NULL
        RETURN f.{pk} AS fid, labels(n)[0] AS nl,
               CASE labels(n)[0]
                 WHEN 'GlobalSKU'     THEN n.sku_id
                 WHEN 'TenantSKU'     THEN n.tenant_sku_id
                 WHEN 'Brand'         THEN n.brand_id
                 WHEN 'PackageType'   THEN n.package_type_id
                 WHEN 'Customer'      THEN n.customer_id
                 WHEN 'TenantSKU'     THEN n.tenant_sku_id
                 WHEN 'TrainingImage' THEN n.image_id
                 WHEN 'MergeEvent'    THEN n.merge_id
                 WHEN 'Pallet'        THEN n.pallet_id
                 ELSE n.name
               END AS nid
        """
    ).data()

    srcs, dsts = [], []
    for r in edge_rows:
        fkey = (focus_label, r["fid"])
        nkey = (r["nl"], r["nid"])
        if fkey in node_idx and nkey in node_idx:
            srcs.append(node_idx[fkey])
            dsts.append(node_idx[nkey])
            # Undirected: add both directions for message passing
            srcs.append(node_idx[nkey])
            dsts.append(node_idx[fkey])

    pos_edges = torch.tensor([srcs, dsts], dtype=torch.long)
    print(f"  total nodes in subgraph: {len(node_idx):,}  edges: {pos_edges.shape[1]:,}")
    return node_idx, x, pos_edges, focus_mask


def build_norm_adj(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalised adjacency with self-loops: D^{-1/2} (A+I) D^{-1/2}
    Stored as dense tensor — acceptable for subgraph sizes up to ~20k nodes.
    """
    N = num_nodes
    adj = torch.zeros(N, N)
    # Self-loops
    adj.fill_diagonal_(1.0)
    # Edges (undirected)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0

    deg = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)
    return D @ adj @ D


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.linear(adj @ x)


class DOMINANTModel(nn.Module):
    """
    2-layer GCN encoder, attribute decoder, structure decoder.
    The bottleneck dimension keeps the structure decoder Z @ Z.T tractable.
    """

    def __init__(self, in_dim: int, hidden_dim: int, bottleneck: int):
        super().__init__()
        # Encoder
        self.enc1 = GCNLayer(in_dim, hidden_dim)
        self.enc2 = GCNLayer(hidden_dim, bottleneck)
        # Attribute decoder: bottleneck → original feature dim
        self.attr_dec = nn.Linear(bottleneck, in_dim)

    def encode(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc1(adj, x))
        return self.enc2(adj, h)     # (N, bottleneck)

    def decode_attr(self, z: torch.Tensor) -> torch.Tensor:
        return self.attr_dec(z)      # (N, in_dim)

    def decode_struct(self, z: torch.Tensor,
                      pos_ei: torch.Tensor,
                      neg_ei: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner product over sampled edges only — avoids full N×N matrix."""
        pos_scores = torch.sigmoid(
            (z[pos_ei[0]] * z[pos_ei[1]]).sum(-1)
        )
        neg_scores = torch.sigmoid(
            (z[neg_ei[0]] * z[neg_ei[1]]).sum(-1)
        )
        return pos_scores, neg_scores

    def forward(self, adj, x, pos_ei, neg_ei):
        z = self.encode(adj, x)
        x_hat = self.decode_attr(z)
        pos_s, neg_s = self.decode_struct(z, pos_ei, neg_ei)
        return z, x_hat, pos_s, neg_s


def random_negative_edges(pos_ei: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Sample the same number of random non-edges as positive edges."""
    n_neg = pos_ei.shape[1]
    pos_set = set(zip(pos_ei[0].tolist(), pos_ei[1].tolist()))

    srcs, dsts = [], []
    while len(srcs) < n_neg:
        batch_s = torch.randint(0, num_nodes, (n_neg * 2,)).tolist()
        batch_d = torch.randint(0, num_nodes, (n_neg * 2,)).tolist()
        for s, d in zip(batch_s, batch_d):
            if s != d and (s, d) not in pos_set and len(srcs) < n_neg:
                srcs.append(s)
                dsts.append(d)

    return torch.tensor([srcs[:n_neg], dsts[:n_neg]], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_dominant(model: DOMINANTModel, x: torch.Tensor,
                   adj: torch.Tensor, pos_ei: torch.Tensor,
                   epochs: int, lr: float, alpha: float,
                   device: torch.device) -> DOMINANTModel:

    model = model.to(device)
    x   = x.to(device)
    adj = adj.to(device)
    pos_ei = pos_ei.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_nodes = x.size(0)

    print(f"\n  Training DOMINANT ({epochs} epochs, N={num_nodes:,}, device={device}) ...")
    for epoch in tqdm(range(1, epochs + 1), desc="  DOMINANT", unit="ep"):
        model.train()
        optimizer.zero_grad()

        neg_ei = random_negative_edges(pos_ei.cpu(), num_nodes).to(device)
        _, x_hat, pos_s, neg_s = model(adj, x, pos_ei, neg_ei)

        attr_loss   = F.mse_loss(x_hat, x)
        struct_loss = (
            F.binary_cross_entropy(pos_s, torch.ones_like(pos_s))
            + F.binary_cross_entropy(neg_s, torch.zeros_like(neg_s))
        )
        loss = alpha * attr_loss + (1.0 - alpha) * struct_loss
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            tqdm.write(
                f"    epoch {epoch:3d}  loss={loss.item():.4f}  "
                f"attr={attr_loss.item():.4f}  struct={struct_loss.item():.4f}"
            )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_scores(model: DOMINANTModel, x: torch.Tensor,
                   adj: torch.Tensor, pos_ei: torch.Tensor,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-node attribute error and structural error.
    Returns (attr_err, struct_err) as numpy arrays of shape (N,).
    """
    model.eval()
    x_d   = x.to(device)
    adj_d = adj.to(device)

    z     = model.encode(adj_d, x_d)
    x_hat = model.decode_attr(z)

    # Attribute error: per-node MSE
    attr_err = F.mse_loss(x_hat, x_d, reduction="none").mean(-1).cpu().numpy()

    # Structural error: for each node, score over its positive edges + equal negatives
    N = x.size(0)
    struct_err = np.zeros(N, dtype=np.float32)
    count      = np.zeros(N, dtype=np.int32)

    # Positive edges: expected adjacency = 1, anomaly = (1 - predicted_score)
    if pos_ei.shape[1] > 0:
        pos_ei_d = pos_ei.to(device)
        pos_scores = torch.sigmoid(
            (z[pos_ei_d[0]] * z[pos_ei_d[1]]).sum(-1)
        ).cpu().numpy()
        for k in range(pos_ei.shape[1]):
            s, d = pos_ei[0, k].item(), pos_ei[1, k].item()
            err = 1.0 - pos_scores[k]        # expected 1, deviation = 1 - score
            struct_err[s] += err
            count[s]      += 1

    # Random negatives: expected adjacency = 0, anomaly = predicted_score
    neg_ei = random_negative_edges(pos_ei.cpu(), N)
    neg_ei_d = neg_ei.to(device)
    neg_scores = torch.sigmoid(
        (z[neg_ei_d[0]] * z[neg_ei_d[1]]).sum(-1)
    ).cpu().numpy()
    for k in range(neg_ei.shape[1]):
        s = neg_ei[0, k].item()
        struct_err[s] += float(neg_scores[k])   # expected 0, deviation = score
        count[s]      += 1

    # Normalise by edge count per node
    mask = count > 0
    struct_err[mask] /= count[mask]

    return attr_err, struct_err


def blend_scores(attr_err: np.ndarray, struct_err: np.ndarray,
                 alpha: float) -> np.ndarray:
    """Min-max normalise each component then blend."""
    def _norm(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-8)

    return alpha * _norm(attr_err) + (1.0 - alpha) * _norm(struct_err)


# ─────────────────────────────────────────────────────────────────────────────
# WRITE-BACK
# ─────────────────────────────────────────────────────────────────────────────

def write_scores(session, focus_label: str, node_idx: dict,
                 focus_mask: torch.Tensor,
                 attr_err: np.ndarray, struct_err: np.ndarray,
                 dominant: np.ndarray):
    pk = PK_MAP[focus_label]
    rows = []
    for (lbl, eid), idx in node_idx.items():
        if lbl != focus_label or not focus_mask[idx]:
            continue
        rows.append({
            "eid":    eid,
            "dom":    round(float(dominant[idx]), 4),
            "attr":   round(float(attr_err[idx]),   4),
            "struct": round(float(struct_err[idx]),  4),
        })

    batch = 200
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        session.run(
            f"""
            UNWIND $rows AS r
            MATCH (n:{focus_label} {{{pk}: r.eid}})
            SET n.dominant_score        = r.dom,
                n.dominant_attr_score   = r.attr,
                n.dominant_struct_score = r.struct
            """,
            rows=chunk,
        )
    print(f"  {focus_label}: {len(rows):,} nodes written")


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_top_anomalies(session, label: str, top_n: int):
    pk = PK_MAP.get(label, "name")
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.dominant_score IS NOT NULL
        RETURN n.{pk} AS id,
               n.dominant_score        AS dom,
               n.dominant_attr_score   AS attr,
               n.dominant_struct_score AS strct
        ORDER BY dom DESC LIMIT $n
        """,
        n=top_n,
    ).data()
    if not rows:
        return
    print(f"\n── Phase 4 top anomalies: {label} {'─' * 30}")
    print(f"  {'ID':<18} {'DOMINANT':<10} {'Attr':<8} {'Struct':<8} Risk")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
    for r in rows:
        risk = (
            "HIGH"   if r["dom"] >= ANOMALY_HIGH_RISK   else
            "MEDIUM" if r["dom"] >= ANOMALY_MEDIUM_RISK else
            "LOW"
        )
        print(f"  {str(r['id']):<18} {r['dom']:<10.4f} "
              f"{r['attr']:<8.4f} {r['strct']:<8.4f} {risk}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

BASE_LABELS = ["GlobalSKU", "TenantSKU", "Brand", "PackageType",
               "Manufacturer", "Supplier", "ProductClass"]


def main():
    parser = argparse.ArgumentParser(description="Phase 4: DOMINANT anomaly detection")
    parser.add_argument("--label",  default="ALL",
                        help="Focus label to run DOMINANT on (default: ALL base labels)")
    parser.add_argument("--epochs", type=int, default=DOMINANT_EPOCHS)
    parser.add_argument("--alpha",  type=float, default=DOMINANT_ALPHA,
                        help="Blend weight: 0=structure-only, 1=attribute-only")
    parser.add_argument("--top",    type=int, default=20)
    args = parser.parse_args()

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )

    labels = BASE_LABELS if args.label == "ALL" else [args.label]
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for label in labels:
            print(f"\n── Phase 4: DOMINANT on {label} {'─' * 40}")
            node_idx, x, pos_ei, focus_mask = fetch_subgraph(session, label)

            N = x.size(0)
            if N == 0:
                print(f"  Skipping {label} — no nodes with self_emb")
                continue

            # Dense adjacency is feasible up to ~15k nodes; warn above that
            if N > 15_000:
                print(f"  WARNING: {N:,} nodes in subgraph — adj matrix {N*N*4/1e6:.0f}MB. "
                      "Consider running with a smaller cohort.")

            adj = build_norm_adj(N, pos_ei)

            model = DOMINANTModel(
                in_dim=EMBEDDING_DIM,
                hidden_dim=DOMINANT_HIDDEN_DIM,
                bottleneck=DOMINANT_BOTTLENECK,
            )
            model = train_dominant(
                model, x, adj, pos_ei,
                epochs=args.epochs, lr=DOMINANT_LR,
                alpha=args.alpha, device=device,
            )

            attr_err, struct_err = compute_scores(model, x, adj, pos_ei, device)
            dominant = blend_scores(attr_err, struct_err, args.alpha)

            write_scores(session, label, node_idx, focus_mask,
                         attr_err, struct_err, dominant)

        print("\n── Phase 4: Top anomalies (dominant_score) ───────────────────────")
        for label in labels:
            print_top_anomalies(session, label, args.top)

    driver.close()
    print("\nPhase 4 complete.\n")


if __name__ == "__main__":
    main()
