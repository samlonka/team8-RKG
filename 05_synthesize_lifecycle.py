"""
05_synthesize_lifecycle.py — Add the SKU-lifecycle layer the handbook requires.

The base graph (02_seed_data) only has GlobalSKU/Brand/PackageType/... It is
missing the entities that actually carry the anomaly signal:
  Customer, TenantSKU, TrainingImage, MergeEvent, Pallet
and the high-weight relationships MERGED_INTO (3.0), SCANNED_ON (2.5),
TRAINED_WITH (2.0), MAPS_TO (2.0), USED_BY (1.0).

This script:
  1. Picks a cohort of real GlobalSKUs (a "new customer onboarding" set).
  2. Gives healthy SKUs consistent neighbours (aligned training images, success
     scans) so their reflect_emb stays close to self_emb.
  3. Plants the handbook's anomaly types on specific SKUs with DIVERGENT
     high-weight neighbours (conflicted merge events, scan failures, wrong-brand
     records, wrong-category tenant mappings, cross-customer sharing).
  4. Generates self_emb for every new node (same encoder/dim as the base graph).
  5. Recomputes and writes reflect_emb for every cohort SKU.
  6. Writes seed_manifest.json (planted anomalies + analyst ground-truth labels)
     — the basis for the seeded-anomaly evaluation (06_evaluate.py).

Usage:
    python 05_synthesize_lifecycle.py [--cohort 300]
"""

from __future__ import annotations

import json
import random
import argparse
import numpy as np
import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBEDDING_MODEL, EMBED_BATCH_SIZE, REL_WEIGHTS,
)
from reflection_core import recompute_cohort_skus

RNG = random.Random(42)
MANIFEST_PATH = "seed_manifest.json"
COHORT_TAG = "ACME_ONBOARDING"

CUSTOMERS = ["ACME FOODS", "BLUE RIDGE DIST", "CITYWIDE BEV", "DELTA SUPPLY", "EVERGREEN WHSE"]


# ─────────────────────────────────────────────────────────────────────────────
# TEXT BUILDERS — control alignment (healthy) vs divergence (anomalous)
# ─────────────────────────────────────────────────────────────────────────────

def t_customer(name):           return f"customer {name} warehouse distributor vor enabled"
def t_tenant_ok(brand, pkg, c): return f"tenant sku {brand} {pkg} mapped exact match customer {c}"
def t_tenant_wrong(c):          return (f"tenant sku frozen vegetables produce dairy aisle wrong product "
                                        f"category mismatched mapping customer {c}")
def t_img_ok(brand, pkg):       return f"training image labeled {brand} {pkg} clear high quality verified"
def t_pallet_ok(brand, pkg):    return f"pallet scan success detected {brand} {pkg} high confidence verified"
def t_pallet_fail():            return ("pallet scan failure misidentified wrong product low confidence "
                                        "error reject defect mismatch unrecognised")
def t_merge_conflict():         return ("merge event conflicted history contradictory upc rollback unavailable "
                                        "three source records duplicate dispute data quality failure")
def t_brand_wrong():            return "brand generic import unverified newly created duplicate unmatched record"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH WRITE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(session, cypher, rows, size=500):
    for i in range(0, len(rows), size):
        session.run(cypher, {"rows": rows[i:i + size]})


def write_nodes(session, label, id_field, rows):
    cypher = f"""
    UNWIND $rows AS r
    MERGE (n:{label} {{{id_field}: r.id}})
    SET n.self_emb = r.self_emb, n += r.props
    """
    run_batch(session, cypher, rows)


def write_edges(session, rel, a_label, a_field, b_label, b_field, rows):
    cypher = f"""
    UNWIND $rows AS r
    MATCH (a:{a_label} {{{a_field}: r.a}})
    MATCH (b:{b_label} {{{b_field}: r.b}})
    MERGE (a)-[e:{rel}]->(b)
    SET e += r.props
    """
    run_batch(session, cypher, rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SYNTHESIS
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=int, default=300)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading encoder on {device} ...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    def embed(texts):
        return model.encode(texts, batch_size=EMBED_BATCH_SIZE,
                            show_progress_bar=False, normalize_embeddings=True)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        # indexes for the new labels (id lookups during edge writes)
        for lbl, fld in [("Customer", "customer_id"), ("TenantSKU", "tenant_sku_id"),
                         ("TrainingImage", "image_id"), ("MergeEvent", "merge_id"),
                         ("Pallet", "pallet_id")]:
            s.run(f"CREATE INDEX idx_{lbl.lower()}_id IF NOT EXISTS FOR (n:{lbl}) ON (n.{fld})")

        # ── pick cohort: real-brand GlobalSKUs with a self_emb ──────────────
        rows = s.run(
            """
            MATCH (g:GlobalSKU)
            WHERE g.self_emb IS NOT NULL AND g.brand_family IS NOT NULL
                  AND g.brand_family <> 'UNKNOWN' AND g.brand_family <> 'nan'
            RETURN g.sku_id AS sku, g.brand_family AS brand,
                   g.package_category_name AS pkg
            ORDER BY g.sku_id LIMIT $n
            """, n=args.cohort).data()
        if len(rows) < args.cohort:
            print(f"  WARN only {len(rows)} cohort SKUs available")
        for r in rows:
            r["pkg"] = r["pkg"] or "unknown"
        print(f"  cohort SKUs: {len(rows)}")

        # tag cohort
        run_batch(s, "UNWIND $rows AS r MATCH (g:GlobalSKU {sku_id:r.sku}) SET g.cohort=$tag".replace(
            "$tag", f"'{COHORT_TAG}'"), [{"sku": r["sku"]} for r in rows])

        # ── assign roles ────────────────────────────────────────────────────
        RNG.shuffle(rows)
        idx = 0
        def grab(n):
            nonlocal idx
            chunk = rows[idx:idx + n]; idx += n; return chunk
        brand_mismatch = grab(8)
        merge_conflict = grab(8)
        evidence_gap   = grab(8)
        shared_sku     = grab(6)
        auto_map       = grab(4)
        healthy        = rows[idx:]
        print(f"  planted: brand_mismatch={len(brand_mismatch)} merge_conflict={len(merge_conflict)} "
              f"evidence_gap={len(evidence_gap)} shared={len(shared_sku)} auto_map={len(auto_map)} "
              f"healthy={len(healthy)}")

        # ── Customers ───────────────────────────────────────────────────────
        cust_rows = []
        cust_texts = [t_customer(c) for c in CUSTOMERS]
        cust_emb = embed(cust_texts)
        for i, c in enumerate(CUSTOMERS):
            cust_rows.append({"id": c, "self_emb": cust_emb[i].tolist(),
                              "props": {"name": c}})
        write_nodes(s, "Customer", "customer_id", cust_rows)

        # accumulators
        tnodes, tedges_map, tedges_use = [], [], []   # TenantSKU
        inodes, iedges = [], []                        # TrainingImage
        mnodes, medges = [], []                        # MergeEvent
        pnodes, pedges = [], []                        # Pallet
        ttexts, itexts, mtexts, ptexts = [], [], [], []

        def add_tenant(sku, brand, pkg, cust, wrong=False):
            tid = f"T{len(tnodes)+1:05d}"
            txt = t_tenant_wrong(cust) if wrong else t_tenant_ok(brand, pkg, cust)
            tnodes.append({"id": tid, "props": {"customer": cust,
                          "match_method": "fuzzy" if wrong else "exact"}})
            ttexts.append(txt)
            tedges_map.append({"a": tid, "b": sku, "props": {}})
            tedges_use.append({"a": sku, "b": cust, "props": {}})
            return tid

        def add_images(sku, brand, pkg, n):
            for _ in range(n):
                iid = f"IMG{len(inodes)+1:06d}"
                inodes.append({"id": iid, "props": {"source": "TrainingStation"}})
                itexts.append(t_img_ok(brand, pkg))
                iedges.append({"a": iid, "b": sku, "props": {}})

        def add_pallets(sku, brand, pkg, n_ok, n_fail):
            for _ in range(n_ok):
                pid = f"PAL{len(pnodes)+1:06d}"
                pnodes.append({"id": pid, "props": {"outcome": "success"}})
                ptexts.append(t_pallet_ok(brand, pkg)); pedges.append({"a": pid, "b": sku, "props": {}})
            for _ in range(n_fail):
                pid = f"PAL{len(pnodes)+1:06d}"
                pnodes.append({"id": pid, "props": {"outcome": "failure"}})
                ptexts.append(t_pallet_fail()); pedges.append({"a": pid, "b": sku, "props": {}})

        def add_merges(sku, n):
            for _ in range(n):
                mid = f"MRG{len(mnodes)+1:05d}"
                mnodes.append({"id": mid, "props": {"rollback_available": False,
                              "status": "conflicted"}})
                mtexts.append(t_merge_conflict()); medges.append({"a": sku, "b": mid, "props": {}})

        # ── HEALTHY: aligned neighbours → reflect ≈ self ────────────────────
        for r in healthy:
            add_tenant(r["sku"], r["brand"], r["pkg"], "ACME FOODS")
            add_images(r["sku"], r["brand"], r["pkg"], RNG.randint(3, 7))
            add_pallets(r["sku"], r["brand"], r["pkg"], RNG.randint(1, 3), 0)

        # ── EVIDENCE GAP: ZERO training images + many scan failures ─────────
        # No aligned TrainingImage neighbours; failure Pallets (w=2.5) dominate.
        for r in evidence_gap:
            add_tenant(r["sku"], r["brand"], r["pkg"], "ACME FOODS")
            add_pallets(r["sku"], r["brand"], r["pkg"], 0, 5)

        # ── MERGE CONFLICT: 5 conflicted merge events (w=3.0), no clean imgs ─
        for r in merge_conflict:
            add_tenant(r["sku"], r["brand"], r["pkg"], "ACME FOODS")
            add_merges(r["sku"], 5)
            add_pallets(r["sku"], r["brand"], r["pkg"], 0, 2)

        # ── AUTO-MAP ERROR: fuzzy tenants + conflicted merge + scan failures ───
        for r in auto_map:
            add_tenant(r["sku"], r["brand"], r["pkg"], "BLUE RIDGE DIST", wrong=True)
            add_tenant(r["sku"], r["brand"], r["pkg"], "CITYWIDE BEV", wrong=True)
            add_merges(r["sku"], 2)
            add_pallets(r["sku"], r["brand"], r["pkg"], 0, 6)

        # ── SHARED-SKU: all tenants divergent, no images, heavy scan failures ───
        for r in shared_sku:
            for c in CUSTOMERS[:4]:
                add_tenant(r["sku"], r["brand"], r["pkg"], c, wrong=True)
            add_pallets(r["sku"], r["brand"], r["pkg"], 0, 5)

        # ── BRAND MISMATCH: re-point SKU's brand edge to a WRONG/duplicate
        # Brand record (the correct one is removed) + scan failures. The
        # BELONGS_TO_BRAND neighbour is now divergent instead of aligned.
        wrong_brand_rows, wbtexts = [], []
        bm_edges_brand = []
        bm_sku_ids = [r["sku"] for r in brand_mismatch]
        for r in brand_mismatch:
            add_tenant(r["sku"], r["brand"], r["pkg"], "ACME FOODS")
            add_pallets(r["sku"], r["brand"], r["pkg"], 0, 3)        # scan failures
            bid = f"BWRONG{len(wrong_brand_rows)+1:04d}"
            wrong_brand_rows.append({"id": bid, "props": {"brand_family": "GENERIC IMPORT",
                                    "canonical": False}})
            wbtexts.append(t_brand_wrong())
            bm_edges_brand.append({"a": r["sku"], "b": bid, "props": {}})
        # remove the original (correct) brand edge so the mismatch is real
        run_batch(s, "UNWIND $rows AS r MATCH (g:GlobalSKU {sku_id:r.sku})"
                     "-[e:BELONGS_TO_BRAND]->(:Brand) DELETE e",
                  [{"sku": x} for x in bm_sku_ids])

        # ── ENCODE all new node texts ───────────────────────────────────────
        print("  encoding lifecycle node texts ...")
        for nodes, texts in [(tnodes, ttexts), (inodes, itexts), (mnodes, mtexts),
                             (pnodes, ptexts), (wrong_brand_rows, wbtexts)]:
            if texts:
                emb = embed(texts)
                for n, e in zip(nodes, emb):
                    n["self_emb"] = e.tolist()

        # ── WRITE nodes ─────────────────────────────────────────────────────
        print("  writing lifecycle nodes ...")
        write_nodes(s, "TenantSKU", "tenant_sku_id", tnodes)
        write_nodes(s, "TrainingImage", "image_id", inodes)
        write_nodes(s, "MergeEvent", "merge_id", mnodes)
        write_nodes(s, "Pallet", "pallet_id", pnodes)
        if wrong_brand_rows:
            write_nodes(s, "Brand", "brand_id", wrong_brand_rows)

        # ── WRITE edges (typed per handbook §3.2) ───────────────────────────
        print("  writing lifecycle relationships ...")
        write_edges(s, "MAPS_TO",    "TenantSKU", "tenant_sku_id", "GlobalSKU", "sku_id", tedges_map)
        write_edges(s, "USED_BY",    "GlobalSKU", "sku_id", "Customer", "customer_id", tedges_use)
        write_edges(s, "TRAINED_WITH","TrainingImage","image_id","GlobalSKU","sku_id", iedges)
        write_edges(s, "SCANNED_ON", "Pallet", "pallet_id", "GlobalSKU", "sku_id", pedges)
        write_edges(s, "MERGED_INTO","GlobalSKU","sku_id","MergeEvent","merge_id", medges)
        write_edges(s, "BELONGS_TO_BRAND","GlobalSKU","sku_id","Brand","brand_id", bm_edges_brand)

        # ── tag planted SKUs so scenario Cypher survives re-runs ─────────────
        def sku_ids(lst):
            return [r["sku"] for r in lst]

        def tag_planted(anomaly_type, sku_list):
            run_batch(
                s,
                "UNWIND $rows AS r MATCH (g:GlobalSKU {sku_id:r.sku}) SET g.planted_type=r.t",
                [{"sku": x, "t": anomaly_type} for x in sku_list],
            )

        tag_planted("brand_mismatch", sku_ids(brand_mismatch))
        tag_planted("merge_conflict", sku_ids(merge_conflict))
        tag_planted("evidence_gap", sku_ids(evidence_gap))
        tag_planted("shared_sku", sku_ids(shared_sku))
        tag_planted("auto_map_error", sku_ids(auto_map))

        # ── RECOMPUTE reflect_emb (attention + edge severity) ─────────────────
        print("  recomputing reflect_emb for cohort (attention) ...")
        cohort_ids = [r["sku"] for r in rows]
        n_refl = recompute_cohort_skus(s, cohort_ids)
        print(f"    → {n_refl:,} cohort SKUs updated")

        # ── node/rel summary ────────────────────────────────────────────────
        n_new = len(tnodes)+len(inodes)+len(mnodes)+len(pnodes)+len(cust_rows)+len(wrong_brand_rows)
        n_edges = len(tedges_map)+len(tedges_use)+len(iedges)+len(pedges)+len(medges)+len(bm_edges_brand)
        print(f"  created {n_new:,} lifecycle nodes, {n_edges:,} relationships")

    driver.close()

    # ── seed_manifest.json (planted anomalies + ground truth) ───────────────
    def ids(lst): return [r["sku"] for r in lst]
    manifest = {
        "cohort_tag": COHORT_TAG,
        "cohort_size": len(rows),
        "planted": (
            [{"sku_id": x, "anomaly_type": "brand_mismatch"} for x in ids(brand_mismatch)] +
            [{"sku_id": x, "anomaly_type": "merge_conflict"} for x in ids(merge_conflict)] +
            [{"sku_id": x, "anomaly_type": "evidence_gap"} for x in ids(evidence_gap)] +
            [{"sku_id": x, "anomaly_type": "shared_sku"} for x in ids(shared_sku)] +
            [{"sku_id": x, "anomaly_type": "auto_map_error"} for x in ids(auto_map)]
        ),
        # problem types graded for top-decile recall (criterion #3)
        "top_decile_types": [
            "brand_mismatch", "merge_conflict", "evidence_gap",
            "shared_sku", "auto_map_error",
        ],
        "healthy": ids(healthy),
        "ground_truth": (
            [{"sku_id": x, "label": "confirmed_anomaly"} for x in
             ids(brand_mismatch) + ids(merge_conflict) + ids(evidence_gap)
             + ids(shared_sku) + ids(auto_map)] +
            [{"sku_id": x, "label": "valid"} for x in ids(healthy)[:20]]
        ),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {MANIFEST_PATH} ({len(manifest['planted'])} planted, "
          f"{len(manifest['ground_truth'])} labels)")
    print("\nDone. Run: python 06_evaluate.py")


if __name__ == "__main__":
    main()
