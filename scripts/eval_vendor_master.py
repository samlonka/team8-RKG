#!/usr/bin/env python3
"""
Evaluate vendor SKUs against the master catalog.

Outputs match_report.csv with UPC match, brand/package API match, and summary stats.

Usage (from project root):
    python scripts/eval_vendor_master.py
    python scripts/eval_vendor_master.py --out reports/match_report.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GLOBAL_SKU_CSV, VENDOR_SKU_XLSX
from data.master_loader import (
    build_global_upc_lookup,
    enrich_global_dataframe,
    match_vendor_to_global,
    normalize_upc,
)
_seed = __import__("importlib").import_module("02_seed_data")
load_vendor_sku = _seed.load_vendor_sku


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=GLOBAL_SKU_CSV)
    ap.add_argument("--vendor", default=VENDOR_SKU_XLSX, help="Vendor XLSX path")
    ap.add_argument("--manifest", default=None, help="Optional manifest JSON for labeling")
    ap.add_argument("--out", default="reports/match_report.csv")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Also run brand/package fuzzy match per row (slow on large catalogs)",
    )
    args = ap.parse_args()

    master_path = ROOT / args.master
    vendor_path = ROOT / args.vendor
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_master = pd.read_csv(master_path, dtype=str, low_memory=False)
    df_master.columns = [c.strip('"').strip() for c in df_master.columns]
    df_master = df_master.drop_duplicates(subset=["sku_id"], keep="first")
    df_master = enrich_global_dataframe(df_master)
    upc_lookup = build_global_upc_lookup(df_master)

    df_vendor = load_vendor_sku(str(vendor_path))

    if args.full:
        from api.main import match_sku  # noqa: WPS433 — loads master once into cache

    rows = []
    for _, v in df_vendor.iterrows():
        vd = v.to_dict()
        pid = str(v["product_id"])
        sku_upc, method_upc = match_vendor_to_global(vd, upc_lookup)

        brand = str(v.get("brand") or "").strip()
        pkg = str(v.get("product_description") or "").strip()
        bp_status, bp_conf, bp_sku = "", 0.0, ""
        if args.full and brand and pkg:
            bp = match_sku(brand, pkg)
            bp_status = bp.get("status", "")
            bp_conf = bp.get("confidence", 0)
            if bp.get("matched_skus"):
                bp_sku = bp["matched_skus"][0].get("sku_id", "")

        rows.append({
            "product_id": pid,
            "brand": brand,
            "product_description": pkg,
            "retail_upc": v.get("retail_upc"),
            "case_upc": v.get("case_upc"),
            "upc_match_sku_id": sku_upc or "",
            "upc_match_method": method_upc or "",
            "brand_package_status": bp_status,
            "brand_package_confidence": bp_conf,
            "brand_package_sku_id": bp_sku,
            "any_master_link": bool(
                sku_upc or bp_status in ("merged", "updated")
            ),
        })

    report = pd.DataFrame(rows)
    report.to_csv(out_path, index=False)

    n = len(report)
    upc_hits = (report["upc_match_sku_id"] != "").sum()
    bp_hits = report["brand_package_status"].isin(["merged", "updated"]).sum()
    any_link = report["any_master_link"].sum()

    print(f"Vendor rows: {n:,}")
    print(f"UPC match to master: {upc_hits:,} ({100*upc_hits/n:.1f}%)")
    print(f"Brand/package merged|updated: {bp_hits:,} ({100*bp_hits/n:.1f}%)")
    print(f"Any master link (UPC or brand/package): {any_link:,} ({100*any_link/n:.1f}%)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
