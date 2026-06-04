#!/usr/bin/env python3
"""
Score synthetic vendor file against master + ground-truth manifest.

Usage:
  python scripts/test_synthetic_vendor.py
  python scripts/test_synthetic_vendor.py --vendor data/synthetic_vendor_70.xlsx --full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GLOBAL_SKU_CSV
from data.master_loader import (
    build_global_upc_lookup,
    enrich_global_dataframe,
    match_vendor_to_global,
)

CORE_CATS = ("exact_match", "brand_only", "package_only", "no_match")
EDGE_CATS = (
    "fuzzy_upc", "upc_conflict", "ambiguous_dims", "package_size_trap",
    "missing_upc_text_match", "brand_typo", "weak_multi_signal", "case_upc_only",
    "hyphen_upc", "lookup_skew",
)


def _score_row(cat: str, expected: str, sku_upc: str | None, bp_status: str, bp_sku: str, full: bool):
    upc_ok = bp_ok = None

    if cat == "exact_match":
        upc_ok = sku_upc == expected
        bp_ok = (bp_sku == expected) if full else None
    elif cat == "missing_upc_text_match":
        upc_ok = not sku_upc
        bp_ok = (bp_sku == expected or bp_status == "merged") if full else None
    elif cat in ("fuzzy_upc", "case_upc_only", "hyphen_upc"):
        upc_ok = sku_upc == expected
        bp_ok = (bp_sku == expected or bp_status == "merged") if full else None
    elif cat == "lookup_skew":
        upc_ok = sku_upc is not None and sku_upc != expected
        bp_ok = bp_status in ("updated", "insert", "merged", "") if full else None
    elif cat == "no_match":
        upc_ok = not sku_upc
        bp_ok = bp_status == "insert" if full else None
    elif cat == "upc_conflict":
        upc_ok = sku_upc is not None and sku_upc != expected
        bp_ok = bp_status in ("updated", "insert", "merged") if full else None
    elif cat in ("brand_only", "package_only", "package_size_trap", "brand_typo", "weak_multi_signal"):
        upc_ok = sku_upc != expected if expected else not sku_upc
        bp_ok = (
            bp_status in ("updated", "insert")
            or (bp_sku and bp_sku != expected)
        ) if full else None
    elif cat == "ambiguous_dims":
        upc_ok = not sku_upc or sku_upc == expected
        bp_ok = (bp_sku == expected or bp_status in ("merged", "updated")) if full else None
    else:
        upc_ok = bool(sku_upc)
        bp_ok = bp_status != "insert" if full else None

    return upc_ok, bp_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", default="data/synthetic_vendor_50.xlsx")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    vendor_path = ROOT / args.vendor
    man_path = ROOT / (args.manifest or args.vendor.replace(".xlsx", "_manifest.json"))

    man = json.load(open(man_path))
    truth = {r["product_id"]: r for r in man["rows"]}

    raw = pd.read_csv(ROOT / GLOBAL_SKU_CSV, dtype=str, low_memory=False)
    raw.columns = [c.strip('"').strip() for c in raw.columns]
    master = enrich_global_dataframe(raw.drop_duplicates(subset=["sku_id"], keep="first"))
    lookup = build_global_upc_lookup(master)

    _seed = __import__("importlib").import_module("02_seed_data")
    vendor = _seed.load_vendor_sku(str(vendor_path))

    if args.full:
        from api.main import match_sku

    results = []
    for _, v in vendor.iterrows():
        pid = str(v["product_id"])
        t = truth[pid]
        vd = v.to_dict()
        sku_upc, method = match_vendor_to_global(vd, lookup)

        bp_status, bp_sku = "", ""
        if args.full:
            bp = match_sku(str(v["brand"]), str(v["product_description"]))
            bp_status = bp.get("status", "")
            if bp.get("matched_skus"):
                bp_sku = bp["matched_skus"][0]["sku_id"]

        cat = t["category"]
        expected = t["expected_master_sku_id"]
        upc_ok, bp_ok = _score_row(cat, expected, sku_upc, bp_status, bp_sku, args.full)

        results.append({
            "product_id": pid,
            "category": cat,
            "anomaly_hint": t.get("anomaly_hint", ""),
            "expected_action": t.get("expected_ingest_action", ""),
            "upc_match_sku": sku_upc or "",
            "upc_method": method or "",
            "expected_sku": expected,
            "upc_ok": upc_ok,
            "brand_package_status": bp_status,
            "brand_package_sku": bp_sku,
            "bp_ok": bp_ok,
        })

    df = pd.DataFrame(results)
    out = ROOT / "reports/synthetic_vendor_test_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {out}\n")
    all_cats = list(CORE_CATS) + list(EDGE_CATS)
    seen = set()
    for cat in all_cats + sorted(df["category"].unique()):
        if cat in seen:
            continue
        seen.add(cat)
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        upc_rate = sub["upc_ok"].mean() * 100
        line = f"  {cat:22} n={len(sub):2}  UPC OK: {upc_rate:.0f}%"
        if args.full and sub["bp_ok"].notna().any():
            line += f"  |  brand/pkg OK: {sub['bp_ok'].mean()*100:.0f}%"
        print(line)

    if "anomaly_hint" in df.columns:
        print("\nBy anomaly_hint (ingest / lifecycle analog):")
        for hint, grp in df.groupby("anomaly_hint"):
            print(f"  {hint:18} n={len(grp):2}  UPC OK: {grp['upc_ok'].mean()*100:.0f}%")


if __name__ == "__main__":
    main()
