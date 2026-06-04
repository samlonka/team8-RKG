#!/usr/bin/env python3
"""
Generate synthetic vendor export for master matching + anomaly edge-case tests.

Core buckets (50):
  exact_match (20), brand_only (10), package_only (10), no_match (10)

Edge buckets (+24 with --with-edges):
  fuzzy_upc, upc_conflict, ambiguous_dims, package_size_trap,
  missing_upc_text_match, brand_typo, weak_multi_signal, case_upc_only,
  hyphen_upc, lookup_skew

Outputs:
  data/synthetic_vendor_50.xlsx  (or synthetic_vendor_70.xlsx with --with-edges)

Usage:
  python scripts/generate_synthetic_vendor.py
  python scripts/generate_synthetic_vendor.py --with-edges
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GLOBAL_SKU_CSV
from data.master_loader import build_global_upc_lookup, enrich_global_dataframe, normalize_upc

WAREHOUSE = "SYNTH TEST VENDOR"
SUPPLIER = "SYNTHETIC QA SUPPLIER LLC"
PRODUCT_CLASS = "SYNTH TEST"


def _f(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none") else s


def _num(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _upc_pair(digits: str | None) -> tuple[str, str]:
    if not digits:
        return "", ""
    d = re.sub(r"[^0-9]", "", digits)
    if not d:
        return "", ""
    return f"00-0-{d}", d


def _brand_label(row: pd.Series) -> str:
    bn = _f(row.get("brand_name"))
    if bn:
        return bn
    return _f(row.get("brand_family")) or "UNKNOWN"


def _package_label(row: pd.Series) -> str:
    return _f(row.get("package_category_name")) or _f(row.get("package_name")) or "UNKNOWN"


def _pick_upc(row: pd.Series, prefer: str = "retail") -> str | None:
    order = (
        ["upc", "each_upc", "case_upc", "unit_upc", "package_upc"]
        if prefer == "retail"
        else ["case_upc", "upc", "each_upc", "unit_upc", "package_upc"]
    )
    for col in order:
        if col in row.index:
            u = normalize_upc(row.get(col))
            if u:
                return u
    return None


def _set_upcs(vendor: dict, upc: str | None, *, retail: bool = True, case: bool = True) -> None:
    pre, digits = _upc_pair(upc)
    if retail:
        vendor["Retail UPC"] = digits
        vendor["Retail UPC with Prefix"] = pre
        vendor["Retail Eaches UPC with Prefix"] = pre
        vendor["Eaches UPC"] = digits
    else:
        vendor["Retail UPC"] = ""
        vendor["Retail UPC with Prefix"] = ""
        vendor["Retail Eaches UPC with Prefix"] = ""
        vendor["Eaches UPC"] = ""
    if case:
        vendor["Case UPC"] = digits
        vendor["Case UPC with Prefix"] = pre
    else:
        vendor["Case UPC"] = ""
        vendor["Case UPC with Prefix"] = ""


def _lookup_owner(lookup: dict[str, tuple[str, str]], upc: str | None) -> str | None:
    if not upc:
        return None
    hit = lookup.get(upc)
    return hit[0] if hit else None


def _rows_with_owned_upc(
    pool: pd.DataFrame,
    lookup: dict[str, tuple[str, str]],
    *,
    prefer: str = "retail",
) -> pd.DataFrame:
    """SKUs whose chosen UPC resolves to that sku_id in the lookup (no first-wins skew)."""
    owners = []
    for _, row in pool.iterrows():
        u = _pick_upc(row, prefer)
        if u and _lookup_owner(lookup, u) == str(row["sku_id"]):
            owners.append(True)
        else:
            owners.append(False)
    return pool[pd.Series(owners, index=pool.index)]


def _master_pool(df: pd.DataFrame, *, require_upc: bool = False) -> pd.DataFrame:
    m = df.copy()
    m["v_brand"] = m.apply(_brand_label, axis=1)
    m["v_pkg"] = m.apply(_package_label, axis=1)
    m = m[(m["v_brand"] != "UNKNOWN") & (m["v_pkg"] != "UNKNOWN")]
    if require_upc:
        m["has_upc"] = m.apply(lambda r: _pick_upc(r) is not None, axis=1)
        m = m[m["has_upc"]]
    return m


def _vendor_row(
    product_id: str,
    master: pd.Series,
    *,
    brand: str | None = None,
    package: str | None = None,
    upc_from: pd.Series | None = None,
    use_master_dims: bool = True,
    sub: str | None = None,
    upc_mode: str = "retail",  # retail | case | none | fuzzy
) -> dict:
    b = brand if brand is not None else _brand_label(master)
    pkg = package if package is not None else _package_label(master)
    sub_val = sub or b[:24]

    w = _num(master.get("weight"), 10.0) if use_master_dims else 9.99
    h = _num(master.get("height"), 8.0) if use_master_dims else 7.5
    ln = _num(master.get("length"), 10.0) if use_master_dims else 9.0
    wd = _num(master.get("width"), 8.0) if use_master_dims else 7.0

    src = upc_from if upc_from is not None else master
    upc = None
    if upc_mode == "retail":
        upc = _pick_upc(src, "retail")
    elif upc_mode == "case":
        upc = _pick_upc(src, "case")
    elif upc_mode == "fuzzy" and (u := _pick_upc(src, "retail")):
        # GTIN-14 style padding (master often stores unpadded digits)
        upc = u.zfill(14) if len(u) < 14 else "0" + u

    row = {
        "• Warehouse": WAREHOUSE,
        "Product ID": product_id,
        "Product Description": pkg,
        "Description": pkg,
        "Sub": sub_val,
        "Brand": b,
        "Supplier": SUPPLIER,
        "Product Class": PRODUCT_CLASS,
        "Selling Units per Case": _f(master.get("units_per_case")) or "12",
        "Unit Weight": str(round(w, 2)),
        "Case Length (Inches)": str(round(ln, 2)),
        "Case Width (Inches)": str(round(wd, 2)),
        "Case Height (Inches)": str(round(h, 2)),
        "• Cases per Tier": "10",
        "Cases per Pallet Receiving": "80",
        "• Cases per Wide Pallet": "80",
        "• Cases per Narrow Pallet": "56",
        "Case UPC": "",
        "Case UPC with Prefix": "",
        "Retail UPC": "",
        "Retail UPC with Prefix": "",
        "Retail Eaches UPC with Prefix": "",
        "Eaches UPC": "",
    }
    if upc_mode == "case":
        _set_upcs(row, upc, retail=False, case=True)
    elif upc_mode in ("retail", "fuzzy"):
        _set_upcs(row, upc, retail=True, case=True)
    return row


def _record(
    rows: list,
    manifest: list,
    category: str,
    product_id: str,
    master_row: pd.Series,
    vendor: dict,
    expected_sku: str,
    notes: str,
    *,
    anomaly_hint: str = "",
    expected_action: str = "",
):
    rows.append(vendor)
    manifest.append({
        "product_id": product_id,
        "category": category,
        "anomaly_hint": anomaly_hint or category,
        "expected_master_sku_id": expected_sku,
        "expected_ingest_action": expected_action,
        "vendor_brand": vendor["Brand"],
        "vendor_package": vendor["Product Description"],
        "notes": notes,
    })


def generate_core(
    pool: pd.DataFrame,
    lookup: dict[str, tuple[str, str]],
    multi_pkg_brands: list,
    multi_brand_pkgs: list,
    seed: int,
):
    rows, manifest = [], []
    owned = _rows_with_owned_upc(pool[pool["has_upc"]], lookup, prefer="retail")
    exact_src = owned.sample(n=min(20, len(owned)), random_state=seed).reset_index(drop=True)

    for i, (_, m) in enumerate(exact_src.iterrows()):
        pid = f"SYNTH-EX{i+1:03d}"
        v = _vendor_row(pid, m, upc_mode="retail")
        _record(rows, manifest, "exact_match", pid, m, v, str(m["sku_id"]),
                "Brand, package, UPC, dimensions from master.",
                anomaly_hint="healthy_baseline",
                expected_action="AUTO_MATCH")

    br_i = 0
    for brand in multi_pkg_brands:
        if br_i >= 10:
            break
        grp = pool[pool["v_brand"] == brand]
        pkgs = grp["v_pkg"].unique().tolist()
        if len(pkgs) < 2:
            continue
        anchor = grp.iloc[0]
        other_pkg = pkgs[1] if pkgs[1] != anchor["v_pkg"] else pkgs[0]
        pid = f"SYNTH-BR{br_i+1:03d}"
        br_i += 1
        v = _vendor_row(pid, anchor, package=other_pkg, upc_mode="none")
        _record(rows, manifest, "brand_only", pid, anchor, v, "",
                f"Same brand '{brand}', wrong package '{other_pkg}'.",
                anomaly_hint="brand_mismatch",
                expected_action="REVIEW_QUEUE")

    pk_i = 0
    for pkg in multi_brand_pkgs:
        if pk_i >= 10:
            break
        grp = pool[pool["v_pkg"] == pkg]
        brands = grp["v_brand"].unique().tolist()
        if len(brands) < 2:
            continue
        anchor = grp[grp["v_brand"] == brands[0]].iloc[0]
        pid = f"SYNTH-PK{pk_i+1:03d}"
        pk_i += 1
        v = _vendor_row(pid, anchor, brand=brands[1], package=pkg, upc_mode="none", use_master_dims=False)
        _record(rows, manifest, "package_only", pid, anchor, v, "",
                f"Shared package '{pkg}', brand '{brands[1]}' vs master '{brands[0]}'.",
                anomaly_hint="auto_map_error",
                expected_action="REVIEW_QUEUE")

    rng = random.Random(seed)
    for i in range(10):
        pid = f"SYNTH-NM{i+1:03d}"
        v = _vendor_row(
            pid, exact_src.iloc[0],
            brand=f"ZZZ_SYNTH_VENDOR_{i+1:03d}",
            package=f"NOT_IN_MASTER_PKG_{rng.randint(1000, 9999)}",
            upc_mode="none", use_master_dims=False,
        )
        _record(rows, manifest, "no_match", pid, exact_src.iloc[0], v, "",
                "Synthetic brand/package; should create draft.",
                anomaly_hint="evidence_gap",
                expected_action="CREATE_NEW")

    return rows, manifest


def generate_edges(pool: pd.DataFrame, lookup: dict[str, tuple[str, str]], seed: int):
    rows, manifest = [], []
    rng = random.Random(seed + 1)
    with_upc = pool[pool["has_upc"]]
    owned_retail = _rows_with_owned_upc(with_upc, lookup, prefer="retail")
    owned_case = _rows_with_owned_upc(with_upc, lookup, prefer="case")

    # fuzzy_upc (3) — ingest fuzzy_upc / GTIN padding
    for i in range(3):
        m = owned_retail.sample(1, random_state=seed + 100 + i).iloc[0]
        pid = f"SYNTH-FU{i+1:03d}"
        v = _vendor_row(pid, m, upc_mode="fuzzy")
        _record(rows, manifest, "fuzzy_upc", pid, m, v, str(m["sku_id"]),
                "UPC with leading zeros; should still match via fuzzy_upc.",
                anomaly_hint="auto_map_error",
                expected_action="AUTO_MATCH")

    # upc_conflict (2) — UPC from SKU B, text from SKU A
    for i in range(2):
        pair = owned_retail.sample(2, random_state=seed + 200 + i)
        a, b = pair.iloc[0], pair.iloc[1]
        if _brand_label(a) == _brand_label(b):
            continue
        pid = f"SYNTH-UC{i+1:03d}"
        v = _vendor_row(pid, a, upc_from=b, upc_mode="retail")
        _record(rows, manifest, "upc_conflict", pid, a, v, str(a["sku_id"]),
                f"Text from SKU {a['sku_id']} but UPC from SKU {b['sku_id']}.",
                anomaly_hint="merge_conflict",
                expected_action="REVIEW_QUEUE")

    # ambiguous_dims (3) — same brand, similar packages, dims disambiguate
    for i in range(3):
        brand = pool.groupby("v_brand").filter(lambda g: len(g) >= 2)["v_brand"].iloc[0]
        grp = pool[pool["v_brand"] == brand].head(2)
        if len(grp) < 2:
            continue
        target = grp.iloc[0]
        decoy = grp.iloc[1]
        pid = f"SYNTH-AD{i+1:03d}"
        v = _vendor_row(pid, target, package=_package_label(decoy), upc_mode="none")
        # restore target dimensions explicitly
        v["Unit Weight"] = str(round(_num(target["weight"]), 2))
        v["Case Length (Inches)"] = str(round(_num(target["length"]), 2))
        v["Case Width (Inches)"] = str(round(_num(target["width"]), 2))
        v["Case Height (Inches)"] = str(round(_num(target["height"]), 2))
        _record(rows, manifest, "ambiguous_dims", pid, target, v, str(target["sku_id"]),
                "Similar brand SKUs; dimensions should pick target.",
                anomaly_hint="shared_sku",
                expected_action="REVIEW_QUEUE")

    # package_size_trap (3) — 10 vs 20 OZ style mismatch
    oz_rows = pool[pool["v_pkg"].str.contains(r"\b10\b", regex=True, na=False)]
    if len(oz_rows) >= 1:
        for i in range(min(3, len(oz_rows))):
            anchor = oz_rows.iloc[i % len(oz_rows)]
            wrong_pkg = re.sub(r"\b10\b", "20", _package_label(anchor), count=1)
            pid = f"SYNTH-PS{i+1:03d}"
            v = _vendor_row(pid, anchor, package=wrong_pkg, upc_mode="none")
            _record(rows, manifest, "package_size_trap", pid, anchor, v, "",
                    "20 OZ-style label vs master 10 OZ SKU — numeric penalty.",
                    anomaly_hint="auto_map_error",
                    expected_action="REVIEW_QUEUE")

    # missing_upc_text_match (3)
    for i in range(3):
        m = with_upc.sample(1, random_state=seed + 300 + i).iloc[0]
        pid = f"SYNTH-MU{i+1:03d}"
        v = _vendor_row(pid, m, upc_mode="none")
        _record(rows, manifest, "missing_upc_text_match", pid, m, v, str(m["sku_id"]),
                "No UPC; brand+package text should match API.",
                anomaly_hint="evidence_gap",
                expected_action="AUTO_MATCH")

    # brand_typo (2)
    for i in range(2):
        m = with_upc.sample(1, random_state=seed + 400 + i).iloc[0]
        brand = _brand_label(m)
        typo = brand[: max(4, len(brand) - 2)]  # truncate slug
        pid = f"SYNTH-BT{i+1:03d}"
        v = _vendor_row(pid, m, brand=typo, upc_mode="none")
        _record(rows, manifest, "brand_typo", pid, m, v, str(m["sku_id"]),
                f"Brand typo '{typo}' vs '{brand}'.",
                anomaly_hint="brand_mismatch",
                expected_action="REVIEW_QUEUE")

    # weak_multi_signal (2)
    for i in range(2):
        m = with_upc.sample(1, random_state=seed + 500 + i).iloc[0]
        pkg = _package_label(m)
        weak_pkg = pkg + " VAR"
        pid = f"SYNTH-WM{i+1:03d}"
        v = _vendor_row(pid, m, package=weak_pkg, upc_mode="none", use_master_dims=False)
        _record(rows, manifest, "weak_multi_signal", pid, m, v, "",
                "Slight package drift, no UPC — multi-signal / low confidence.",
                anomaly_hint="merge_conflict",
                expected_action="REVIEW_QUEUE")

    # case_upc_only (2)
    cu_i = 0
    for _, m in owned_case.iterrows():
        if cu_i >= 2:
            break
        case_u = _pick_upc(m, "case")
        if not case_u or case_u == _pick_upc(m, "retail"):
            continue
        pid = f"SYNTH-CU{cu_i+1:03d}"
        cu_i += 1
        v = _vendor_row(pid, m, upc_mode="case")
        _record(rows, manifest, "case_upc_only", pid, m, v, str(m["sku_id"]),
                "Match via case_upc when retail UPC absent.",
                anomaly_hint="healthy_baseline",
                expected_action="AUTO_MATCH")

    # hyphen_upc (2) — dashed export format still normalizes
    for i in range(2):
        m = owned_retail.sample(1, random_state=seed + 700 + i).iloc[0]
        u = _pick_upc(m, "retail")
        if not u:
            continue
        pid = f"SYNTH-HY{i+1:03d}"
        v = _vendor_row(pid, m, upc_mode="retail")
        pre, _ = _upc_pair(u)
        v["Retail UPC"] = f"{u[:4]}-{u[4:8]}-{u[8:]}" if len(u) >= 8 else u
        v["Retail UPC with Prefix"] = pre
        v["Eaches UPC"] = v["Retail UPC"]
        _record(rows, manifest, "hyphen_upc", pid, m, v, str(m["sku_id"]),
                "Dashed UPC in export; digits-only match expected.",
                anomaly_hint="healthy_baseline",
                expected_action="AUTO_MATCH")

    # lookup_skew (2) — UPC on row A but lookup first-wins to SKU B
    skew_rows = []
    for _, row in with_upc.iterrows():
        u = _pick_upc(row, "retail")
        owner = _lookup_owner(lookup, u)
        if u and owner and owner != str(row["sku_id"]):
            skew_rows.append((row, owner, u))
        if len(skew_rows) >= 2:
            break
    for i, (victim, owner, u) in enumerate(skew_rows):
        pid = f"SYNTH-LS{i+1:03d}"
        v = _vendor_row(pid, victim, upc_mode="retail")
        pre, digits = _upc_pair(u)
        v["Retail UPC"] = digits
        v["Eaches UPC"] = digits
        v["Case UPC"] = digits
        v["Retail UPC with Prefix"] = pre
        v["Case UPC with Prefix"] = pre
        _record(rows, manifest, "lookup_skew", pid, victim, v, str(victim["sku_id"]),
                f"UPC {u} lookup owner SKU {owner}; text from SKU {victim['sku_id']}.",
                anomaly_hint="merge_conflict",
                expected_action="REVIEW_QUEUE")

    return rows, manifest


def generate(seed: int = 42, with_edges: bool = False):
    raw = pd.read_csv(ROOT / GLOBAL_SKU_CSV, dtype=str, low_memory=False)
    raw.columns = [c.strip('"').strip() for c in raw.columns]
    master = enrich_global_dataframe(raw.drop_duplicates(subset=["sku_id"], keep="first"))
    pool = _master_pool(master, require_upc=False)
    pool["has_upc"] = pool.apply(lambda r: _pick_upc(r) is not None, axis=1)
    lookup = build_global_upc_lookup(master)

    multi_pkg_brands = [b for b, g in pool.groupby("v_brand") if g["v_pkg"].nunique() >= 2]
    multi_brand_pkgs = [p for p, g in pool.groupby("v_pkg") if g["v_brand"].nunique() >= 2]
    random.Random(seed).shuffle(multi_pkg_brands)
    random.Random(seed).shuffle(multi_brand_pkgs)

    rows, manifest = generate_core(pool, lookup, multi_pkg_brands, multi_brand_pkgs, seed)
    if with_edges:
        er, em = generate_edges(pool, lookup, seed)
        rows.extend(er)
        manifest.extend(em)

    return pd.DataFrame(rows), manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-edges", action="store_true", help="Add 24 anomaly edge-case rows")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    suffix = "74" if args.with_edges else "50"
    out = ROOT / (args.out or f"data/synthetic_vendor_{suffix}.xlsx")
    man = ROOT / (args.manifest or f"data/synthetic_vendor_{suffix}_manifest.json")

    df, manifest = generate(seed=args.seed, with_edges=args.with_edges)
    buckets = Counter(r["category"] for r in manifest)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)
    with open(man, "w") as f:
        json.dump({
            "warehouse": WAREHOUSE,
            "total_rows": len(manifest),
            "buckets": dict(buckets),
            "anomaly_hints": dict(Counter(r.get("anomaly_hint", "") for r in manifest)),
            "rows": manifest,
        }, f, indent=2)

    print(f"Wrote {out} ({len(df)} rows)")
    print(f"Wrote {man}")
    print("\nBuckets:")
    for cat, n in sorted(buckets.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
