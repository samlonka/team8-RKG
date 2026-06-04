"""
Shared master SKU loading: normalization, derived IDs, multi-UPC aliases.

Used by 02_seed_data.py, api/main.py, and api/agent_matcher.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import GLOBAL_SKU_CSV

MASTER_UPC_COLUMNS = ("upc", "each_upc", "case_upc", "unit_upc", "package_upc")


def normalize_upc(val: Any) -> str | None:
    """Normalize a UPC/GTIN to digits-only string, or None if missing."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    v = str(val).strip().strip('"')
    if v in ("", "nan", "None") or v.startswith("00000000"):
        return None
    digits = re.sub(r"[^0-9]", "", v)
    return digits if digits else None


def slug_id(prefix: str, text: str, max_len: int = 48) -> str:
    """Stable graph key from human-readable text."""
    t = str(text or "UNKNOWN").upper().strip()
    t = re.sub(r"[^A-Z0-9]+", "_", t).strip("_")
    if not t:
        t = "UNKNOWN"
    sid = f"{prefix}_{t}"
    return sid[:max_len] if len(sid) > max_len else sid


def derive_brand_id(brand_family: str, brand_name: str = "") -> str:
    family = str(brand_family or "").strip().upper()
    if family and family not in ("UNKNOWN", "NAN"):
        return slug_id("BR", family)
    return slug_id("BR", brand_name or "UNKNOWN")


def derive_package_type_id(package_name: str, package_category_name: str = "") -> str:
    name = str(package_name or "").strip()
    if name and name.upper() not in ("UNKNOWN", "NAN"):
        return slug_id("PKG", name)
    return slug_id("PKG", package_category_name or "UNKNOWN")


def collect_row_upcs(row: dict | pd.Series) -> list[str]:
    """All normalized UPCs for a master SKU row (unique, ordered)."""
    seen: set[str] = set()
    out: list[str] = []
    for col in MASTER_UPC_COLUMNS:
        u = normalize_upc(row.get(col) if hasattr(row, "get") else getattr(row, col, None))
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def enrich_global_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure brand_id, package_type_id, normalized UPC columns, and upc_aliases.
    Safe to call on already-enriched frames.
    """
    df = df.copy()

    if "brand_id" not in df.columns or df["brand_id"].isna().all():
        df["brand_id"] = [
            derive_brand_id(r.get("brand_family", ""), r.get("brand_name", ""))
            for _, r in df.iterrows()
        ]

    if "package_type_id" not in df.columns or df["package_type_id"].fillna("").eq("").all():
        df["package_type_id"] = [
            derive_package_type_id(
                r.get("package_name", ""),
                r.get("package_category_name", ""),
            )
            for _, r in df.iterrows()
        ]
    else:
        df["package_type_id"] = df["package_type_id"].fillna("0").astype(str).str.strip()
        missing = df["package_type_id"].isin(("", "0", "nan"))
        if missing.any():
            df.loc[missing, "package_type_id"] = [
                derive_package_type_id(
                    r.get("package_name", ""),
                    r.get("package_category_name", ""),
                )
                for _, r in df.loc[missing].iterrows()
            ]

    for col in MASTER_UPC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_upc)

    if "upc" in df.columns:
        df["upc_missing"] = df["upc"].isna()

    df["upc_aliases"] = [collect_row_upcs(r) for _, r in df.iterrows()]
    return df


def build_global_upc_lookup(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """
    Map normalized UPC digit string -> (sku_id, match_field).
    First SKU wins on duplicate UPC across catalog.
    """
    lookup: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        sku_id = str(row["sku_id"])
        for col in MASTER_UPC_COLUMNS:
            u = normalize_upc(row.get(col))
            if u and u not in lookup:
                lookup[u] = (sku_id, col)
    return lookup


def _upc_lookup_variants(digits: str) -> list[str]:
    """Digit strings to try against master UPC lookup (exact + GTIN padding)."""
    if not digits:
        return []
    variants = [digits]
    stripped = digits.lstrip("0")
    if stripped and stripped not in variants:
        variants.append(stripped)
    for width in (12, 13, 14):
        padded = digits.zfill(width)
        if padded not in variants:
            variants.append(padded)
        if stripped:
            sp = stripped.zfill(width)
            if sp not in variants:
                variants.append(sp)
    return variants


def match_vendor_to_global(
    vendor_row: dict,
    upc_lookup: dict[str, tuple[str, str]],
) -> tuple[str | None, str | None]:
    """
    Try vendor retail_upc, case_upc, eaches_upc against master lookup.
    Returns (sku_id, match_method) or (None, None).
    """
    checks = [
        ("retail_upc", "exact_retail_upc"),
        ("case_upc", "exact_case_upc"),
        ("eaches_upc", "exact_eaches_upc"),
    ]
    for field, method in checks:
        u = normalize_upc(vendor_row.get(field))
        if not u:
            continue
        for variant in _upc_lookup_variants(u):
            if variant in upc_lookup:
                sku_id, _ = upc_lookup[variant]
                tag = method if variant == u else "fuzzy_upc"
                return sku_id, tag
    return None, None


def _safe_float(val: Any) -> float | None:
    try:
        v = float(str(val).strip())
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def load_master_sku_records(path: str | Path | None = None) -> list[dict]:
    """
    Load master CSV by column name for API matching (brand + package scores).
    """
    path = Path(path or GLOBAL_SKU_CSV)
    if not path.is_absolute():
        root = Path(__file__).resolve().parent.parent
        path = root / path

    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip('"').strip() for c in df.columns]
    df = df.drop_duplicates(subset=["sku_id"], keep="first")
    df = enrich_global_dataframe(df)

    records: list[dict] = []
    for _, row in df.iterrows():
        h = _safe_float(row.get("height"))
        records.append({
            "sku_id":                str(row["sku_id"]).strip(),
            "status":                str(row.get("status", "")).strip(),
            "package_type_id":       str(row.get("package_type_id", "")).strip(),
            "package_category_name": str(row.get("package_category_name", "")).strip(),
            "short_description":     str(row.get("short_description", "")).strip(),
            "weight":                _safe_float(row.get("weight")),
            "height":                h,
            "length":                _safe_float(row.get("length")),
            "width":                 _safe_float(row.get("width")),
            "brand_name":            str(row.get("brand_name", "")).strip(),
            "brand_family":          str(row.get("brand_family", "")).strip(),
            "package_name":          str(row.get("package_name", "")).strip(),
        })
    return records
