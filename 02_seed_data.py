"""
02_seed_data.py — Load, normalize, seed Neo4j, and generate self_emb

Pipeline:
  1. Load & normalize Global SKU CSV
  2. Load & normalize Vendor SKU XLSX
  3. Derive Brand, PackageType, Manufacturer, Supplier, ProductClass nodes
  4. Insert all nodes into Neo4j (MERGE — idempotent)
  5. Create all relationships (MAPS_TO via UPC, BELONGS_TO_BRAND, etc.)
  6. Detect fuzzy brand matches → FUZZY_MATCH edges
  7. Generate self_emb for every node → store on Neo4j node

Usage:
    python 02_seed_data.py
"""

import re
import uuid
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GLOBAL_SKU_CSV, VENDOR_SKU_XLSX,
    EMBEDDING_MODEL, EMBED_BATCH_SIZE,
    BRAND_FUZZY_TOP_K, BRAND_FUZZY_MIN_SIM,
    GLOBAL_SKU_FIELDS, VENDOR_SKU_FIELDS,
)
from data.master_loader import (
    enrich_global_dataframe,
    build_global_upc_lookup,
    match_vendor_to_global,
    normalize_upc,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def load_global_sku(path: str) -> pd.DataFrame:
    """
    Load Global SKU CSV and normalize fields.

    Key normalizations:
    - status: uppercase → single canonical form
    - upc: '00000000None' / '00000000none' → None (flag as missing)
    - brand_family: strip whitespace, uppercase
    - package_category_name: strip whitespace
    - numeric fields: coerce to float, fill NaN with 0
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # Strip quotes that CSV may have left on column names
    df.columns = [c.strip('"').strip() for c in df.columns]

    # Deduplicate: the CSV is a join — keep first occurrence per sku_id
    df = df.drop_duplicates(subset=["sku_id"], keep="first")

    # Status normalization
    df["status"] = df["status"].str.strip().str.upper().fillna("UNKNOWN")

    df["upc"] = df["upc"].apply(normalize_upc)

    # Brand family: human-readable, uppercase, strip underscores for display
    df["brand_family"] = (
        df["brand_family"]
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )

    # brand_name (encoded slug like AQUA_WTR) — keep as-is for graph, decode for text
    df["brand_name"] = df["brand_name"].fillna("UNKNOWN").str.strip()

    # Package
    df["package_category_name"] = (
        df["package_category_name"].fillna("UNKNOWN").str.strip()
    )
    if "package_name" in df.columns:
        df["package_name"] = df["package_name"].fillna("").str.strip()

    df = enrich_global_dataframe(df)

    # Manufacturer — abbreviations kept as-is
    df["manufacturer"] = df["manufacturer"].fillna("UNKNOWN").str.strip().str.upper()

    # Numeric fields
    for col in ["weight", "height", "length", "width", "units_per_case"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Boolean flags
    for col in ["is_imaged_on_training_station", "is_imaged_on_wrapper",
                "is_review_needed", "is_inserted_through_picklist_api"]:
        if col in df.columns:
            df[col] = df[col].fillna("0").astype(str).str.strip().isin(["1", "true", "True"])

    # product_category
    df["product_category"] = df["product_category"].fillna("UNKNOWN").str.strip()

    print(f"  Global SKU loaded: {len(df):,} rows | "
          f"{df['upc_missing'].sum():,} missing UPCs | "
          f"{df['brand_family'].nunique()} unique brand families")
    return df


def load_vendor_sku(path: str) -> pd.DataFrame:
    """
    Load Vendor SKU XLSX and normalize fields.

    Key normalizations:
    - Column rename via VENDOR_SKU_FIELDS map
    - Parse package info from Product Description (e.g. '12/16Z CN' → qty, size, container)
    - UPC fields: coerce to string, strip prefix noise
    - Brand, Supplier, Product Class: uppercase, strip
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Rename to canonical names
    rename_map = {k: v for k, v in VENDOR_SKU_FIELDS.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Drop rows with no product_id
    df = df.dropna(subset=["product_id"])
    df["product_id"] = df["product_id"].str.strip()
    df = df.drop_duplicates(subset=["product_id"], keep="first")

    # Text fields — uppercase, strip
    for col in ["brand", "supplier", "product_class", "product_description", "warehouse"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").str.strip().str.upper()

    # UPC fields — remove dashes and non-digits introduced by the export format
    def clean_upc(val):
        v = str(val).strip()
        if v in ("nan", "None", ""):
            return None
        # remove hyphens and leading zeros in prefix format "00-0-XXXXXXXXX"
        digits = re.sub(r"[^0-9]", "", v)
        return digits if digits else None

    for col in ["case_upc", "retail_upc", "eaches_upc"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_upc)

    # Numeric fields
    for col in ["units_per_case", "unit_weight", "case_length", "case_width", "case_height"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Parse package info from description: "3D ALPHALAND 12/16Z CN"
    # Pattern: <qty>/<size><unit> <container>
    pkg_pattern = re.compile(r"(\d+)/(\d+\.?\d*)(OZ|L|ML|GAL)\s*(CN|PL|BT|CAN|BTL|PKG|CS)?", re.I)

    def parse_package(desc):
        m = pkg_pattern.search(str(desc))
        if m:
            return {
                "pkg_qty":       int(m.group(1)),
                "pkg_size":      float(m.group(2)),
                "pkg_unit":      m.group(3).upper(),
                "pkg_container": (m.group(4) or "").upper(),
            }
        return {"pkg_qty": 0, "pkg_size": 0.0, "pkg_unit": "", "pkg_container": ""}

    pkg_info = df["product_description"].apply(parse_package).apply(pd.Series)
    df = pd.concat([df, pkg_info], axis=1)

    print(f"  Vendor SKU loaded: {len(df):,} rows | "
          f"{df['brand'].nunique()} brands | "
          f"{df['supplier'].nunique()} suppliers | "
          f"{df['product_class'].nunique()} product classes")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT CONSTRUCTION FOR SELF-EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def global_sku_to_text(row) -> str:
    """
    Build a rich natural-language string from Global SKU attributes.
    UPC status is a boolean signal (present/missing), not the raw number.
    Package described via human-readable category name, not numeric ID.

    NOTE: the package code is included for context only — a sentence encoder
    cannot distinguish sizes from it ('16OZ' vs '12OZ' embed ~0.96 similar,
    and expanding to '16 ounce' does not help). Size discrimination must come
    from a numeric channel, not this text. See parse_package_code().
    """
    upc_status = "missing" if row.get("upc_missing") else "present"
    return (
        f"global SKU "
        f"brand {row.get('brand_family', 'unknown')} "
        f"package {row.get('package_category_name', 'unknown')} "
        f"manufacturer {row.get('manufacturer', 'unknown')} "
        f"category {row.get('product_category', 'unknown')} "
        f"units per case {int(row.get('units_per_case', 0))} "
        f"weight {row.get('weight', 0):.2f} "
        f"upc {upc_status} "
        f"status {row.get('status', 'unknown')} "
        f"imaged {int(row.get('is_imaged_on_training_station', 0))}"
    )


def vendor_sku_to_text(row) -> str:
    """
    Build embedding text for Vendor SKU.
    Package info extracted from description is included as structured fields.
    """
    container = row.get("pkg_container", "") or "unknown"
    return (
        f"vendor SKU "
        f"brand {row.get('brand', 'unknown')} "
        f"supplier {row.get('supplier', 'unknown')} "
        f"class {row.get('product_class', 'unknown')} "
        f"units per case {int(row.get('units_per_case', 0))} "
        f"size {row.get('pkg_size', 0)}{row.get('pkg_unit', '')} "
        f"container {container} "
        f"warehouse {row.get('warehouse', 'unknown')}"
    )


def brand_to_text(brand_family: str) -> str:
    return f"brand {brand_family}"


def package_type_to_text(name: str, packages_per_case) -> str:
    return f"package type {name} packages per case {packages_per_case}"


def manufacturer_to_text(name: str) -> str:
    return f"manufacturer {name}"


def supplier_to_text(name: str) -> str:
    return f"supplier {name}"


def product_class_to_text(name: str) -> str:
    return f"product class {name}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDING GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Batch encode a list of strings → (N, 768) float32 array."""
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2-normalized → cosine = dot product
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. NEO4J WRITE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(session, cypher: str, rows: list[dict], batch_size: int = 500):
    """Execute a parameterised Cypher statement in batches."""
    for i in range(0, len(rows), batch_size):
        chunk = rows[i: i + batch_size]
        session.run(cypher, {"rows": chunk})


# ─────────────────────────────────────────────────────────────────────────────
# 5. NODE INSERTION
# ─────────────────────────────────────────────────────────────────────────────

def seed_global_skus(session, df: pd.DataFrame, model: SentenceTransformer):
    print("  Seeding GlobalSKU nodes ...")
    texts = [global_sku_to_text(row) for _, row in df.iterrows()]
    embeddings = embed_texts(texts, model)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append({
            "sku_id":                     str(row["sku_id"]),
            "upc":                        row.get("upc"),
            "upc_missing":                bool(row.get("upc_missing", False)),
            "status":                     row["status"],
            "package_type_id":            str(row.get("package_type_id", "")),
            "package_category_name":      row.get("package_category_name", ""),
            "brand_id":                   str(row.get("brand_id", "")),
            "brand_family":               row.get("brand_family", ""),
            "brand_name":                 row.get("brand_name", ""),
            "manufacturer":               row.get("manufacturer", ""),
            "product_category":           row.get("product_category", ""),
            "units_per_case":             float(row.get("units_per_case", 0)),
            "weight":                     float(row.get("weight", 0)),
            "height":                     float(row.get("height", 0)),
            "length":                     float(row.get("length", 0)),
            "width":                      float(row.get("width", 0)),
            "is_imaged_on_training_station": bool(row.get("is_imaged_on_training_station", False)),
            "is_imaged_on_wrapper":       bool(row.get("is_imaged_on_wrapper", False)),
            "is_review_needed":           bool(row.get("is_review_needed", False)),
            "vor_reference_number":       str(row.get("vor_reference_number", "")),
            "each_upc":                   row.get("each_upc"),
            "case_upc":                   row.get("case_upc"),
            "unit_upc":                   row.get("unit_upc"),
            "package_upc":                row.get("package_upc"),
            "package_name":               str(row.get("package_name", "")),
            "upc_aliases":                list(row.get("upc_aliases") or []),
            "self_emb":                   embeddings[i].tolist(),
        })

    cypher = """
    UNWIND $rows AS r
    MERGE (s:GlobalSKU {sku_id: r.sku_id})
    SET s += {
        upc:                          r.upc,
        upc_missing:                  r.upc_missing,
        status:                       r.status,
        package_type_id:              r.package_type_id,
        package_category_name:        r.package_category_name,
        package_name:                 r.package_name,
        brand_id:                     r.brand_id,
        brand_family:                 r.brand_family,
        brand_name:                   r.brand_name,
        manufacturer:                 r.manufacturer,
        product_category:             r.product_category,
        units_per_case:               r.units_per_case,
        weight:                       r.weight,
        height:                       r.height,
        length:                       r.length,
        width:                        r.width,
        is_imaged_on_training_station: r.is_imaged_on_training_station,
        is_imaged_on_wrapper:         r.is_imaged_on_wrapper,
        is_review_needed:             r.is_review_needed,
        vor_reference_number:         r.vor_reference_number,
        each_upc:                     r.each_upc,
        case_upc:                     r.case_upc,
        unit_upc:                     r.unit_upc,
        package_upc:                  r.package_upc,
        upc_aliases:                  r.upc_aliases,
        self_emb:                     r.self_emb
    }
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} GlobalSKU nodes written")


def seed_vendor_skus(session, df: pd.DataFrame, model: SentenceTransformer):
    print("  Seeding VendorSKU nodes ...")
    texts = [vendor_sku_to_text(row) for _, row in df.iterrows()]
    embeddings = embed_texts(texts, model)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append({
            "product_id":       str(row["product_id"]),
            "product_description": row.get("product_description", ""),
            "brand":            row.get("brand", ""),
            "supplier":         row.get("supplier", ""),
            "product_class":    row.get("product_class", ""),
            "warehouse":        row.get("warehouse", ""),
            "units_per_case":   float(row.get("units_per_case", 0)),
            "unit_weight":      float(row.get("unit_weight", 0)),
            "case_length":      float(row.get("case_length", 0)),
            "case_width":       float(row.get("case_width", 0)),
            "case_height":      float(row.get("case_height", 0)),
            "case_upc":         row.get("case_upc"),
            "retail_upc":       row.get("retail_upc"),
            "eaches_upc":       row.get("eaches_upc"),
            "pkg_qty":          int(row.get("pkg_qty", 0)),
            "pkg_size":         float(row.get("pkg_size", 0)),
            "pkg_unit":         row.get("pkg_unit", ""),
            "pkg_container":    row.get("pkg_container", ""),
            "self_emb":         embeddings[i].tolist(),
        })

    cypher = """
    UNWIND $rows AS r
    MERGE (v:VendorSKU {product_id: r.product_id})
    SET v += {
        product_description: r.product_description,
        brand:               r.brand,
        supplier:            r.supplier,
        product_class:       r.product_class,
        warehouse:           r.warehouse,
        units_per_case:      r.units_per_case,
        unit_weight:         r.unit_weight,
        case_length:         r.case_length,
        case_width:          r.case_width,
        case_height:         r.case_height,
        case_upc:            r.case_upc,
        retail_upc:          r.retail_upc,
        eaches_upc:          r.eaches_upc,
        pkg_qty:             r.pkg_qty,
        pkg_size:            r.pkg_size,
        pkg_unit:            r.pkg_unit,
        pkg_container:       r.pkg_container,
        self_emb:            r.self_emb
    }
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} VendorSKU nodes written")


def seed_brands(session, df_global: pd.DataFrame, model: SentenceTransformer):
    """
    Brand nodes are derived from brand_family (human-readable).
    brand_id is the canonical key — brand_family is the display label.
    One brand_family can have many brand_name slugs (the fragmentation problem).
    """
    print("  Seeding Brand nodes ...")

    # Deduplicate by brand_id — keep most common brand_family for that id
    brands = (
        df_global.groupby("brand_id")["brand_family"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"brand_family": "brand_family"})
    )

    texts = [brand_to_text(r["brand_family"]) for _, r in brands.iterrows()]
    embeddings = embed_texts(texts, model)

    rows = [
        {
            "brand_id":     str(r["brand_id"]),
            "brand_family": r["brand_family"],
            "self_emb":     embeddings[i].tolist(),
        }
        for i, (_, r) in enumerate(brands.iterrows())
    ]

    cypher = """
    UNWIND $rows AS r
    MERGE (b:Brand {brand_id: r.brand_id})
    SET b.brand_family = r.brand_family,
        b.self_emb     = r.self_emb
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} Brand nodes written")


def seed_package_types(session, df_global: pd.DataFrame, model: SentenceTransformer):
    print("  Seeding PackageType nodes ...")

    pkgs = (
        df_global[["package_type_id", "package_category_name"]]
        .drop_duplicates(subset=["package_type_id"])
    )

    texts = [
        package_type_to_text(r["package_category_name"], 1)
        for _, r in pkgs.iterrows()
    ]
    embeddings = embed_texts(texts, model)

    rows = [
        {
            "package_type_id":       str(r["package_type_id"]),
            "package_category_name": r["package_category_name"],
            "self_emb":              embeddings[i].tolist(),
        }
        for i, (_, r) in enumerate(pkgs.iterrows())
    ]

    cypher = """
    UNWIND $rows AS r
    MERGE (p:PackageType {package_type_id: r.package_type_id})
    SET p.package_category_name = r.package_category_name,
        p.self_emb              = r.self_emb
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} PackageType nodes written")


def seed_manufacturers(session, df_global: pd.DataFrame, model: SentenceTransformer):
    print("  Seeding Manufacturer nodes ...")

    mfrs = df_global["manufacturer"].dropna().unique().tolist()
    texts = [manufacturer_to_text(m) for m in mfrs]
    embeddings = embed_texts(texts, model)

    rows = [
        {"name": mfrs[i], "self_emb": embeddings[i].tolist()}
        for i in range(len(mfrs))
    ]

    cypher = """
    UNWIND $rows AS r
    MERGE (m:Manufacturer {name: r.name})
    SET m.self_emb = r.self_emb
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} Manufacturer nodes written")


def seed_suppliers(session, df_vendor: pd.DataFrame, model: SentenceTransformer):
    print("  Seeding Supplier nodes ...")

    suppliers = df_vendor["supplier"].dropna().unique().tolist()
    texts = [supplier_to_text(s) for s in suppliers]
    embeddings = embed_texts(texts, model)

    rows = [
        {"name": suppliers[i], "self_emb": embeddings[i].tolist()}
        for i in range(len(suppliers))
    ]

    cypher = """
    UNWIND $rows AS r
    MERGE (s:Supplier {name: r.name})
    SET s.self_emb = r.self_emb
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} Supplier nodes written")


def seed_product_classes(session, df_vendor: pd.DataFrame, model: SentenceTransformer):
    print("  Seeding ProductClass nodes ...")

    classes = df_vendor["product_class"].dropna().unique().tolist()
    texts = [product_class_to_text(c) for c in classes]
    embeddings = embed_texts(texts, model)

    rows = [
        {"name": classes[i], "self_emb": embeddings[i].tolist()}
        for i in range(len(classes))
    ]

    cypher = """
    UNWIND $rows AS r
    MERGE (c:ProductClass {name: r.name})
    SET c.self_emb = r.self_emb
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} ProductClass nodes written")


# ─────────────────────────────────────────────────────────────────────────────
# 6. RELATIONSHIP CREATION
# ─────────────────────────────────────────────────────────────────────────────

def create_belongs_to_brand(session, df_global: pd.DataFrame):
    """GlobalSKU -[:BELONGS_TO_BRAND]-> Brand"""
    print("  Creating BELONGS_TO_BRAND relationships ...")
    rows = [
        {"sku_id": str(r["sku_id"]), "brand_id": str(r["brand_id"])}
        for _, r in df_global.iterrows()
        if str(r.get("brand_id", "")).strip()
    ]
    cypher = """
    UNWIND $rows AS r
    MATCH (s:GlobalSKU {sku_id: r.sku_id})
    MATCH (b:Brand     {brand_id: r.brand_id})
    MERGE (s)-[:BELONGS_TO_BRAND]->(b)
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} BELONGS_TO_BRAND edges")


def create_has_package(session, df_global: pd.DataFrame):
    """GlobalSKU -[:HAS_PACKAGE]-> PackageType"""
    print("  Creating HAS_PACKAGE relationships ...")
    rows = [
        {"sku_id": str(r["sku_id"]), "package_type_id": str(r["package_type_id"])}
        for _, r in df_global.iterrows()
        if str(r.get("package_type_id", "")).strip() not in ("", "0")
    ]
    cypher = """
    UNWIND $rows AS r
    MATCH (s:GlobalSKU  {sku_id: r.sku_id})
    MATCH (p:PackageType {package_type_id: r.package_type_id})
    MERGE (s)-[:HAS_PACKAGE]->(p)
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} HAS_PACKAGE edges")


def create_made_by(session, df_global: pd.DataFrame):
    """GlobalSKU -[:MADE_BY]-> Manufacturer"""
    print("  Creating MADE_BY relationships ...")
    rows = [
        {"sku_id": str(r["sku_id"]), "manufacturer": r["manufacturer"]}
        for _, r in df_global.iterrows()
        if r.get("manufacturer", "UNKNOWN") != "UNKNOWN"
    ]
    cypher = """
    UNWIND $rows AS r
    MATCH (s:GlobalSKU   {sku_id: r.sku_id})
    MATCH (m:Manufacturer {name: r.manufacturer})
    MERGE (s)-[:MADE_BY]->(m)
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} MADE_BY edges")


def create_maps_to(session, df_global: pd.DataFrame, df_vendor: pd.DataFrame):
    """
    VendorSKU -[:MAPS_TO]-> GlobalSKU via retail/case/eaches UPC match
    against all master UPC fields (upc, each_upc, case_upc, unit_upc, package_upc).
    """
    print("  Creating MAPS_TO relationships (multi-UPC join) ...")

    upc_lookup = build_global_upc_lookup(df_global)

    rows = []
    for _, r in df_vendor.iterrows():
        pid = str(r["product_id"])
        matched_sku, method = match_vendor_to_global(r.to_dict(), upc_lookup)
        if matched_sku:
            rows.append({
                "product_id":   pid,
                "sku_id":       matched_sku,
                "match_method": method,
            })

    cypher = """
    UNWIND $rows AS r
    MATCH (v:VendorSKU {product_id: r.product_id})
    MATCH (g:GlobalSKU {sku_id:     r.sku_id})
    MERGE (v)-[e:MAPS_TO]->(g)
    SET e.match_method = r.match_method
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} MAPS_TO edges ({len(rows)} vendor SKUs matched to global)")


def create_supplied_by(session, df_vendor: pd.DataFrame):
    """VendorSKU -[:SUPPLIED_BY]-> Supplier"""
    print("  Creating SUPPLIED_BY relationships ...")
    rows = [
        {"product_id": str(r["product_id"]), "supplier": r["supplier"]}
        for _, r in df_vendor.iterrows()
        if r.get("supplier", "UNKNOWN") != "UNKNOWN"
    ]
    cypher = """
    UNWIND $rows AS r
    MATCH (v:VendorSKU {product_id: r.product_id})
    MATCH (s:Supplier  {name: r.supplier})
    MERGE (v)-[:SUPPLIED_BY]->(s)
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} SUPPLIED_BY edges")


def create_in_class(session, df_vendor: pd.DataFrame):
    """VendorSKU -[:IN_CLASS]-> ProductClass"""
    print("  Creating IN_CLASS relationships ...")
    rows = [
        {"product_id": str(r["product_id"]), "product_class": r["product_class"]}
        for _, r in df_vendor.iterrows()
        if r.get("product_class", "UNKNOWN") != "UNKNOWN"
    ]
    cypher = """
    UNWIND $rows AS r
    MATCH (v:VendorSKU   {product_id: r.product_id})
    MATCH (c:ProductClass {name: r.product_class})
    MERGE (v)-[:IN_CLASS]->(c)
    """
    run_batch(session, cypher, rows)
    print(f"    → {len(rows):,} IN_CLASS edges")


def create_fuzzy_brand_matches(session, model: SentenceTransformer, df_global: pd.DataFrame):
    """
    Detect Brand nodes whose self_emb cosine similarity exceeds threshold.
    Creates bidirectional FUZZY_MATCH edges with a confidence score.
    This surfaces the brand fragmentation problem (Scenario 1).
    """
    print("  Detecting FUZZY_MATCH brand pairs ...")

    result = session.run(
        "MATCH (b:Brand) RETURN b.brand_id AS id, b.brand_family AS name, b.self_emb AS emb"
    )
    # Exclude placeholder 'UNKNOWN' brands: ~65% of brand nodes have no name, so
    # they all share the identical text "brand UNKNOWN" -> cosine 1.0 with each
    # other. Including them produces ~20M spurious 1.0 pairs and is the real
    # cause of the FUZZY_MATCH blow-up — not the embedding model.
    all_rows = [r for r in result if r["emb"]]
    brands = [
        (r["id"], r["name"], np.array(r["emb"]))
        for r in all_rows
        if r["name"] and str(r["name"]).strip().upper() != "UNKNOWN"
    ]
    skipped_unknown = len(all_rows) - len(brands)
    if skipped_unknown:
        print(f"    (excluding {skipped_unknown:,} UNKNOWN/unnamed brands from matching)")

    if not brands:
        print("    → No brand vectors found, skipping fuzzy match")
        return

    ids    = [b[0] for b in brands]
    names  = [b[1] for b in brands]
    embs   = np.stack([b[2] for b in brands])  # (N, 768)

    # Cosine similarity matrix — embeddings are already L2-normalized
    sim_matrix = embs @ embs.T  # (N, N)

    # Never let a brand match itself.
    np.fill_diagonal(sim_matrix, -1.0)

    # TOP-K nearest neighbours per brand — NOT a global similarity threshold.
    #
    # Brand text is just "brand <family>", which all-mpnet-base-v2 maps into a
    # tiny, dense region: ~43% of ALL brand pairs score >= 0.98. A fixed
    # threshold therefore matches tens of millions of pairs and blows up the
    # graph (~41M edges). Top-K bounds edges to ~N*K and keeps only each
    # brand's closest partners; BRAND_FUZZY_MIN_SIM drops dissimilar ones.
    N = len(ids)
    k = min(BRAND_FUZZY_TOP_K, N - 1)

    rows = []
    for i in range(N):
        sims = sim_matrix[i]
        topk = np.argpartition(sims, -k)[-k:]   # k highest sims (unordered)
        for j in topk:
            j = int(j)
            s = float(sims[j])
            if s >= BRAND_FUZZY_MIN_SIM and ids[i] != ids[j]:
                rows.append({
                    "id_a":       ids[i],
                    "id_b":       ids[j],
                    "confidence": round(s, 4),
                })

    if rows:
        cypher = """
        UNWIND $rows AS r
        MATCH (a:Brand {brand_id: r.id_a})
        MATCH (b:Brand {brand_id: r.id_b})
        MERGE (a)-[e:FUZZY_MATCH]->(b)
        SET e.confidence = r.confidence
        MERGE (b)-[f:FUZZY_MATCH]->(a)
        SET f.confidence = r.confidence
        """
        run_batch(session, cypher, rows)
        print(f"    → {len(rows):,} FUZZY_MATCH edges (top-{k} per brand, min_sim={BRAND_FUZZY_MIN_SIM})")
    else:
        print(f"    → No fuzzy brand pairs above floor {BRAND_FUZZY_MIN_SIM}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n── Loading embedding model ──────────────────────────────────")
    print(f"  Model: {EMBEDDING_MODEL}")
    # Use the Mac GPU (Metal/MPS) when available, else CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"  Device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("\n── Loading source data ──────────────────────────────────────")
    df_global = load_global_sku(GLOBAL_SKU_CSV)
    df_vendor = load_vendor_sku(VENDOR_SKU_XLSX)

    print("\n── Connecting to Neo4j ──────────────────────────────────────")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # ── Nodes ─────────────────────────────────────────────────────────
        print("\n── Seeding nodes ────────────────────────────────────────────")
        seed_global_skus(session, df_global, model)
        seed_vendor_skus(session, df_vendor, model)
        seed_brands(session, df_global, model)
        seed_package_types(session, df_global, model)
        seed_manufacturers(session, df_global, model)
        seed_suppliers(session, df_vendor, model)
        seed_product_classes(session, df_vendor, model)

        # ── Relationships ──────────────────────────────────────────────────
        print("\n── Creating relationships ────────────────────────────────────")
        create_belongs_to_brand(session, df_global)
        create_has_package(session, df_global)
        create_made_by(session, df_global)
        create_maps_to(session, df_global, df_vendor)
        create_supplied_by(session, df_vendor)
        create_in_class(session, df_vendor)
        create_fuzzy_brand_matches(session, model, df_global)

        # ── Summary ────────────────────────────────────────────────────────
        print("\n── Node counts ──────────────────────────────────────────────")
        for label in ["GlobalSKU", "VendorSKU", "Brand", "PackageType",
                       "Manufacturer", "Supplier", "ProductClass"]:
            count = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  {label}: {count:,}")

        print("\n── Relationship counts ───────────────────────────────────────")
        for rel in ["BELONGS_TO_BRAND", "HAS_PACKAGE", "MADE_BY",
                    "MAPS_TO", "SUPPLIED_BY", "IN_CLASS", "FUZZY_MATCH"]:
            count = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
            print(f"  {rel}: {count:,}")

    driver.close()
    print("\nSeeding complete. Run 03_reflection.py next.\n")


if __name__ == "__main__":
    main()
