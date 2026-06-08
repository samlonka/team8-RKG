"""
PostgreSQL persistence for master catalog and tenant SKU imports.

Tables:
  master_data      — global SKU catalog (data/vor_sku_data.csv)
  tenant_sku_data  — per-tenant product rows (Excel imports)
  tenant_data      — one row per tenant/warehouse with aggregate stats
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from config import (
    AWS_REGION,
    GLOBAL_SKU_CSV,
    POSTGRES_CONNECT_TIMEOUT,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_SECRET_ID,
    POSTGRES_SSLMODE,
    POSTGRES_USER,
    TENANT_SKU_XLSX,
)
from data.master_loader import (
    build_global_upc_lookup,
    enrich_global_dataframe,
    match_tenant_to_global,
    slug_id,
)

ROOT = Path(__file__).resolve().parent.parent

MASTER_COLUMNS = [
    "sku_id", "vor_reference_number", "upc", "status", "package_category_name",
    "brand_name", "short_description", "long_description", "weight", "height",
    "length", "width", "manufacturer", "is_imaged_on_training_station",
    "is_imaged_on_wrapper", "product_category", "is_unverifiable_sku",
    "units_per_case", "primary_reference_number", "is_inserted_through_picklist_api",
    "creation_date", "created_by", "last_updated_date", "last_updated_by",
    "brand_family", "sku_packages_per_case", "date_imaged_on_wrapper",
    "is_labeled_by_ul", "date_labeled_by_ul", "is_weight_verified_on_scale",
    "date_weight_was_verified", "each_upc", "case_upc", "unit_upc", "package_upc",
    "is_review_needed", "package_name", "brand_id", "package_type_id",
    "upc_missing", "upc_aliases",
]

TENANT_SKU_COLUMNS = [
    "tenant_id", "tenant_sku_id", "product_id", "warehouse", "customer",
    "product_description", "brand", "supplier", "product_class", "units_per_case",
    "unit_weight", "case_length", "case_width", "case_height",
    "case_upc", "retail_upc", "eaches_upc", "match_method",
    "pkg_qty", "pkg_size", "pkg_unit", "pkg_container",
    "matched_global_sku_id", "match_method_global", "source_file", "extra_fields",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def postgres_configured() -> bool:
    """True when minimum connection settings are present in .env."""
    return bool(POSTGRES_HOST and POSTGRES_DB and POSTGRES_USER)


def _resolve_password() -> str:
    if POSTGRES_PASSWORD:
        return POSTGRES_PASSWORD
    if not POSTGRES_SECRET_ID:
        raise RuntimeError(
            "PostgreSQL password not configured. Set POSTGRES_PASSWORD in .env "
            "or set POSTGRES_SECRET_ID with valid AWS credentials."
        )
    import boto3

    sm = boto3.client("secretsmanager", region_name=AWS_REGION)
    payload = json.loads(sm.get_secret_value(SecretId=POSTGRES_SECRET_ID)["SecretString"])
    return payload["password"]


def get_connection():
    """Open a psycopg2 connection using settings from .env / config."""
    if not postgres_configured():
        raise RuntimeError(
            "PostgreSQL is not configured. Set POSTGRES_HOST, POSTGRES_DB, "
            "and POSTGRES_USER in .env (see .env.example)."
        )
    kwargs: dict[str, Any] = {
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "dbname": POSTGRES_DB,
        "user": POSTGRES_USER,
        "password": _resolve_password(),
        "connect_timeout": POSTGRES_CONNECT_TIMEOUT,
    }
    if POSTGRES_SSLMODE:
        kwargs["sslmode"] = POSTGRES_SSLMODE
    return psycopg2.connect(**kwargs)


@contextmanager
def pg_session() -> Iterator[Any]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def tenant_id_from_warehouse(warehouse: str) -> str:
    """Stable tenant key from warehouse / customer name."""
    return slug_id("TEN", str(warehouse or "UNKNOWN").strip())


def _clean_value(val: Any) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, bool):
        return val
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "nat"):
        return None
    return s


def _bool_val(val: Any) -> bool | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "t")


def _float_val(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def ensure_schema(conn) -> None:
    """Create catalog tables if they do not exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS master_data (
        sku_id                      TEXT PRIMARY KEY,
        vor_reference_number        TEXT,
        upc                         TEXT,
        status                      TEXT,
        package_category_name       TEXT,
        brand_name                  TEXT,
        short_description           TEXT,
        long_description            TEXT,
        weight                      DOUBLE PRECISION,
        height                      DOUBLE PRECISION,
        length                      DOUBLE PRECISION,
        width                       DOUBLE PRECISION,
        manufacturer                TEXT,
        is_imaged_on_training_station BOOLEAN,
        is_imaged_on_wrapper        BOOLEAN,
        product_category            TEXT,
        is_unverifiable_sku         BOOLEAN,
        units_per_case              DOUBLE PRECISION,
        primary_reference_number    TEXT,
        is_inserted_through_picklist_api BOOLEAN,
        creation_date               TEXT,
        created_by                  TEXT,
        last_updated_date           TEXT,
        last_updated_by             TEXT,
        brand_family                TEXT,
        sku_packages_per_case       TEXT,
        date_imaged_on_wrapper      TEXT,
        is_labeled_by_ul            BOOLEAN,
        date_labeled_by_ul          TEXT,
        is_weight_verified_on_scale BOOLEAN,
        date_weight_was_verified    TEXT,
        each_upc                    TEXT,
        case_upc                    TEXT,
        unit_upc                    TEXT,
        package_upc                 TEXT,
        is_review_needed            BOOLEAN,
        package_name                TEXT,
        brand_id                    TEXT,
        package_type_id             TEXT,
        upc_missing                 BOOLEAN,
        upc_aliases                 JSONB,
        loaded_at                   TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS tenant_data (
        tenant_id                   TEXT PRIMARY KEY,
        tenant_name                 TEXT NOT NULL,
        warehouse                   TEXT,
        sku_count                   INTEGER NOT NULL DEFAULT 0,
        brand_count                 INTEGER NOT NULL DEFAULT 0,
        supplier_count              INTEGER NOT NULL DEFAULT 0,
        product_class_count         INTEGER NOT NULL DEFAULT 0,
        matched_master_count        INTEGER NOT NULL DEFAULT 0,
        master_brand_count          INTEGER NOT NULL DEFAULT 0,
        source_files                JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS tenant_sku_data (
        tenant_id                   TEXT NOT NULL REFERENCES tenant_data(tenant_id) ON DELETE CASCADE,
        tenant_sku_id               TEXT NOT NULL,
        product_id                  TEXT NOT NULL,
        warehouse                   TEXT,
        customer                    TEXT,
        product_description         TEXT,
        brand                       TEXT,
        supplier                    TEXT,
        product_class               TEXT,
        units_per_case              DOUBLE PRECISION,
        unit_weight                 DOUBLE PRECISION,
        case_length                 DOUBLE PRECISION,
        case_width                  DOUBLE PRECISION,
        case_height                 DOUBLE PRECISION,
        case_upc                    TEXT,
        retail_upc                  TEXT,
        eaches_upc                  TEXT,
        match_method                TEXT,
        pkg_qty                     INTEGER,
        pkg_size                    DOUBLE PRECISION,
        pkg_unit                    TEXT,
        pkg_container               TEXT,
        matched_global_sku_id       TEXT REFERENCES master_data(sku_id) ON DELETE SET NULL,
        match_method_global         TEXT,
        source_file                 TEXT,
        extra_fields                JSONB,
        created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (tenant_id, tenant_sku_id)
    );

    CREATE INDEX IF NOT EXISTS idx_tenant_sku_product ON tenant_sku_data (product_id);
    CREATE INDEX IF NOT EXISTS idx_tenant_sku_global  ON tenant_sku_data (matched_global_sku_id);
    CREATE INDEX IF NOT EXISTS idx_master_brand       ON master_data (brand_family);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)


def _master_row(row: pd.Series) -> tuple:
    aliases = row.get("upc_aliases") or []
    if isinstance(aliases, str):
        aliases = json.loads(aliases) if aliases.startswith("[") else [aliases]
    return (
        _clean_value(row.get("sku_id")),
        _clean_value(row.get("vor_reference_number")),
        _clean_value(row.get("upc")),
        _clean_value(row.get("status")),
        _clean_value(row.get("package_category_name")),
        _clean_value(row.get("brand_name")),
        _clean_value(row.get("short_description")),
        _clean_value(row.get("long_description")),
        _float_val(row.get("weight")),
        _float_val(row.get("height")),
        _float_val(row.get("length")),
        _float_val(row.get("width")),
        _clean_value(row.get("manufacturer")),
        _bool_val(row.get("is_imaged_on_training_station")),
        _bool_val(row.get("is_imaged_on_wrapper")),
        _clean_value(row.get("product_category")),
        _bool_val(row.get("is_unverifiable_sku")),
        _float_val(row.get("units_per_case")),
        _clean_value(row.get("primary_reference_number")),
        _bool_val(row.get("is_inserted_through_picklist_api")),
        _clean_value(row.get("creation_date")),
        _clean_value(row.get("created_by")),
        _clean_value(row.get("last_updated_date")),
        _clean_value(row.get("last_updated_by")),
        _clean_value(row.get("brand_family")),
        _clean_value(row.get("sku_packages_per_case")),
        _clean_value(row.get("Date_imaged_on_wrapper") or row.get("date_imaged_on_wrapper")),
        _bool_val(row.get("is_labeled_by_UL") or row.get("is_labeled_by_ul")),
        _clean_value(row.get("date_labeled_by_UL") or row.get("date_labeled_by_ul")),
        _bool_val(row.get("is_weight_verified_on_scale")),
        _clean_value(row.get("date_weight_was_verified")),
        _clean_value(row.get("each_upc")),
        _clean_value(row.get("case_upc")),
        _clean_value(row.get("unit_upc")),
        _clean_value(row.get("package_upc")),
        _bool_val(row.get("is_review_needed")),
        _clean_value(row.get("package_name")),
        _clean_value(row.get("brand_id")),
        _clean_value(row.get("package_type_id")),
        _bool_val(row.get("upc_missing")),
        Json(list(aliases)),
    )


def load_master_csv(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or GLOBAL_SKU_CSV)
    if not path.is_absolute():
        path = ROOT / path
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip('"').strip() for c in df.columns]
    df = df.drop_duplicates(subset=["sku_id"], keep="first")
    return enrich_global_dataframe(df)


def upsert_master_data(conn, df: pd.DataFrame) -> int:
    """Upsert all master catalog rows. Returns row count."""
    rows = [_master_row(r) for _, r in df.iterrows()]
    sql = f"""
    INSERT INTO master_data ({", ".join(MASTER_COLUMNS)}, loaded_at, updated_at)
    VALUES %s
    ON CONFLICT (sku_id) DO UPDATE SET
        {", ".join(f"{c} = EXCLUDED.{c}" for c in MASTER_COLUMNS if c != "sku_id")},
        updated_at = now()
    """
    template = "(" + ", ".join(["%s"] * len(MASTER_COLUMNS)) + ", now(), now())"
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template=template, page_size=500)
    return len(rows)


def _tenant_sku_records(
    df: pd.DataFrame,
    source_file: str,
    upc_lookup: dict[str, tuple[str, str]] | None = None,
    valid_master_skus: set[str] | None = None,
) -> list[tuple]:
    known = set(TENANT_SKU_COLUMNS) - {"tenant_id", "extra_fields", "source_file",
                                         "matched_global_sku_id", "match_method_global"}
    records: list[tuple] = []
    for _, row in df.iterrows():
        warehouse = _clean_value(row.get("warehouse") or row.get("customer")) or "UNKNOWN"
        tid = tenant_id_from_warehouse(warehouse)
        tenant_sku_id = _clean_value(row.get("tenant_sku_id") or row.get("product_id"))
        product_id = _clean_value(row.get("product_id") or tenant_sku_id)

        matched_sku, match_method = (None, None)
        if upc_lookup:
            matched_sku, match_method = match_tenant_to_global(row.to_dict(), upc_lookup)
            if matched_sku and valid_master_skus is not None:
                if matched_sku not in valid_master_skus:
                    matched_sku, match_method = None, None

        extra = {}
        for col in df.columns:
            if col not in known and col not in ("tenant_sku_id", "customer"):
                v = _clean_value(row.get(col))
                if v is not None:
                    extra[col] = v

        records.append((
            tid,
            tenant_sku_id,
            product_id,
            warehouse,
            _clean_value(row.get("customer") or warehouse),
            _clean_value(row.get("product_description")),
            _clean_value(row.get("brand")),
            _clean_value(row.get("supplier")),
            _clean_value(row.get("product_class")),
            _float_val(row.get("units_per_case")),
            _float_val(row.get("unit_weight")),
            _float_val(row.get("case_length")),
            _float_val(row.get("case_width")),
            _float_val(row.get("case_height")),
            _clean_value(row.get("case_upc")),
            _clean_value(row.get("retail_upc")),
            _clean_value(row.get("eaches_upc")),
            _clean_value(row.get("match_method")),
            int(_float_val(row.get("pkg_qty")) or 0) if _float_val(row.get("pkg_qty")) else None,
            _float_val(row.get("pkg_size")),
            _clean_value(row.get("pkg_unit")),
            _clean_value(row.get("pkg_container")),
            matched_sku,
            match_method,
            source_file,
            Json(extra) if extra else None,
        ))
    return records


def upsert_tenant_sku_data(
    conn,
    df: pd.DataFrame,
    source_file: str,
    master_df: pd.DataFrame | None = None,
) -> int:
    """Upsert tenant SKU rows and refresh tenant_data aggregates."""
    if master_df is None:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM master_data LIMIT 1")
            has_master = cur.fetchone() is not None
        master_df = load_master_csv() if has_master else None

    upc_lookup = build_global_upc_lookup(master_df) if master_df is not None else {}
    valid_skus = set(master_df["sku_id"].astype(str)) if master_df is not None else None
    rows = _tenant_sku_records(df, source_file, upc_lookup, valid_skus)

    # Ensure tenant_data stubs exist before FK insert
    tenant_names: dict[str, str] = {}
    for r in rows:
        tenant_names[r[0]] = r[3] or r[0]

    with conn.cursor() as cur:
        for tid, name in tenant_names.items():
            cur.execute(
                """
                INSERT INTO tenant_data (tenant_id, tenant_name, warehouse, source_files)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (tenant_id) DO UPDATE SET
                    tenant_name = EXCLUDED.tenant_name,
                    warehouse   = EXCLUDED.warehouse,
                    updated_at  = now()
                """,
                (tid, name, name, json.dumps([source_file])),
            )

    cols = TENANT_SKU_COLUMNS
    sql = f"""
    INSERT INTO tenant_sku_data ({", ".join(cols)}, created_at, updated_at)
    VALUES %s
    ON CONFLICT (tenant_id, tenant_sku_id) DO UPDATE SET
        {", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("tenant_id", "tenant_sku_id"))},
        updated_at = now()
    """
    template = "(" + ", ".join(["%s"] * len(cols)) + ", now(), now())"
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template=template, page_size=500)

    refresh_tenant_data(conn, tenant_ids=list(tenant_names.keys()), source_file=source_file)
    return len(rows)


def refresh_tenant_data(
    conn,
    tenant_ids: list[str] | None = None,
    source_file: str | None = None,
) -> int:
    """
    Recompute tenant_data aggregates from tenant_sku_data joined to master_data.
    """
    filter_sql = ""
    params: list[Any] = []
    if tenant_ids:
        filter_sql = "WHERE t.tenant_id = ANY(%s)"
        params.append(tenant_ids)

    sql = f"""
    WITH stats AS (
        SELECT
            t.tenant_id,
            MAX(t.warehouse) AS warehouse,
            MAX(t.customer)  AS tenant_name,
            COUNT(*)         AS sku_count,
            COUNT(DISTINCT NULLIF(t.brand, ''))         AS brand_count,
            COUNT(DISTINCT NULLIF(t.supplier, ''))      AS supplier_count,
            COUNT(DISTINCT NULLIF(t.product_class, '')) AS product_class_count,
            COUNT(DISTINCT t.matched_global_sku_id)     AS matched_master_count,
            COUNT(DISTINCT m.brand_family) FILTER (
                WHERE m.brand_family IS NOT NULL AND m.brand_family <> ''
            ) AS master_brand_count
        FROM tenant_sku_data t
        LEFT JOIN master_data m ON m.sku_id = t.matched_global_sku_id
        {filter_sql}
        GROUP BY t.tenant_id
    )
    UPDATE tenant_data td SET
        tenant_name          = s.tenant_name,
        warehouse            = s.warehouse,
        sku_count            = s.sku_count,
        brand_count          = s.brand_count,
        supplier_count       = s.supplier_count,
        product_class_count  = s.product_class_count,
        matched_master_count = s.matched_master_count,
        master_brand_count   = s.master_brand_count,
        source_files         = CASE
            WHEN %s IS NULL THEN td.source_files
            WHEN td.source_files @> to_jsonb(%s::text)
            THEN td.source_files
            ELSE td.source_files || to_jsonb(%s::text)
        END,
        updated_at = now()
    FROM stats s
    WHERE td.tenant_id = s.tenant_id
    """
    with conn.cursor() as cur:
        cur.execute(sql, params + [source_file, source_file, source_file])
        updated = cur.rowcount
    return updated


def load_tenant_xlsx(path: str | Path) -> pd.DataFrame:
    """Load tenant Excel using the same normalization as 02_seed_data."""
    import importlib

    seed = importlib.import_module("02_seed_data")
    return seed.load_vendor_sku(str(path))


def sync_tenant_xlsx(path: str | Path, conn=None) -> dict[str, int]:
    """Load an Excel file and upsert tenant_sku_data + tenant_data."""
    path = Path(path)
    df = load_tenant_xlsx(path)
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        ensure_schema(conn)
        counts = upsert_tenant_sku_data(conn, df, source_file=str(path))
        if own_conn:
            conn.commit()
        return {"tenant_sku_rows": counts, "tenants": df["warehouse"].nunique()}
    except Exception:
        if own_conn:
            conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()


def load_all(
    master_path: str | Path | None = None,
    tenant_path: str | Path | None = None,
    wipe: bool = False,
) -> dict[str, int]:
    """
    Full initial load: schema + master_data + sample tenant Excel.
    """
    master_path = master_path or GLOBAL_SKU_CSV
    tenant_path = tenant_path or TENANT_SKU_XLSX

    with pg_session() as conn:
        ensure_schema(conn)
        if wipe:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE tenant_sku_data, tenant_data, master_data CASCADE")

        master_df = load_master_csv(master_path)
        n_master = upsert_master_data(conn, master_df)

        n_tenant = 0
        n_tenants = 0
        if Path(ROOT / tenant_path if not Path(tenant_path).is_absolute() else tenant_path).exists():
            tpath = ROOT / tenant_path if not Path(tenant_path).is_absolute() else Path(tenant_path)
            df_tenant = load_tenant_xlsx(tpath)
            n_tenant = upsert_tenant_sku_data(conn, df_tenant, str(tpath), master_df=master_df)
            n_tenants = df_tenant["warehouse"].nunique()

    return {
        "master_rows": n_master,
        "tenant_sku_rows": n_tenant,
        "tenants": n_tenants,
    }


def check_connection() -> tuple[bool, str]:
    if not postgres_configured():
        return False, "PostgreSQL not configured — set POSTGRES_* in .env"
    try:
        with pg_session() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True, f"Connected to {POSTGRES_DB}@{POSTGRES_HOST}"
    except Exception as exc:
        return False, str(exc)


def table_counts(conn) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in ("master_data", "tenant_data", "tenant_sku_data"):
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            out[table] = cur.fetchone()[0]
    return out
