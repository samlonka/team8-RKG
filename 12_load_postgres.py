"""
12_load_postgres.py — Load master + tenant data into PostgreSQL (AWS RDS)

Creates tables master_data, tenant_data, tenant_sku_data and loads:
  - data/vor_sku_data.csv  → master_data
  - data/SKU_Export.xlsx   → tenant_sku_data + tenant_data

Connection settings are read from .env (see .env.example).
When POSTGRES_PASSWORD is empty, the password is fetched from RDS Secrets Manager
using POSTGRES_SECRET_ID and AWS credentials.

Usage:
    python 12_load_postgres.py
    python 12_load_postgres.py --wipe
    python 12_load_postgres.py --tenant-only data/new_client.xlsx
"""

from __future__ import annotations

import argparse
import sys

from data.postgres_store import (
    check_connection,
    load_all,
    pg_session,
    sync_tenant_xlsx,
    table_counts,
)


def main():
    parser = argparse.ArgumentParser(description="Load SKU data into PostgreSQL")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Truncate catalog tables before loading",
    )
    parser.add_argument(
        "--tenant-only",
        metavar="XLSX",
        help="Upsert a tenant Excel file without reloading master_data",
    )
    args = parser.parse_args()

    ok, msg = check_connection()
    if not ok:
        print(f"PostgreSQL connection failed: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"PostgreSQL: {msg}")

    if args.tenant_only:
        result = sync_tenant_xlsx(args.tenant_only)
        print(f"  Tenant SKUs upserted: {result['tenant_sku_rows']:,}")
        print(f"  Tenants touched:      {result['tenants']:,}")
    else:
        result = load_all(wipe=args.wipe)
        print(f"  master_data rows:     {result['master_rows']:,}")
        print(f"  tenant_sku_data rows: {result['tenant_sku_rows']:,}")
        print(f"  tenants:              {result['tenants']:,}")

    with pg_session() as conn:
        counts = table_counts(conn)
    print("\n── Table counts ─────────────────────────────────────")
    for table, n in counts.items():
        print(f"  {table:<20} {n:>8,}")


if __name__ == "__main__":
    main()
