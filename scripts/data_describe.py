"""
scripts/data_describe.py — Summary statistics for master_data in PostgreSQL.

Run from the repo root:
    python scripts/data_describe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from repo root (config, data.postgres_store)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.postgres_store import pg_session  # noqa: E402


def _col(cur, label: str, query: str, params=()) -> None:
    cur.execute(query, params)
    row = cur.fetchone()
    print(f"  {label:<45}: {row[0]:,}")


def _section(title: str) -> None:
    print()
    print(f"{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def run() -> None:
    with pg_session() as conn:
        with conn.cursor() as cur:

            # ── Overall ───────────────────────────────────────────────────
            _section("Overall  (master_data)")
            _col(cur, "Total SKU rows",
                 "SELECT COUNT(*) FROM master_data")
            _col(cur, "SKUs with blank brand_name",
                 "SELECT COUNT(*) FROM master_data WHERE brand_name IS NULL OR brand_name = ''")

            # ── Distinct counts ───────────────────────────────────────────
            _section("Distinct counts")
            for label, col in [
                ("brand_name",            "brand_name"),
                ("package_category_name", "package_category_name"),
                ("brand_family",          "brand_family"),
                ("package_name",          "package_name"),
                ("product_category",      "product_category"),
                ("manufacturer",          "manufacturer"),
                ("created_by",            "created_by"),
            ]:
                _col(cur, label,
                     f"SELECT COUNT(DISTINCT {col}) FROM master_data "
                     f"WHERE {col} IS NOT NULL AND {col} <> ''")

            # ── Status breakdown ──────────────────────────────────────────
            _section("Status breakdown")
            cur.execute("""
                SELECT COALESCE(status, '(null)'), COUNT(*)
                FROM master_data
                GROUP BY status
                ORDER BY COUNT(*) DESC
            """)
            for val, cnt in cur.fetchall():
                print(f"  {val:<20}: {cnt:,}")

            # ── Product category breakdown ────────────────────────────────
            _section("Product category breakdown")
            cur.execute("""
                SELECT COALESCE(NULLIF(product_category,''), '(blank)'), COUNT(*)
                FROM master_data
                GROUP BY product_category
                ORDER BY COUNT(*) DESC
            """)
            for val, cnt in cur.fetchall():
                print(f"  {val:<45}: {cnt:,}")

            # ── Manufacturer breakdown (top 10) ───────────────────────────
            _section("Manufacturer breakdown  (top 10)")
            cur.execute("""
                SELECT COALESCE(NULLIF(manufacturer,''), '(blank)'), COUNT(*)
                FROM master_data
                GROUP BY manufacturer
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)
            for val, cnt in cur.fetchall():
                print(f"  {val:<25}: {cnt:,}")

            # ── Duplicate (brand_name, package_category_name) combos ─────
            _section("Duplicate (brand_name, package_category_name) combos")
            cur.execute("""
                SELECT
                    COALESCE(brand_name, '(blank)'),
                    COALESCE(package_category_name, '(blank)'),
                    COUNT(*) AS cnt,
                    ARRAY_AGG(sku_id ORDER BY sku_id) AS sku_ids
                FROM master_data
                GROUP BY brand_name, package_category_name
                HAVING COUNT(*) > 1
                ORDER BY cnt DESC, brand_name
            """)
            rows = cur.fetchall()
            if not rows:
                print("  No duplicates found.")
            else:
                print(f"  {'brand_name':<40} {'package_category_name':<35} cnt  sku_ids")
                print(f"  {'─'*40} {'─'*35} {'─'*3}  {'─'*30}")
                for brand, pkg, cnt, skus in rows:
                    print(f"  {brand:<40} {pkg:<35} {cnt:>3}  {skus}")

            # ── Tenant summary (tenant_data) ──────────────────────────────
            _section("Tenant summary  (tenant_data)")
            cur.execute("SELECT COUNT(*) FROM tenant_data")
            print(f"  {'Total tenants':<45}: {cur.fetchone()[0]:,}")

            cur.execute("""
                SELECT tenant_name, warehouse, sku_count, brand_count,
                       supplier_count, matched_master_count
                FROM tenant_data
                ORDER BY sku_count DESC
            """)
            rows = cur.fetchall()
            if rows:
                print()
                hdr = f"  {'tenant_name':<25} {'warehouse':<20} {'skus':>6} {'brands':>7} {'suppliers':>10} {'matched':>8}"
                print(hdr)
                print(f"  {'─'*80}")
                for name, wh, skus, brands, suppliers, matched in rows:
                    print(f"  {(name or ''):<25} {(wh or ''):<20} {skus:>6,} {brands:>7,} {suppliers:>10,} {matched:>8,}")

    print()


if __name__ == "__main__":
    run()
