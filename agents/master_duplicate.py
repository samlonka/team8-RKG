"""
agents/master_duplicate.py — Detect duplicate SKUs in the master catalog.

Checks PostgreSQL master_data (preferred), CSV fallback, and Neo4j GlobalSKU nodes
for:
  - duplicate UPC → multiple sku_id rows
  - duplicate brand_name + package_category_name → multiple sku_id rows
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.models import CandidateChain, EntityNode

DUPLICATE_GROUP_LIMIT = 25


@dataclass
class DuplicateGroup:
    kind: str  # upc | brand_package
    key: str
    sku_ids: list[str] = field(default_factory=list)
    brand_name: str | None = None
    package_type: str | None = None
    upc: str | None = None

    @property
    def count(self) -> int:
        return len(self.sku_ids)


@dataclass
class DuplicateReport:
    source: str
    upc_groups: list[DuplicateGroup] = field(default_factory=list)
    brand_package_groups: list[DuplicateGroup] = field(default_factory=list)
    upc_groups_total: int = 0
    brand_package_groups_total: int = 0

    @property
    def total_groups(self) -> int:
        return self.upc_groups_total + self.brand_package_groups_total

    @property
    def total_groups_shown(self) -> int:
        return len(self.upc_groups) + len(self.brand_package_groups)

    @property
    def has_duplicates(self) -> bool:
        return self.total_groups > 0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "has_duplicates": self.has_duplicates,
            "total_groups": self.total_groups,
            "total_groups_shown": self.total_groups_shown,
            "upc_groups_total": self.upc_groups_total,
            "brand_package_groups_total": self.brand_package_groups_total,
            "upc_duplicate_groups": [
                {
                    "upc": g.upc,
                    "sku_ids": g.sku_ids,
                    "count": g.count,
                }
                for g in self.upc_groups
            ],
            "brand_package_duplicate_groups": [
                {
                    "brand_name": g.brand_name,
                    "package_type": g.package_type,
                    "sku_ids": g.sku_ids,
                    "count": g.count,
                }
                for g in self.brand_package_groups
            ],
        }


def _groups_from_dataframe(df, limit: int = DUPLICATE_GROUP_LIMIT) -> DuplicateReport:
    import pandas as pd

    report = DuplicateReport(source="postgres")
    if df is None or df.empty:
        return report

    work = df.copy()
    if "upc" in work.columns:
        upc_df = work[
            work["upc"].notna()
            & (work["upc"].astype(str).str.strip() != "")
        ]
        for upc, grp in upc_df.groupby("upc"):
            ids = [str(x) for x in grp["sku_id"].astype(str).tolist()]
            if len(ids) > 1:
                report.upc_groups.append(
                    DuplicateGroup(kind="upc", key=str(upc), upc=str(upc), sku_ids=ids)
                )
        report.upc_groups.sort(key=lambda g: g.count, reverse=True)
        report.upc_groups_total = len(report.upc_groups)
        report.upc_groups = report.upc_groups[:limit]

    brand_col = "brand_name" if "brand_name" in work.columns else None
    pkg_col = "package_category_name" if "package_category_name" in work.columns else None
    if brand_col and pkg_col:
        bp = work[work[brand_col].notna() & work[pkg_col].notna()]
        for (brand, pkg), grp in bp.groupby([brand_col, pkg_col]):
            ids = [str(x) for x in grp["sku_id"].astype(str).tolist()]
            if len(ids) > 1:
                report.brand_package_groups.append(
                    DuplicateGroup(
                        kind="brand_package",
                        key=f"{brand}|{pkg}",
                        brand_name=str(brand),
                        package_type=str(pkg),
                        sku_ids=ids,
                    )
                )
        report.brand_package_groups.sort(key=lambda g: g.count, reverse=True)
        report.brand_package_groups_total = len(report.brand_package_groups)
        report.brand_package_groups = report.brand_package_groups[:limit]

    return report


def find_duplicates_postgres(limit: int = DUPLICATE_GROUP_LIMIT) -> DuplicateReport | None:
    try:
        from data.postgres_store import load_master_dataframe, postgres_configured

        if not postgres_configured():
            return None
        df = load_master_dataframe()
        report = _groups_from_dataframe(df, limit=limit)
        report.source = "postgres"
        return report
    except Exception as exc:
        print(f"  [master_duplicate] Postgres scan failed: {exc}")
        return None


def find_duplicates_csv(limit: int = DUPLICATE_GROUP_LIMIT) -> DuplicateReport | None:
    try:
        from config import GLOBAL_SKU_CSV
        from data.postgres_store import load_master_csv

        df = load_master_csv(GLOBAL_SKU_CSV)
        if df is None or df.empty:
            return None
        report = _groups_from_dataframe(df, limit=limit)
        report.source = "csv"
        return report
    except Exception as exc:
        print(f"  [master_duplicate] CSV scan failed: {exc}")
        return None


def find_duplicates_neo4j(session, limit: int = DUPLICATE_GROUP_LIMIT) -> DuplicateReport:
    report = DuplicateReport(source="neo4j")

    upc_total = session.run(
        """
        MATCH (g:GlobalSKU)
        WHERE g.upc IS NOT NULL AND trim(g.upc) <> ''
        WITH g.upc AS upc, collect(DISTINCT g.sku_id) AS sku_ids
        WHERE size(sku_ids) > 1
        RETURN count(*) AS n
        """
    ).single()["n"]
    report.upc_groups_total = int(upc_total or 0)

    upc_rows = session.run(
        """
        MATCH (g:GlobalSKU)
        WHERE g.upc IS NOT NULL AND trim(g.upc) <> ''
        WITH g.upc AS upc, collect(DISTINCT g.sku_id) AS sku_ids
        WHERE size(sku_ids) > 1
        RETURN upc, sku_ids
        ORDER BY size(sku_ids) DESC
        LIMIT $limit
        """,
        limit=limit,
    ).data()
    for row in upc_rows:
        ids = [str(x) for x in row["sku_ids"]]
        report.upc_groups.append(
            DuplicateGroup(kind="upc", key=str(row["upc"]), upc=str(row["upc"]), sku_ids=ids)
        )

    bp_total = session.run(
        """
        MATCH (g:GlobalSKU)
        WHERE g.brand_name IS NOT NULL AND g.package_category_name IS NOT NULL
        WITH g.brand_name AS brand, g.package_category_name AS pkg,
             collect(DISTINCT g.sku_id) AS sku_ids
        WHERE size(sku_ids) > 1
        RETURN count(*) AS n
        """
    ).single()["n"]
    report.brand_package_groups_total = int(bp_total or 0)

    bp_rows = session.run(
        """
        MATCH (g:GlobalSKU)
        WHERE g.brand_name IS NOT NULL AND g.package_category_name IS NOT NULL
        WITH g.brand_name AS brand, g.package_category_name AS pkg,
             collect(DISTINCT g.sku_id) AS sku_ids
        WHERE size(sku_ids) > 1
        RETURN brand, pkg, sku_ids
        ORDER BY size(sku_ids) DESC
        LIMIT $limit
        """,
        limit=limit,
    ).data()
    for row in bp_rows:
        ids = [str(x) for x in row["sku_ids"]]
        report.brand_package_groups.append(
            DuplicateGroup(
                kind="brand_package",
                key=f"{row['brand']}|{row['pkg']}",
                brand_name=str(row["brand"]),
                package_type=str(row["pkg"]),
                sku_ids=ids,
            )
        )
    return report


def merge_reports(primary: DuplicateReport, secondary: DuplicateReport) -> DuplicateReport:
    """Prefer primary counts; fill from secondary if primary is empty."""
    if primary.has_duplicates:
        return primary
    if secondary.has_duplicates:
        merged = DuplicateReport(source=f"{primary.source}+{secondary.source}")
        merged.upc_groups = secondary.upc_groups or primary.upc_groups
        merged.brand_package_groups = (
            secondary.brand_package_groups or primary.brand_package_groups
        )
        return merged
    return primary


def scan_master_duplicates(session=None, limit: int = DUPLICATE_GROUP_LIMIT) -> DuplicateReport:
    report = find_duplicates_postgres(limit=limit)
    if report is None:
        report = find_duplicates_csv(limit=limit)
    if report is None:
        report = DuplicateReport(source="none")

    if session is not None:
        neo = find_duplicates_neo4j(session, limit=limit)
        report = merge_reports(report, neo)

    return report


def build_duplicate_chain(report: DuplicateReport) -> CandidateChain:
    """Turn duplicate groups into a Critic-friendly evidence chain."""
    path: list[EntityNode] = []

    if not report.has_duplicates:
        path.append(
            EntityNode(
                entity_id="NO_DUPLICATES",
                label="GlobalSKU",
                display_name="No duplicate UPC or brand+package groups in master catalog",
                properties={
                    "duplicate_scan_source": report.source,
                    "total_groups": 0,
                },
                anomaly_score=0.05,
                source="master_duplicate",
            )
        )
        return CandidateChain(
            chain_id="catalog_duplicate_clean",
            path=path,
            source="master_duplicate",
            llm_summary=(
                f"Master catalog scan ({report.source}): no duplicate UPC or "
                f"brand+package groups found."
            ),
        )

    for group in report.upc_groups[:10]:
        path.append(
            EntityNode(
                entity_id=f"UPC:{group.upc}",
                label="GlobalSKU",
                display_name=f"Duplicate UPC {group.upc} ({group.count} SKUs)",
                properties={
                    "duplicate_kind": "upc",
                    "upc": group.upc,
                    "sku_ids": group.sku_ids,
                    "count": group.count,
                    "duplicate_scan_source": report.source,
                },
                anomaly_score=min(0.95, 0.55 + 0.1 * group.count),
                source="master_duplicate",
            )
        )

    for group in report.brand_package_groups[:10]:
        path.append(
            EntityNode(
                entity_id=f"BP:{group.key[:40]}",
                label="GlobalSKU",
                display_name=(
                    f"Duplicate {group.brand_name} / {group.package_type} "
                    f"({group.count} SKUs)"
                ),
                properties={
                    "duplicate_kind": "brand_package",
                    "brand_name": group.brand_name,
                    "package_type": group.package_type,
                    "sku_ids": group.sku_ids,
                    "count": group.count,
                    "duplicate_scan_source": report.source,
                },
                anomaly_score=min(0.92, 0.50 + 0.08 * group.count),
                source="master_duplicate",
            )
        )

    summary = (
        f"Found {report.total_groups} duplicate group(s) in master catalog "
        f"({report.source}): {report.upc_groups_total} UPC, "
        f"{report.brand_package_groups_total} brand+package."
    )
    if report.total_groups_shown < report.total_groups:
        summary += (
            f" Showing top {report.total_groups_shown} groups in the UI "
            f"({report.upc_groups_total} UPC + {report.brand_package_groups_total} brand+package total)."
        )
    return CandidateChain(
        chain_id="catalog_duplicate",
        path=path,
        source="master_duplicate",
        llm_summary=summary,
    )
