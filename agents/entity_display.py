"""
agents/entity_display.py — Human-readable entity lines for demo / inference logs.
"""

from __future__ import annotations

from agents.models import EntityNode


def brand_display(brand_name: str | None, brand_family: str | None = None) -> str:
    """Prefer brand_name; include brand_family when both differ."""
    bn = str(brand_name or "").strip()
    bf = str(brand_family or "").strip()
    if bn.upper() in ("", "UNKNOWN", "NAN"):
        bn = ""
    if bf.upper() in ("", "UNKNOWN", "NAN"):
        bf = ""
    if bn and bf and bn != bf:
        return f"{bn} / {bf}"
    return bn or bf or "—"


def sku_summary(
    sku_id: str,
    brand_name: str | None = None,
    brand_family: str | None = None,
    package_type: str | None = None,
    detail: str = "",
) -> str:
    """Compact SKU line for causal chains and scenario logs."""
    brand = brand_display(brand_name, brand_family)
    pkg = str(package_type or "").strip() or "—"
    line = f"SKU {sku_id} | brand={brand} | pkg={pkg}"
    if detail:
        line = f"{line} | {detail}"
    return line


def format_entity_node(node: EntityNode) -> str:
    """Format one chain entity for pipeline / scenario output."""
    p = node.properties or {}
    score = f"{node.anomaly_score:.3f}" if node.anomaly_score is not None else "—"

    if node.label == "GlobalSKU":
        return (
            f"[GlobalSKU] {sku_summary(node.entity_id, p.get('brand_name'), p.get('brand_family'), p.get('package_type'))} "
            f"score={score}"
        )

    brand = p.get("brand_name") or p.get("brand") or p.get("brand_family") or ""
    pkg = p.get("package_type") or p.get("package") or ""
    extras = []
    if brand:
        extras.append(f"brand={brand}")
    if pkg:
        extras.append(f"pkg={pkg}")
    extra = (" " + " ".join(extras)) if extras else ""
    return f"[{node.label}] id={node.entity_id}{extra} | {node.display_name} | score={score}"
