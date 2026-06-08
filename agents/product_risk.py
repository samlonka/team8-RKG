"""
agents/product_risk.py — Product-level anomaly roll-up for catalog matches.

Given a matched GlobalSKU (brand + package product), aggregate anomaly scores
across the SKU and its 1-hop ecosystem (Brand, TenantSKU, Pallet, MergeEvent,
TrainingImage, Customer, etc.) so analysts see *product* risk, not only node risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import ANOMALY_HIGH_RISK, REL_WEIGHTS

# Primary key field per Neo4j label (aligned with score_log.PK_MAP).
_PK: dict[str, str] = {
    "GlobalSKU":     "sku_id",
    "Brand":         "brand_id",
    "TenantSKU":     "tenant_sku_id",
    "PackageType":   "package_type_id",
    "TrainingImage": "image_id",
    "MergeEvent":    "merge_id",
    "Pallet":        "pallet_id",
    "Customer":      "customer_id",
    "Manufacturer":  "name",
    "Supplier":      "name",
    "ProductClass":  "name",
}

_ANOMALY_KEYS: tuple[str, ...] = (
    "anomaly_attn",
    "anomaly_baseline",
    "anomaly_score",
    "anomaly_dir",
    "anomaly_ensemble",
)

_EGO_CYPHER = """
MATCH (g:GlobalSKU {sku_id: $sku_id})
OPTIONAL MATCH (g)-[r_out]->(n_out)
WHERE n_out IS NOT NULL AND NOT n_out:GlobalSKU
WITH g, collect(DISTINCT {rel: type(r_out), node: n_out, dir: 'out'}) AS outs
OPTIONAL MATCH (n_in)-[r_in]->(g)
WHERE n_in IS NOT NULL AND NOT n_in:GlobalSKU
WITH g, outs, collect(DISTINCT {rel: type(r_in), node: n_in, dir: 'in'}) AS ins
WITH g, outs + ins AS edges
UNWIND edges AS e
WITH g, e
WHERE e.node IS NOT NULL
RETURN
  g.sku_id              AS sku_id,
  g.anomaly_attn        AS sku_anomaly,
  g.brand_name          AS sku_brand,
  g.package_category_name AS sku_package,
  labels(e.node)[0]     AS label,
  e.rel                 AS relationship,
  e.dir                 AS direction,
  properties(e.node)    AS props
"""


@dataclass
class EcosystemNode:
    label:         str
    entity_id:     str
    display_name:  str
    anomaly:       float | None
    relationship:  str
    direction:     str
    context:       str = ""


def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        v = float(val)
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None


def _anomaly_from_props(props: dict[str, Any]) -> float | None:
    for key in _ANOMALY_KEYS:
        v = _safe_float(props.get(key))
        if v is not None:
            return v
    return None


def _entity_id(label: str, props: dict[str, Any]) -> str:
    pk = _PK.get(label)
    if pk and props.get(pk) is not None:
        return str(props[pk])
    for k in ("name", "product_id", "sku_id", "id"):
        if props.get(k) is not None:
            return str(props[k])
    return "?"


def _display_name(label: str, props: dict[str, Any]) -> str:
    if label == "Brand":
        return str(props.get("brand_family") or props.get("brand_name") or _entity_id(label, props))
    if label == "TenantSKU":
        return str(props.get("product_id") or props.get("tenant_sku_id") or "?")
    if label == "Pallet":
        outcome = props.get("outcome")
        base = str(props.get("pallet_id") or "?")
        return f"{base} ({outcome})" if outcome else base
    if label == "MergeEvent":
        status = props.get("status")
        base = str(props.get("merge_id") or "?")
        return f"{base} ({status})" if status else base
    if label == "TrainingImage":
        return str(props.get("image_id") or "?")
    if label == "Customer":
        return str(props.get("customer_name") or props.get("customer_id") or "?")
    return _entity_id(label, props)


def _context_hint(label: str, props: dict[str, Any], relationship: str) -> str:
    if label == "Pallet" and props.get("outcome") == "failure":
        return "production scan failure"
    if label == "MergeEvent" and props.get("status") == "conflicted":
        return "conflicted merge history"
    if label == "TenantSKU" and props.get("match_method") == "fuzzy":
        return "fuzzy tenant mapping"
    if label == "Brand" and props.get("flag") == "duplicate":
        return "duplicate brand node"
    if relationship == "MAPS_TO":
        return "tenant maps to this GlobalSKU"
    if relationship == "SCANNED_ON":
        return "pallet scanned on this SKU"
    return relationship.replace("_", " ").lower()


def _parse_ecosystem_rows(rows: list[dict]) -> tuple[str, float | None, list[EcosystemNode]]:
    if not rows:
        return "", None, []

    sku_id = str(rows[0].get("sku_id") or "")
    sku_anomaly = _safe_float(rows[0].get("sku_anomaly"))
    neighbors: list[EcosystemNode] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        props = dict(row.get("props") or {})
        label = str(row.get("label") or "Unknown")
        rel = str(row.get("relationship") or "")
        direction = str(row.get("direction") or "")
        eid = _entity_id(label, props)
        key = (label, eid)
        if key in seen:
            continue
        seen.add(key)
        neighbors.append(EcosystemNode(
            label=label,
            entity_id=eid,
            display_name=_display_name(label, props),
            anomaly=_anomaly_from_props(props),
            relationship=rel,
            direction=direction,
            context=_context_hint(label, props, rel),
        ))

    return sku_id, sku_anomaly, neighbors


def aggregate_product_risk(
    sku_id: str,
    sku_anomaly: float | None,
    neighbors: list[EcosystemNode],
    *,
    brand_name: str = "",
    package_type: str = "",
) -> dict[str, Any]:
    """Roll up SKU + neighbor anomalies into a product-level risk summary."""
    entries: list[dict[str, Any]] = []

    if sku_anomaly is not None:
        entries.append({
            "label": "GlobalSKU",
            "entity_id": sku_id,
            "display_name": f"{brand_name or sku_id} / {package_type or '—'}".strip(" /"),
            "anomaly": round(sku_anomaly, 4),
            "relationship": "self",
            "context": "matched master SKU",
        })

    for n in neighbors:
        if n.anomaly is None:
            continue
        entries.append({
            "label": n.label,
            "entity_id": n.entity_id,
            "display_name": n.display_name,
            "anomaly": round(n.anomaly, 4),
            "relationship": n.relationship,
            "context": n.context,
        })

    if not entries:
        return {
            "sku_id": sku_id,
            "brand_name": brand_name,
            "package_type": package_type,
            "sku_anomaly": sku_anomaly,
            "anomaly_max": None,
            "anomaly_mean": None,
            "anomaly_weighted": None,
            "classification": "unknown",
            "neighbor_count": len(neighbors),
            "drivers": [],
            "summary": "No anomaly scores available for this product ecosystem.",
        }

    values = [e["anomaly"] for e in entries]
    anomaly_max = max(values)
    anomaly_mean = sum(values) / len(values)

    weighted_num = 0.0
    weighted_den = 0.0
    for e in entries:
        if e["label"] == "GlobalSKU":
            w = 1.0
        else:
            w = REL_WEIGHTS.get(e["relationship"], REL_WEIGHTS.get("_DEFAULT", 1.0))
        weighted_num += e["anomaly"] * w
        weighted_den += w
    anomaly_weighted = weighted_num / weighted_den if weighted_den else anomaly_mean

    if anomaly_max >= ANOMALY_HIGH_RISK:
        classification = "high"
    elif anomaly_max >= 0.50:
        classification = "moderate"
    else:
        classification = "healthy"

    drivers = sorted(
        [e for e in entries if e["label"] != "GlobalSKU"],
        key=lambda x: x["anomaly"],
        reverse=True,
    )[:5]

    top_driver = drivers[0] if drivers else None
    if top_driver and top_driver["anomaly"] > (sku_anomaly or 0) + 0.15:
        headline = (
            f"Product ecosystem risk is driven by {top_driver['label']} "
            f"{top_driver['entity_id']} ({top_driver['context']}, "
            f"anomaly={top_driver['anomaly']:.2f})."
        )
    elif anomaly_max == (sku_anomaly or anomaly_max):
        headline = (
            f"Product risk aligns with matched GlobalSKU "
            f"(anomaly={sku_anomaly:.2f})." if sku_anomaly is not None else
            "Product risk derived from matched GlobalSKU neighborhood."
        )
    else:
        headline = (
            f"Product anomaly max={anomaly_max:.2f}, mean={anomaly_mean:.2f} "
            f"across {len(entries)} scored entities."
        )

    return {
        "sku_id": sku_id,
        "brand_name": brand_name,
        "package_type": package_type,
        "sku_anomaly": round(sku_anomaly, 4) if sku_anomaly is not None else None,
        "anomaly_max": round(anomaly_max, 4),
        "anomaly_mean": round(anomaly_mean, 4),
        "anomaly_weighted": round(anomaly_weighted, 4),
        "classification": classification,
        "neighbor_count": len(neighbors),
        "scored_entity_count": len(entries),
        "drivers": drivers,
        "summary": headline,
    }


def compute_product_risk(session, sku_id: str) -> dict[str, Any] | None:
    """Fetch 1-hop ecosystem from Neo4j and return product risk dict."""
    if not sku_id:
        return None
    rows = list(session.run(_EGO_CYPHER, sku_id=str(sku_id)))
    if not rows:
        return None
    parsed_id, sku_anomaly, neighbors = _parse_ecosystem_rows(rows)
    if not parsed_id:
        return None
    brand = str(rows[0].get("sku_brand") or "")
    package = str(rows[0].get("sku_package") or "")
    return aggregate_product_risk(parsed_id, sku_anomaly, neighbors, brand_name=brand, package_type=package)


def product_risk_for_sku(driver, sku_id: str) -> dict[str, Any] | None:
    """Convenience wrapper when only the driver is available."""
    if driver is None or not sku_id:
        return None
    try:
        with driver.session() as session:
            return compute_product_risk(session, sku_id)
    except Exception as exc:
        print(f"[product_risk] Neo4j fetch failed for sku={sku_id}: {exc}")
        return None


def format_product_risk_for_prompt(product_risk: dict[str, Any] | None) -> str:
    if not product_risk or product_risk.get("anomaly_max") is None:
        return "Product ecosystem risk: not available (Neo4j scores missing)."
    lines = [
        "PRODUCT ECOSYSTEM RISK (SKU + 1-hop neighbors):",
        f"  classification: {product_risk.get('classification')}",
        f"  sku_anomaly: {product_risk.get('sku_anomaly')}",
        f"  product anomaly_max: {product_risk.get('anomaly_max')}",
        f"  product anomaly_mean: {product_risk.get('anomaly_mean')}",
        f"  summary: {product_risk.get('summary')}",
    ]
    drivers = product_risk.get("drivers") or []
    if drivers:
        lines.append("  top drivers:")
        for d in drivers[:3]:
            lines.append(
                f"    - {d['label']} {d['entity_id']} ({d['relationship']}): "
                f"anomaly={d['anomaly']:.3f} — {d.get('context', '')}"
            )
    return "\n".join(lines)
