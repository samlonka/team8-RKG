"""Unit tests for product-level anomaly roll-up."""

from __future__ import annotations

from agents.product_risk import EcosystemNode, aggregate_product_risk


def test_aggregate_product_risk_healthy_sku_only():
    result = aggregate_product_risk(
        "6584", 0.15, [], brand_name="AQUA_WTR", package_type="28OZ PL 1/15",
    )
    assert result["classification"] == "healthy"
    assert result["anomaly_max"] == 0.15
    assert result["anomaly_mean"] == 0.15


def test_aggregate_product_risk_driver_from_neighbor():
    neighbors = [
        EcosystemNode(
            label="Pallet",
            entity_id="P1",
            display_name="P1 (failure)",
            anomaly=0.88,
            relationship="SCANNED_ON",
            direction="in",
            context="production scan failure",
        ),
        EcosystemNode(
            label="Brand",
            entity_id="B1",
            display_name="AQUA",
            anomaly=0.20,
            relationship="BELONGS_TO_BRAND",
            direction="out",
            context="brand link",
        ),
    ]
    result = aggregate_product_risk(
        "6584", 0.22, neighbors, brand_name="AQUA_WTR", package_type="28OZ PL 1/15",
    )
    assert result["classification"] == "high"
    assert result["anomaly_max"] == 0.88
    assert result["drivers"][0]["label"] == "Pallet"
    assert "Pallet" in result["summary"]


def test_aggregate_product_risk_no_scores():
    result = aggregate_product_risk("999", None, [])
    assert result["classification"] == "unknown"
    assert result["anomaly_max"] is None
