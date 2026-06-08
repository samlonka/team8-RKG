"""Tests for physical dimension parsing and tie-breaking."""

from __future__ import annotations

import pytest

from agents.catalog_intent import parse_catalog_match
from agents.dim_match import (
    apply_dim_disambiguation,
    apply_dimension_ranking,
    dim_boost,
    dim_distance,
    parse_dimensions_from_text,
)


def test_parse_dimensions_explicit():
    q = "Weight: 10.5, Case Length (Inches): 12, Width: 8, Height: 10"
    dims = parse_dimensions_from_text(q)
    assert dims["weight"] == 10.5
    assert dims["length"] == 12.0
    assert dims["width"] == 8.0
    assert dims["height"] == 10.0


def test_parse_dimensions_lwh_compact():
    dims = parse_dimensions_from_text("L=12 W=8 H=10 inches")
    assert dims["length"] == 12.0
    assert dims["width"] == 8.0
    assert dims["height"] == 10.0


def test_parse_catalog_match_includes_dims():
    q = "Identify SKU AQUA WATER 28OZ PL 1/15 weight 10.5 length 12 width 8"
    parsed = parse_catalog_match(q)
    assert parsed is not None
    assert parsed["brand_name"] == "AQUA WATER"
    assert parsed["query_dims"]["weight"] == 10.5
    assert parsed["query_dims"]["length"] == 12.0
    assert parsed["query_dims"]["width"] == 8.0


def test_dim_distance_perfect_match():
    assert dim_distance({"weight": 10.0, "length": 12.0}, {"weight": 10.0, "length": 12.0}) == 0.0


def test_dim_boost_closer_wins():
    query = {"weight": 10.0, "length": 12.0}
    close = {"weight": 10.2, "length": 12.1}
    far = {"weight": 20.0, "length": 24.0}
    assert dim_boost(query, close) > dim_boost(query, far)


def test_apply_dimension_ranking_nudge_when_not_tied():
    candidates = [
        {"sku_id": "A", "composite_score": 0.84, "weight": 20.0, "length": 24.0},
        {"sku_id": "B", "composite_score": 0.78, "weight": 10.0, "length": 12.0},
    ]
    query = {"weight": 10.0, "length": 12.0}
    reranked, applied, mode = apply_dimension_ranking(
        candidates, query, score_key="composite_score",
    )
    assert applied is True
    assert mode == "nudge"
    assert reranked[0]["sku_id"] == "B"


def test_apply_dim_disambiguation_on_tie():
    candidates = [
        {"sku_id": "A", "composite_score": 0.82, "weight": 20.0, "length": 24.0},
        {"sku_id": "B", "composite_score": 0.81, "weight": 10.0, "length": 12.0},
    ]
    query = {"weight": 10.0, "length": 12.0}
    reranked, applied = apply_dim_disambiguation(candidates, query, score_key="composite_score")
    assert applied is True
    assert reranked[0]["sku_id"] == "B"


def test_apply_dim_disambiguation_skips_without_query_dims():
    candidates = [
        {"sku_id": "A", "composite_score": 0.82},
        {"sku_id": "B", "composite_score": 0.81},
    ]
    reranked, applied = apply_dim_disambiguation(candidates, {}, score_key="composite_score")
    assert applied is False
    assert reranked[0]["sku_id"] == "A"
