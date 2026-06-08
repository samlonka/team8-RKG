"""Tests for catalog_match intent parsing and pipeline routing."""

from __future__ import annotations

import pytest

from agents.catalog_intent import FEW_SHOT_EXAMPLES, is_catalog_question, parse_catalog_match


@pytest.mark.parametrize(
    "question,brand,package",
    [
        (
            "Is the product available in the master list: AQUA WATER 28OZ PL 1/15",
            "AQUA WATER",
            "28OZ PL 1/15",
        ),
        (
            "Can you identify if this SKU exists Brand name: AQUA_WTR, Package Type: 28OZ PL 1/15",
            "AQUA_WTR",
            "28OZ PL 1/15",
        ),
        (
            "Can you identify if this SKU exists Brand name: AQUA_WTR, Package Type: '28OZ PL 1/fifteen'",
            "AQUA_WTR",
            "28OZ PL 1/15",
        ),
        (
            "Can you identify if this SKU exists Brand name: GATBLT WSTW MB TM, Package Type: 16.9OZ PL 15/1",
            "GATBLT WSTW MB TM",
            "16.9OZ PL 15/1",
        ),
    ],
)
def test_parse_catalog_match(question: str, brand: str, package: str):
    parsed = parse_catalog_match(question)
    assert parsed is not None
    assert parsed["brand_name"] == brand
    assert parsed["package_type"] == package


def test_investigation_not_catalog():
    q = "Why are so many brands created as duplicates during customer import?"
    assert not is_catalog_question(q)
    assert parse_catalog_match(q) is None


def test_few_shot_examples_populated():
    assert len(FEW_SHOT_EXAMPLES) >= 4
