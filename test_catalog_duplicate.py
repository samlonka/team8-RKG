"""Tests for catalog duplicate (Q2) intent and lifecycle disambiguation."""

from agents.catalog_intent import is_catalog_duplicate_question, parse_catalog_match
from agents.lifecycle_doer import detect_scenario
from agents.supervisor import heuristic_parse

Q2 = "Are there any duplicate sku exists in the master list/database"
Q8 = (
    "Why are so many brands created as duplicates during customer import? "
    "Trace the root cause of brand mismatch across all SKUs."
)
Q4 = (
    "Why did model accuracy degrade after the recent customer import? "
    "Which brands were created as duplicates during import?"
)


def test_q2_catalog_duplicate_intent():
    assert is_catalog_duplicate_question(Q2)
    spec = heuristic_parse(Q2)
    assert spec is not None
    assert spec.task_type == "catalog_duplicate"
    assert spec.scenario_num is None
    assert detect_scenario(Q2) is None


def test_q8_still_scenario_1():
    assert not is_catalog_duplicate_question(Q8)
    assert detect_scenario(Q8) == 1


def test_q4_still_scenario_4():
    assert not is_catalog_duplicate_question(Q4)
    assert detect_scenario(Q4) == 4


def test_catalog_match_not_duplicate():
    q = "Can you identyfy the product/sku: AQUA Water, 28OZ PL 1/15"
    assert parse_catalog_match(q) is not None
    assert not is_catalog_duplicate_question(q)
