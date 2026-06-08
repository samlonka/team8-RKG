"""Unit tests for Supervisor heuristic intent classification (no Bedrock)."""

from agents.lifecycle_doer import SCENARIO_QUESTIONS
from agents.supervisor import heuristic_parse


def test_catalog_match_intent():
    spec = heuristic_parse(
        "Is the product available in the master list: AQUA WATER 28OZ PL 1/15"
    )
    assert spec is not None
    assert spec.task_type == "catalog_match"
    assert spec.brand_name == "AQUA WATER"
    assert spec.package_type == "28OZ PL 1/15"
    assert spec.scenario_num is None


def test_lifecycle_scenario_intents():
    expectations = {
        1: "root_cause",
        2: "anomaly_explain",
        3: "risk_rank",
        4: "root_cause",
        5: "anomaly_explain",
        6: "root_cause",
    }
    for num, task_type in expectations.items():
        spec = heuristic_parse(SCENARIO_QUESTIONS[num])
        assert spec is not None, f"scenario {num} not detected"
        assert spec.scenario_num == num
        assert spec.task_type == task_type


def test_scenario4_keyword_routing():
    spec = heuristic_parse(
        "Why did model accuracy degrade after the recent customer import?"
    )
    assert spec is not None
    assert spec.scenario_num == 4
    assert spec.task_type == "root_cause"


def test_open_ended_returns_none_without_bedrock():
    assert heuristic_parse("Tell me something random about bananas") is None
