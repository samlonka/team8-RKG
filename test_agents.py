"""
test_agents.py — Agent-layer acceptance criteria (handbook §5: #6, #7, #8, #9, #10).

Runs the four-agent pipeline on the lifecycle graph and asserts:
  #6  Scenario 1 brand-mismatch chain validated (>= 0.65, multi-hop)
  #7  Scenario 2 surfaces a cross-source multi-signal SKU
  #8  Scenario 3 returns a top-20 at-risk ranking
  #9  Scenario 4 A/B: closed-world query = 0 rows, reflexive = validated chain
  #10 Critic rejects a weak chain (< 0.65)
Also checks the Supervisor parse (Bedrock Claude Opus 4.7 — skipped if AWS/Bedrock unavailable).

Requires Bedrock model access in us-east-1 (or BEDROCK_REGION).

Run:  python -m pytest test_agents.py -v
"""
import importlib

import pytest
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from agents.critic import CriticAgent
from agents.llm import LLMError, probe_bedrock
from agents.models import CandidateChain, EntityNode

scen = importlib.import_module("07_agent_scenarios")


def _require_bedrock():
    """Skip LLM-only tests when AWS/Bedrock is not configured."""
    try:
        probe_bedrock()
    except LLMError as e:
        pytest.skip(str(e))


def _session():
    d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return d, d.session()


def test_supervisor_parse():
    _require_bedrock()
    spec = scen.supervise("Why is this customer's model underperforming after import?")
    assert spec.task_type in ("root_cause", "risk_rank", "anomaly_explain")


def test_scenario1_brand_mismatch_validated():       # criterion #6
    d, s = _session()
    try:
        chain = scen.Doer(s).brand_mismatch_chain()
        res = CriticAgent().validate([chain])
        assert res.validated, "no validated chain"
        assert res.validated[0].confidence >= 0.65
        assert len(res.validated[0].path) >= 3        # multi-hop
    finally:
        s.close(); d.close()


def test_scenario2_multi_signal():                   # criterion #7
    d, s = _session()
    try:
        chain = scen.Doer(s).multi_signal_chain()
        assert len(chain.path) >= 1
        res = CriticAgent().validate([chain])
        assert res.validated, "multi-signal chain not validated"
        assert res.validated[0].avg_anomaly_score > 0.3
    finally:
        s.close(); d.close()


def test_scenario3_top20_rank():                     # criterion #8
    d, s = _session()
    try:
        ranked = scen.Doer(s).risk_rank(20)
        assert len(ranked) == 20
        scores = [v for _, v in ranked]
        assert scores == sorted(scores, reverse=True)
    finally:
        s.close(); d.close()


def test_scenario4_ab_comparison():                  # criterion #9
    d, s = _session()
    try:
        doer = scen.Doer(s)
        closed = doer.closed_world_brand_dupes()
        res = CriticAgent().validate([doer.brand_mismatch_chain()])
        assert len(closed) == 0          # flag never set -> closed world blind
        assert res.validated             # reflexive KG finds it
    finally:
        s.close(); d.close()


def test_critic_rejects_weak_chain():                # criterion #10
    weak = CandidateChain(
        chain_id="weak", source="cypher",
        path=[EntityNode(entity_id="x", label="GlobalSKU", display_name="x",
                         anomaly_score=0.05)])
    res = CriticAgent().validate([weak])
    assert not res.validated
    assert res.rejected and res.rejected[0].confidence < 0.65


def test_scenario5_shared_sku_validated():
    d, s = _session()
    try:
        chain = scen.Doer(s).shared_sku_chain()
        assert len(chain.path) >= 3
        res = CriticAgent().validate([chain])
        assert res.validated, "shared-SKU chain must validate"
        assert res.validated[0].confidence >= 0.65
    finally:
        s.close(); d.close()


def test_scenario6_auto_map_validated():
    d, s = _session()
    try:
        chain = scen.Doer(s).auto_map_chain()
        assert len(chain.path) >= 3
        res = CriticAgent().validate([chain])
        assert res.validated, "auto-map chain must validate"
        assert res.validated[0].confidence >= 0.65
    finally:
        s.close(); d.close()
