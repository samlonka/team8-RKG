"""Unit tests for dynamic Doer chain assembly."""

from agents.doer import _assemble_chains
from agents.models import EntityNode


def _sku(sku_id: str, score: float) -> EntityNode:
    return EntityNode(
        entity_id=sku_id,
        label="GlobalSKU",
        display_name=f"SKU {sku_id}",
        anomaly_score=score,
        source="anomaly_rank",
    )


def test_risk_rank_single_sorted_chain():
    nodes = [_sku("100", 0.5), _sku("200", 0.9), _sku("300", 0.7)]
    chains, meta = _assemble_chains(nodes, "risk_rank", None)
    assert len(chains) == 1
    assert chains[0].chain_id == "dynamic_rank"
    scores = [n.anomaly_score for n in chains[0].path]
    assert scores == sorted(scores, reverse=True)
    assert meta["dynamic_rank"]["total"] == 3


def test_anchor_chain_preserves_evidence_order():
    evidence = [
        EntityNode("t1", "TenantSKU", "import", source="cypher"),
        EntityNode("2674", "GlobalSKU", "CORONA", anomaly_score=0.68, source="cypher"),
        EntityNode("p1", "Pallet", "scan failure", source="cypher"),
    ]
    ann_noise = [_sku("9999", 0.99)]
    chains, _ = _assemble_chains(
        evidence + ann_noise,
        "root_cause",
        "2674",
        ordered_evidence=evidence,
    )
    assert len(chains) == 1
    assert chains[0].chain_id == "anchor_2674"
    labels = [n.label for n in chains[0].path[:3]]
    assert labels == ["TenantSKU", "GlobalSKU", "Pallet"]
    assert chains[0].path[0].entity_id == "t1"
    assert chains[0].path[1].entity_id == "2674"


def test_extract_anchor_from_question():
    from agents.supervisor import extract_anchor_from_question

    assert extract_anchor_from_question("Explain GlobalSKU 2813 anomaly") == (
        "GlobalSKU",
        "2813",
    )
    assert extract_anchor_from_question("failures for sku 2674") == ("GlobalSKU", "2674")
