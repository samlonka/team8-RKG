"""
07_agent_scenarios.py — Demo scenario runner (thin CLI over agents/lifecycle_doer.py).

Prefer `python 04_agent_pipeline.py --demo` for the full four-agent orchestration.
This script remains for quick scenario checks and pytest imports.
"""
from __future__ import annotations

import argparse

from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from agents.critic import CriticAgent
from agents.lifecycle_doer import LifecycleDoer, SCENARIO_QUESTIONS
from agents.llm import get_llm
from agents.models import QuerySpec


def supervise(question: str) -> QuerySpec:
    """NL → QuerySpec (Bedrock when available, else heuristic)."""
    llm = get_llm()
    spec = llm.json(
        f'Question: "{question}"\n'
        "Classify it for a SKU-data-health knowledge graph. Return JSON: "
        '{"task_type": one of root_cause|risk_rank|anomaly_explain, '
        '"entity_types": [labels], "traversal_depth": int 1-4}.',
        system="You are a query orchestrator. Reply with ONLY a JSON object.",
        max_tokens=200,
    )
    if spec and spec.get("task_type") in ("root_cause", "risk_rank", "anomaly_explain"):
        et = [e for e in (spec.get("entity_types") or []) if isinstance(e, str)]
        return QuerySpec(
            question=question,
            task_type=spec["task_type"],
            entity_types=et or ["GlobalSKU"],
            traversal_depth=int(spec.get("traversal_depth", 3)),
        )
    q = question.lower()
    tt = (
        "root_cause"
        if any(w in q for w in ["why", "cause", "underperform", "trace"])
        else "risk_rank"
        if any(w in q for w in ["risk", "rank", "top", "at-risk", "before training"])
        else "anomaly_explain"
    )
    return QuerySpec(question=question, task_type=tt, entity_types=["GlobalSKU"])


# Backward-compatible alias used by test_agents.py / test_full_criteria.py
Doer = LifecycleDoer


def show(title, chain, critic):
    res = critic.validate([chain])
    print(f"\n== {title} ==")
    if res.validated:
        c = res.validated[0]
        print(
            f"  VALIDATED  confidence={c.confidence:.3f} "
            f"(temporal={c.temporal_validity:.2f} density={c.evidence_density:.2f} "
            f"anomaly={c.avg_anomaly_score:.2f})  classify={critic.classify(c)}"
        )
        labels = {}
        for nde in c.path:
            labels[nde.label] = labels.get(nde.label, 0) + 1
        print(f"  evidence: {dict(labels)}")
    else:
        print(f"  REJECTED  confidence={res.rejected[0].confidence:.3f} (< {critic.threshold})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    llm = get_llm()
    print(
        f"LLM: Bedrock Sonnet 4.6 "
        f"{'AVAILABLE' if llm.available else 'unavailable -> heuristic Supervisor'}"
    )
    critic = CriticAgent()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        doer = LifecycleDoer(session)

        def s1():
            spec = supervise(SCENARIO_QUESTIONS[1])
            print(f"[Supervisor] task_type={spec.task_type} depth={spec.traversal_depth}")
            show("SCENARIO 1 - Brand-mismatch cascade", doer.brand_mismatch_chain(), critic)

        def s2():
            show("SCENARIO 2 - Cross-source weak-signal fusion", doer.multi_signal_chain(), critic)

        def s3():
            print("\n== SCENARIO 3 - Top-20 at-risk SKUs ==")
            for i, (sku, s) in enumerate(doer.risk_rank(20), 1):
                print(f"  {i:>2}. {sku:>10}  anomaly={s:.3f}")

        def s4():
            print("\n== SCENARIO 4 - A/B: closed world vs Reflexive KG ==")
            cw = doer.closed_world_brand_dupes()
            print(f"  closed-world (b.flag='duplicate'): {len(cw)} rows")
            res = show("Reflexive KG agent", doer.brand_mismatch_chain(), critic)
            print(
                f"  A/B: closed=0 | reflexive={'validated' if res.validated else 'none'} -> "
                f"{'PASS' if (len(cw) == 0 and res.validated) else 'FAIL'}"
            )

        def s5():
            show("SCENARIO 5 - Shared-SKU boundary", doer.shared_sku_chain(), critic)

        def s6():
            show("SCENARIO 6 - Picklist auto-map error", doer.auto_map_chain(), critic)

        runners = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6}
        if args.demo or not args.scenario:
            for fn in runners.values():
                fn()
        else:
            runners[args.scenario]()
    driver.close()


if __name__ == "__main__":
    main()
