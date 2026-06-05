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
from agents.llm import bedrock_model_label, get_llm
from agents.supervisor import SupervisorAgent


def supervise(question: str):
    """NL → QuerySpec via Bedrock Claude Opus 4.7 (required)."""
    return SupervisorAgent().parse(question)


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

    get_llm()
    print(f"LLM: {bedrock_model_label()}")
    critic = CriticAgent()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        doer = LifecycleDoer(session)

        def s1():
            spec = supervise(SCENARIO_QUESTIONS[1])
            print(f"  Supervisor → {spec.task_type} depth={spec.traversal_depth}")

        def s2():
            show("SCENARIO 2 - Cross-source weak-signal fusion", doer.multi_signal_chain(), critic)

        def s3():
            for i, (sku, s) in enumerate(doer.risk_rank(20), 1):
                print(f"  {i:2}. SKU {sku}  score={s:.3f}")

        def s4():
            cw = doer.closed_world_brand_dupes()
            print(f"  Closed-world rows: {len(cw)}")
            res = show("Reflexive KG agent", doer.brand_mismatch_chain(), critic)
            return res

        def s5():
            show("SCENARIO 5 - Shared-SKU boundary", doer.shared_sku_chain(), critic)

        def s6():
            show("SCENARIO 6 - Picklist auto-map error", doer.auto_map_chain(), critic)

        if args.scenario:
            {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6}[args.scenario]()
        elif args.demo:
            for fn in (s1, s2, s3, s4, s5, s6):
                fn()
        else:
            ap.print_help()

    driver.close()


if __name__ == "__main__":
    main()
