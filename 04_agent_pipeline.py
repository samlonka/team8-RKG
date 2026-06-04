"""
04_agent_pipeline.py — Full Pipeline Orchestrator

Wires the four agents end-to-end:
  Supervisor → Planner → Doer → Critic

Also implements:
  - Closed-world A/B comparison (Scenario 4 — the core demo moment)
  - All 6 hackathon demo scenarios
  - Auto re-routing when Critic rejects all chains

Usage:
    # Run all demo scenarios
    python 04_agent_pipeline.py --demo

    # Ask a specific question
    python 04_agent_pipeline.py --ask "Which SKUs are most at risk for training failure?"

    # A/B comparison only
    python 04_agent_pipeline.py --ab

    # Specific scenario
    python 04_agent_pipeline.py --scenario 1
"""

from __future__ import annotations

import argparse
import time

from neo4j import GraphDatabase

from agents.supervisor import SupervisorAgent
from agents.planner    import PlannerAgent
from agents.doer       import DoerAgent
from agents.critic     import CriticAgent
from agents.models     import PipelineResult, TaskList
from agents.lifecycle_doer import (
    SCENARIO_QUESTIONS,
    detect_scenario,
    run_lifecycle_scenario,
    run_sku_investigation,
)

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLOSED-WORLD A/B COMPARISON (Scenario 4)
# ─────────────────────────────────────────────────────────────────────────────

CLOSED_WORLD_QUERY = """
-- Closed-world Cypher: Find brand mismatch cascade
-- This query only works if someone has manually set b.flag = 'duplicate'
-- (which never happened — the flag was never set)
MATCH (s:GlobalSKU)-[:BELONGS_TO_BRAND]->(b:Brand)
WHERE b.flag = 'duplicate'
RETURN s.sku_id AS sku_id, b.brand_family AS brand
LIMIT 20
"""

def run_closed_world(driver) -> list[dict]:
    """Run the closed-world query and return raw results."""
    with driver.session() as session:
        result = session.run(
            "MATCH (s:GlobalSKU)-[:BELONGS_TO_BRAND]->(b:Brand) "
            "WHERE b.flag = 'duplicate' "
            "RETURN s.sku_id AS sku_id, b.brand_family AS brand "
            "LIMIT 20"
        )
        return [dict(r) for r in result]


def print_ab_comparison(closed_results: list[dict], pipeline_result: PipelineResult):
    """Print the side-by-side A/B comparison — the core demo moment."""
    print("\n" + "═" * 70)
    print("  SCENARIO 4 — A/B COMPARISON: CLOSED WORLD vs REFLEXIVE KG")
    print("═" * 70)

    print("\n┌─ CLOSED-WORLD CYPHER QUERY ─────────────────────────────────────┐")
    print(f"│ {CLOSED_WORLD_QUERY.strip()[:200]}")
    print("│")
    if closed_results:
        print(f"│ Results: {len(closed_results)} rows")
        for r in closed_results[:5]:
            print(f"│   {r}")
    else:
        print("│ Results: (no data)")
        print("│")
        print("│ The flag was never set. The pattern was never encoded.")
        print("│ The system acts as if the brand mismatch cascade does not exist.")
    print("└──────────────────────────────────────────────────────────────────┘")

    print("\n┌─ REFLEXIVE KG — AGENT PIPELINE ─────────────────────────────────┐")
    best = pipeline_result.critic_result.best()
    if best:
        print(f"│ Confidence: {best.confidence:.3f}")
        print(f"│ Avg anomaly score: {best.avg_anomaly_score:.3f}")
        print(f"│ Entities in chain: {len(best.path)}")
        print(f"│")
        print(f"│ Reasoning:")
        for line in best.reasoning.split(". "):
            if line.strip():
                print(f"│   {line.strip()}.")
        print(f"│")
        print(f"│ Top anomalous entities:")
        top = sorted(best.path, key=lambda n: n.anomaly_score or 0, reverse=True)[:5]
        for n in top:
            print(f"│   [{n.label}] {n.display_name:<30} score={n.anomaly_score}")
        print(f"│")
        print(f"│ No rule was written. Geometry detected it.")
    else:
        print("│ No validated chain found.")
        print("│ Try running 03_reflection.py first to compute reflect_emb.")
    print("└──────────────────────────────────────────────────────────────────┘")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE SCENARIO RUNNER (demo scenarios 1–6)
# ─────────────────────────────────────────────────────────────────────────────

def run_lifecycle_pipeline(
    scenario_num: int | None = None,
    question: str | None = None,
    use_llm: bool = True,
) -> PipelineResult:
    """
    Run a hackathon demo scenario using lifecycle-specific Cypher chains.

    Uses the proven traversals from agents/lifecycle_doer.py so mandatory
    scenarios validate under the Critic.
    """
    if scenario_num is None:
        if not question:
            raise ValueError("scenario_num or question required")
        scenario_num = detect_scenario(question)
        if scenario_num is None:
            return run_pipeline(question, use_llm=use_llm, max_rerouts=0)

    question = question or SCENARIO_QUESTIONS[scenario_num]
    start = time.time()
    critic = CriticAgent()

    spec, task_list, candidates, closed_rows = run_lifecycle_scenario(
        scenario_num, question=question, use_llm=use_llm
    )
    result = critic.validate(candidates)
    elapsed = round(time.time() - start, 2)

    pipeline_result = PipelineResult(
        question=question,
        spec=spec,
        tasks=task_list,
        candidates=candidates,
        critic_result=result,
        latency_seconds=elapsed,
    )
    pipeline_result._closed_world_rows = closed_rows  # type: ignore[attr-defined]
    return pipeline_result


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC PIPELINE RUNNER (open-ended NL questions)
# ─────────────────────────────────────────────────────────────────────────────

def run_sku_pipeline(
    sku_id: str,
    question: str | None = None,
    use_llm: bool = True,
) -> PipelineResult:
    """Investigate a single SKU with a distributed root-cause chain."""
    question = question or (
        f"Explain why GlobalSKU {sku_id} has a high anomaly score "
        f"and what graph evidence supports it."
    )
    start = time.time()
    critic = CriticAgent()
    spec, task_list, candidates, _ = run_sku_investigation(
        sku_id, question=question, use_llm=use_llm
    )
    result = critic.validate(candidates)
    elapsed = round(time.time() - start, 2)
    return PipelineResult(
        question=question,
        spec=spec,
        tasks=task_list,
        candidates=candidates,
        critic_result=result,
        latency_seconds=elapsed,
    )


def run_pipeline(
    question: str,
    use_llm: bool = True,
    max_rerouts: int = 1,
    anchor_sku: str | None = None,
) -> PipelineResult:
    """
    Run the full 4-agent pipeline on a natural-language question.

    Routes known demo questions to the lifecycle scenario path first.
    """
    if anchor_sku:
        return run_sku_pipeline(anchor_sku, question, use_llm=use_llm)

    scenario_num = detect_scenario(question)
    if scenario_num is not None:
        return run_lifecycle_pipeline(scenario_num, question, use_llm=use_llm)

    start = time.time()

    supervisor = SupervisorAgent(use_llm=use_llm)
    planner    = PlannerAgent()
    doer       = DoerAgent()
    critic     = CriticAgent()

    spec       = supervisor.parse(question)
    task_list  = planner.plan(spec)
    candidates = doer.execute(task_list)
    result     = critic.validate(candidates)

    # Re-routing: if Critic rejects all chains, increase depth and retry once
    rerout_count = 0
    while not result.validated and rerout_count < max_rerouts:
        rerout_count += 1
        print(f"\n[Pipeline] Critic rejected all chains. Re-routing (attempt {rerout_count}) ...")
        spec      = supervisor.parse(question, adjust_depth=1)
        task_list = planner.plan(spec)
        candidates = doer.execute(task_list)
        result    = critic.validate(candidates)

    doer.close()

    elapsed = round(time.time() - start, 2)

    pipeline_result = PipelineResult(
        question=question,
        spec=spec,
        tasks=task_list,
        candidates=candidates,
        critic_result=result,
        latency_seconds=elapsed,
    )

    return pipeline_result


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def print_pipeline_result(result: PipelineResult, scenario_num: int | None = None):
    heading = f"SCENARIO {scenario_num}" if scenario_num else "QUERY RESULT"
    print("\n" + "═" * 70)
    print(f"  {heading}")
    print("═" * 70)
    print(f"\n  Question: {result.question}")
    print(f"  Task type: {result.spec.task_type} | Depth: {result.spec.traversal_depth}")
    print(f"  Latency: {result.latency_seconds}s")
    print(f"\n  Candidates: {len(result.candidates)} | "
          f"Accepted: {len(result.critic_result.validated)} | "
          f"Acceptance rate: {result.critic_result.acceptance_rate:.0%}")

    critic = CriticAgent()
    for i, chain in enumerate(result.critic_result.validated, 1):
        classification = critic.classify(chain)
        print(f"\n  ── Chain #{i} [{classification}] ─────────────────────────────")
        print(f"     Confidence:    {chain.confidence:.3f}")
        print(f"     Temporal:      {chain.temporal_validity:.2f}")
        print(f"     Density:       {chain.evidence_density:.2f}")
        print(f"     Avg Anomaly:   {chain.avg_anomaly_score:.3f}")
        print(f"     Source:        {chain.source}")
        print(f"     Entities ({len(chain.path)}):")

        top_entities = sorted(chain.path, key=lambda n: n.anomaly_score or 0, reverse=True)
        for node in top_entities[:8]:
            score_str = f"{node.anomaly_score:.3f}" if node.anomaly_score else "  —  "
            print(f"       [{node.label:<13}] {node.display_name:<35} score={score_str}")

        print(f"\n     Reasoning:")
        for line in chain.reasoning.split(". "):
            if line.strip():
                print(f"       {line.strip()}.")

    if not result.critic_result.validated:
        print("\n  No validated chains. Possible causes:")
        print("  - reflect_emb not yet computed (run 03_reflection.py)")
        print("  - Confidence threshold too high (adjust CRITIC_CONFIDENCE_THRESHOLD in config.py)")
        print("  - Anchor entity ID not found in graph")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario_4_ab(use_llm: bool = True):
    """Scenario 4 — A/B comparison: the core demo moment."""
    print("\n[Scenario 4] Running closed-world query + reflexive lifecycle agent ...")
    result = run_lifecycle_pipeline(4, use_llm=use_llm)
    closed = getattr(result, "_closed_world_rows", None) or []
    print_ab_comparison(closed, result)
    return result


def run_all_scenarios(use_llm: bool = True):
    """Run all 6 hackathon demo scenarios."""
    print("\n" + "█" * 70)
    print("  REFLEXIVE KG — ALL DEMO SCENARIOS")
    print("█" * 70)

    for num, question in SCENARIO_QUESTIONS.items():
        if num == 4:
            run_scenario_4_ab(use_llm=use_llm)
        else:
            result = run_lifecycle_pipeline(num, question, use_llm=use_llm)
            print_pipeline_result(result, scenario_num=num)

    print("\n" + "█" * 70)
    print("  ALL SCENARIOS COMPLETE")
    print("█" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

def print_benchmarks(results: list[PipelineResult]):
    """Print latency and acceptance rate benchmarks across all scenarios."""
    print("\n── Benchmarks ───────────────────────────────────────────────────")
    print(f"  {'Question':<55} {'Latency':>8} {'Accepted':>9} {'Rate':>6}")
    print(f"  {'-'*55} {'-'*8} {'-'*9} {'-'*6}")
    for r in results:
        accepted = len(r.critic_result.validated)
        rate     = r.critic_result.acceptance_rate
        q        = r.question[:52] + "..." if len(r.question) > 52 else r.question
        print(f"  {q:<55} {r.latency_seconds:>7.1f}s {accepted:>9} {rate:>5.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reflexive KG Agent Pipeline")
    parser.add_argument("--ask",      type=str, help="Ask a custom question")
    parser.add_argument("--scenario", type=int, choices=range(1, 7),
                        help="Run a specific scenario (1-6)")
    parser.add_argument("--demo",     action="store_true",
                        help="Run all 6 demo scenarios")
    parser.add_argument("--ab",       action="store_true",
                        help="Run A/B comparison only (Scenario 4)")
    parser.add_argument("--no-llm",  action="store_true",
                        help="Use heuristic parser instead of Claude API")
    parser.add_argument("--bench",   action="store_true",
                        help="Run all scenarios and print benchmark table")
    args = parser.parse_args()

    use_llm = not args.no_llm

    if args.ask:
        result = run_pipeline(args.ask, use_llm=use_llm)
        print_pipeline_result(result)

    elif args.scenario == 4 or args.ab:
        run_scenario_4_ab(use_llm=use_llm)

    elif args.scenario:
        question = SCENARIO_QUESTIONS[args.scenario]
        result   = run_lifecycle_pipeline(args.scenario, question, use_llm=use_llm)
        print_pipeline_result(result, scenario_num=args.scenario)

    elif args.demo or args.bench:
        all_results = []
        for num, question in SCENARIO_QUESTIONS.items():
            if num == 4:
                r = run_scenario_4_ab(use_llm=use_llm)
            else:
                r = run_lifecycle_pipeline(num, question, use_llm=use_llm)
                print_pipeline_result(r, scenario_num=num)
            all_results.append(r)

        if args.bench:
            print_benchmarks(all_results)

    else:
        # Default: interactive mode
        print("\nReflexive KG Agent Pipeline")
        print("Type a question, or 'quit' to exit.\n")
        while True:
            try:
                q = input("Question: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    result = run_pipeline(q, use_llm=use_llm)
                    print_pipeline_result(result)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
