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

from agents.llm import bedrock_model_label, get_llm
from agents.supervisor import SupervisorAgent
from agents.planner    import PlannerAgent
from agents.doer       import DoerAgent
from agents.critic     import CriticAgent
from agents.entity_display import format_entity_node, sku_summary
from agents.models     import PipelineResult, TaskList
from agents.lifecycle_doer import (
    SCENARIO_QUESTIONS,
    detect_scenario,
    run_lifecycle_scenario,
    run_sku_investigation,
)

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    AGENT_USE_LLM,
)
from agents.pipeline_trace import trace


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
            line = sku_summary(
                r.get("sku") or r.get("sku_id", "?"),
                r.get("brand_name"),
                r.get("brand_family") or r.get("brand"),
                r.get("package_type"),
            )
            print(f"│   {line}")
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
            print(f"│   {format_entity_node(n)}")
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
    Run a hackathon demo scenario through the unified four-agent pipeline.

    Supervisor (forced scenario) → Planner → Doer → Critic.
    """
    if scenario_num is None:
        if not question:
            raise ValueError("scenario_num or question required")
        scenario_num = detect_scenario(question)
        if scenario_num is None:
            return run_pipeline(question, use_llm=use_llm, max_rerouts=0)

    question = question or SCENARIO_QUESTIONS[scenario_num]
    return run_pipeline(question, use_llm=use_llm, max_rerouts=0, forced_scenario=scenario_num)


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC PIPELINE RUNNER (open-ended NL questions)
# ─────────────────────────────────────────────────────────────────────────────

def run_sku_pipeline(
    sku_id: str,
    question: str | None = None,
    use_llm: bool = AGENT_USE_LLM,
) -> PipelineResult:
    """Investigate a single SKU with a distributed root-cause chain."""
    question = question or (
        f"Explain why GlobalSKU {sku_id} has a high anomaly score "
        f"and what graph evidence supports it."
    )
    start = time.time()
    critic = CriticAgent(use_llm=use_llm)
    trace(
        "supervisor", "running",
        "Reading your question",
        question[:120] + ("…" if len(question) > 120 else ""),
    )
    spec, task_list, candidates, _ = run_sku_investigation(
        sku_id, question=question, use_llm=use_llm
    )
    trace(
        "supervisor", "done",
        "SKU investigation scoped",
        f"Tracing evidence for GlobalSKU {sku_id}",
        task_type=spec.task_type,
        anchor_sku=sku_id,
    )
    trace(
        "critic", "running",
        "Quality-checking the evidence",
        f"Reviewing paths for SKU {sku_id}",
    )
    result = critic.validate(
        candidates, question=question, task_type=spec.task_type,
    )
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
    use_llm: bool = AGENT_USE_LLM,
    max_rerouts: int = 1,
    anchor_sku: str | None = None,
    forced_scenario: int | None = None,
    extra_query_dims: dict | None = None,
) -> PipelineResult:
    """
    Run the full 4-agent pipeline on a natural-language question.

    All intent routing goes through Supervisor → Planner → Doer → Critic.
    """
    if anchor_sku:
        return run_sku_pipeline(anchor_sku, question, use_llm=use_llm)

    start = time.time()
    trace(
        "supervisor", "running",
        "Reading your question",
        question[:120] + ("…" if len(question) > 120 else ""),
    )
    supervisor = SupervisorAgent(use_llm=use_llm)
    spec = supervisor.parse(question, forced_scenario=forced_scenario)
    if extra_query_dims:
        from agents.dim_match import merge_query_dims
        spec.query_dims = merge_query_dims(spec.query_dims, extra_query_dims)

    is_lifecycle = spec.scenario_num is not None
    if use_llm and spec.task_type != "catalog_match" and not is_lifecycle:
        print(f"[Pipeline] LLM: {bedrock_model_label()}")

    planner = PlannerAgent(use_llm=use_llm)
    doer    = DoerAgent(use_llm=use_llm)
    critic  = CriticAgent(use_llm=use_llm)

    if is_lifecycle:
        print(f"[Pipeline] Lifecycle scenario {spec.scenario_num}")

    trace("planner", "running", "Designing the investigation plan", "Choosing Neo4j queries and retrieval steps")
    task_list  = planner.plan(spec)
    trace(
        "doer", "running",
        "Exploring the knowledge graph",
        f"Running {len(task_list.tasks)} step(s) against Neo4j",
        task_count=len(task_list.tasks),
    )
    candidates = doer.execute(task_list)
    trace(
        "critic", "running",
        "Quality-checking the evidence",
        f"Reviewing {len(candidates)} candidate chain(s)",
        candidates=len(candidates),
    )
    result     = critic.validate(
        candidates, question=question, task_type=spec.task_type,
    )

    rerout_count = 0
    while (
        spec.task_type != "catalog_match"
        and spec.task_type != "catalog_duplicate"
        and spec.scenario_num is None
        and not result.validated
        and rerout_count < max_rerouts
    ):
        rerout_count += 1
        print(f"\n[Pipeline] Critic rejected all chains. Re-routing (attempt {rerout_count}) ...")
        trace(
            "supervisor", "running",
            "Searching deeper",
            f"Widening the graph search (attempt {rerout_count})",
            reroute=rerout_count,
        )
        spec      = supervisor.parse(question, adjust_depth=1)
        task_list = planner.plan(spec)
        trace(
            "doer", "running",
            "Re-querying Neo4j",
            f"Running {len(task_list.tasks)} expanded step(s)",
            task_count=len(task_list.tasks),
        )
        candidates = doer.execute(task_list)
        trace(
            "critic", "running",
            "Re-checking the evidence",
            f"Reviewing {len(candidates)} new candidate chain(s)",
        )
        result    = critic.validate(
            candidates, question=question, task_type=spec.task_type,
        )

    closed_rows = getattr(doer, "_closed_world_rows", None)
    if is_lifecycle and spec.scenario_num == 4:
        n_closed = len(closed_rows or [])
        print(f"[Pipeline] Scenario 4 A/B — closed_world rows: {n_closed} | reflexive chains: {len(candidates)}")

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
    match_result = getattr(doer, "_last_match_result", None)
    if match_result:
        pipeline_result._match_result = match_result  # type: ignore[attr-defined]
    dup_report = getattr(doer, "_last_duplicate_report", None)
    if dup_report is not None:
        pipeline_result._duplicate_report = dup_report  # type: ignore[attr-defined]
    result_meta = getattr(doer, "_result_meta", None)
    if result_meta:
        pipeline_result._result_meta = result_meta  # type: ignore[attr-defined]
    if closed_rows is not None:
        pipeline_result._closed_world_rows = closed_rows  # type: ignore[attr-defined]

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
            print(f"       {format_entity_node(node)}")

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
    parser.add_argument("--bench",   action="store_true",
                        help="Run all scenarios and print benchmark table")
    args = parser.parse_args()

    use_llm = True

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
