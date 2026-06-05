"""
agents/supervisor.py — Supervisor Agent

Responsibilities:
  - Accept a natural-language question from the user
  - Use Amazon Bedrock (Claude Opus 4.7) to extract structured QuerySpec JSON
  - Return a QuerySpec for the Planner

On Critic rejection, the Supervisor can be called again with adjust_depth=True.
"""

from __future__ import annotations

from agents.llm import LLMError, bedrock_model_label, get_llm
from agents.models import QuerySpec

KNOWN_LABELS = {
    "GlobalSKU", "VendorSKU", "Brand",
    "PackageType", "Manufacturer", "Supplier", "ProductClass",
}

SYSTEM_PROMPT = """You are a query parser for a knowledge graph system that manages SKU (product) data.

The graph has these node types:
- GlobalSKU: master product records with UPC, brand, package, manufacturer
- VendorSKU: customer-submitted product records that map to GlobalSKUs
- Brand: brand entities (e.g. "BIG RED", "PEPSI", "3D ENERGY")
- PackageType: package formats (e.g. "20OZ PL 1/24", "16OZ CN 1/12")
- Manufacturer: e.g. "PEPSI", "PBC", "GULF"
- Supplier: vendor companies e.g. "3D ENERGY DRINKS LLC"
- ProductClass: e.g. "NON ALC", "CRAFT BEER", "FMB"

Given a natural-language question, extract and return ONLY valid JSON with these fields:

{
  "task_type": "<root_cause|risk_rank|anomaly_explain>",
  "entity_types": ["<label1>", ...],
  "anchor_label": "<label or null>",
  "anchor_entity_id": "<specific ID string or null>",
  "time_window": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} or null,
  "traversal_depth": <integer 1-5>
}

Rules:
- task_type "root_cause": user wants to trace WHY something went wrong (use depth 3-5)
- task_type "risk_rank": user wants a ranked list of at-risk entities (use depth 1-2)
- task_type "anomaly_explain": user wants to understand a specific anomaly (use depth 2-3)
- anchor_entity_id: extract if the question names a specific SKU ID, brand, or product
- entity_types: list all node types relevant to the question
- traversal_depth: how many relationship hops to traverse (default 3)
- Return ONLY the JSON object, no explanation."""


def _build_spec(question: str, parsed: dict) -> QuerySpec:
    task_type = parsed.get("task_type", "anomaly_explain")
    if task_type not in ("root_cause", "risk_rank", "anomaly_explain"):
        task_type = "anomaly_explain"

    raw_types = parsed.get("entity_types") or ["GlobalSKU"]
    entity_types = [t for t in raw_types if t in KNOWN_LABELS]
    if not entity_types:
        entity_types = ["GlobalSKU"]

    anchor_label = parsed.get("anchor_label")
    if anchor_label not in KNOWN_LABELS:
        anchor_label = None

    depth = int(parsed.get("traversal_depth") or 3)
    depth = max(1, min(depth, 5))

    return QuerySpec(
        question=question,
        task_type=task_type,
        entity_types=entity_types,
        anchor_label=anchor_label,
        anchor_entity_id=parsed.get("anchor_entity_id"),
        time_window=parsed.get("time_window"),
        traversal_depth=depth,
    )


class SupervisorAgent:
    """
    Agent 1 — Query Orchestrator.

    Parses natural-language questions via Bedrock Claude Opus 4.7 only.
    """

    def __init__(self, use_llm: bool = True):
        # use_llm kept for call-site compatibility; Bedrock is always required.
        if not use_llm:
            raise LLMError(
                "Supervisor requires Bedrock LLM (use_llm=False is not supported). "
                f"Configure {bedrock_model_label()}."
            )
        print(f"  [Supervisor] Using {bedrock_model_label()}")

    def parse(self, question: str, adjust_depth: int = 0) -> QuerySpec:
        print(f"\n[Supervisor] Parsing: '{question}'")

        llm = get_llm()
        parsed = llm.json(
            f"Question: {question}",
            system=SYSTEM_PROMPT,
            max_tokens=512,
        )
        spec = _build_spec(question, parsed)

        if adjust_depth:
            spec.traversal_depth = min(spec.traversal_depth + adjust_depth, 5)
            print(f"  [Supervisor] Re-routing with depth {spec.traversal_depth}")

        print(
            f"  [Supervisor] → task_type={spec.task_type} | "
            f"entity_types={spec.entity_types} | "
            f"anchor={spec.anchor_entity_id} ({spec.anchor_label}) | "
            f"depth={spec.traversal_depth}"
        )
        return spec
