"""
agents/supervisor.py — Supervisor Agent

Responsibilities:
  - Accept a natural-language question from the user
  - Use Claude API to extract: entity_types, anchor_entity_id, task_type, traversal_depth
  - Return a QuerySpec
  - On Critic rejection: re-route with adjusted traversal depth

The LLM call uses structured JSON output via a system prompt.
If the API key is missing, falls back to a deterministic heuristic parser.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from agents.models import QuerySpec

# Known node labels in the graph — used to validate LLM extraction
KNOWN_LABELS = {
    "GlobalSKU", "VendorSKU", "Brand",
    "PackageType", "Manufacturer", "Supplier", "ProductClass",
}

# Task type keyword hints for heuristic fallback
TASK_TYPE_HINTS = {
    "root_cause":      ["why", "root cause", "how did", "trace", "what caused", "explain"],
    "risk_rank":       ["risk", "at risk", "likely to fail", "rank", "top", "worst", "before training"],
    "anomaly_explain": ["anomal", "flag", "unusual", "high score", "diverge", "what is wrong"],
}

# ─────────────────────────────────────────────────────────────────────────────
# LLM-BASED EXTRACTION (Claude API)
# ─────────────────────────────────────────────────────────────────────────────

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


def _call_claude(question: str) -> dict:
    """Call Claude API and parse the JSON response."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",   # fast + cheap for structured extraction
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        raw = message.content[0].text.strip()

        # Strip markdown code blocks if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        return json.loads(raw)

    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTIC FALLBACK (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_parse(question: str) -> dict:
    """
    Simple rule-based extraction when no Claude API key is available.
    Covers the main demo scenarios well enough for the POC.
    """
    q = question.lower()

    # task_type
    task_type = "anomaly_explain"  # default
    for ttype, hints in TASK_TYPE_HINTS.items():
        if any(h in q for h in hints):
            task_type = ttype
            break

    # entity_types — scan for known labels (case-insensitive)
    entity_types = []
    label_map = {
        "sku": "GlobalSKU", "global sku": "GlobalSKU", "globalsku": "GlobalSKU",
        "vendor sku": "VendorSKU", "vendorsku": "VendorSKU",
        "brand": "Brand",
        "package": "PackageType", "packagetype": "PackageType",
        "manufacturer": "Manufacturer",
        "supplier": "Supplier",
        "product class": "ProductClass", "class": "ProductClass",
    }
    for kw, label in label_map.items():
        if kw in q and label not in entity_types:
            entity_types.append(label)

    if not entity_types:
        entity_types = ["GlobalSKU"]  # safe default

    # anchor_entity_id — look for numeric IDs or quoted strings
    anchor_id = None
    anchor_label = None
    id_match = re.search(r"\b(\d{5,})\b", question)
    if id_match:
        anchor_id = id_match.group(1)
        anchor_label = "GlobalSKU"

    # traversal_depth
    depth_map = {"root_cause": 4, "risk_rank": 2, "anomaly_explain": 3}
    depth = depth_map.get(task_type, 3)

    return {
        "task_type":       task_type,
        "entity_types":    entity_types,
        "anchor_label":    anchor_label,
        "anchor_entity_id": anchor_id,
        "time_window":     None,
        "traversal_depth": depth,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUERY SPEC BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_spec(question: str, parsed: dict) -> QuerySpec:
    """Validate and construct a QuerySpec from parsed dict."""

    task_type = parsed.get("task_type", "anomaly_explain")
    if task_type not in ("root_cause", "risk_rank", "anomaly_explain"):
        task_type = "anomaly_explain"

    # Validate entity types
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


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Agent 1 — Query Orchestrator.

    Takes a natural-language question, extracts structured intent,
    returns a QuerySpec for the Planner.

    If the Critic rejects all chains, the Supervisor can be called again
    with adjust_depth=True to increase traversal depth by 1.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.use_llm:
            print("  [Supervisor] Using Claude API for query parsing")
        else:
            print("  [Supervisor] Using heuristic parser (no ANTHROPIC_API_KEY found)")

    def parse(self, question: str, adjust_depth: int = 0) -> QuerySpec:
        """
        Parse a natural-language question into a QuerySpec.

        Args:
            question: The user's question.
            adjust_depth: Add this many hops to the extracted depth (for re-routing).
        """
        print(f"\n[Supervisor] Parsing: '{question}'")

        if self.use_llm:
            try:
                parsed = _call_claude(question)
            except Exception as e:
                print(f"  [Supervisor] LLM call failed ({e}), falling back to heuristic")
                parsed = _heuristic_parse(question)
        else:
            parsed = _heuristic_parse(question)

        spec = _build_spec(question, parsed)

        # Apply depth adjustment for re-routing
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
