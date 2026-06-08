"""
agents/supervisor.py — Supervisor Agent

Responsibilities:
  - Accept a natural-language question from the user
  - Classify intent for ALL hackathon flows:
      catalog_match | root_cause | risk_rank | anomaly_explain
      + optional scenario_num (1–6) for planted lifecycle demos
  - Heuristic parse when Bedrock unavailable; Bedrock for open-ended questions
  - Return a QuerySpec for the Planner
"""

from __future__ import annotations

import re

from agents.catalog_intent import (
    few_shot_prompt_block,
    is_catalog_duplicate_question,
    parse_catalog_duplicate,
    parse_catalog_match,
)
from agents.dim_match import merge_query_dims
from agents.lifecycle_doer import SCENARIO_QUESTIONS, SCENARIO_TITLES, detect_scenario, spec_for_lifecycle_scenario
from agents.llm import LLMError, bedrock_model_label, get_llm
from agents.pipeline_trace import TASK_TYPE_LABELS, trace
from agents.models import QuerySpec

KNOWN_LABELS = {
    "GlobalSKU", "TenantSKU", "Brand", "Customer",
    "PackageType", "TrainingImage", "MergeEvent", "Pallet",
    "Manufacturer", "Supplier", "ProductClass",
}

_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"GlobalSKU\s+(\d+)", re.I),
    re.compile(r"global\s+sku\s+(\d+)", re.I),
    re.compile(r"\bsku\s+(\d{4,})\b", re.I),
)


def extract_anchor_from_question(question: str) -> tuple[str | None, str | None]:
    """Heuristic anchor extraction when the LLM omits anchor_entity_id."""
    for pat in _ANCHOR_PATTERNS:
        m = pat.search(question)
        if m:
            return "GlobalSKU", m.group(1)
    return None, None


def apply_anchor_heuristic(spec: QuerySpec, question: str) -> QuerySpec:
    """Fill anchor_label / anchor_entity_id from the question text when missing."""
    if spec.anchor_entity_id:
        if not spec.anchor_label:
            spec.anchor_label = "GlobalSKU"
        return spec
    label, entity_id = extract_anchor_from_question(question)
    if entity_id:
        spec.anchor_label = label
        spec.anchor_entity_id = entity_id
    return spec

# Hackathon lifecycle scenarios — included in Supervisor few-shots and heuristics.
LIFECYCLE_FEW_SHOTS: list[dict] = [
    {
        "question": SCENARIO_QUESTIONS[1],
        "task_type": "root_cause",
        "scenario_num": 1,
        "entity_types": ["Brand", "GlobalSKU", "TenantSKU"],
        "traversal_depth": 4,
        "note": "brand duplication cascade during import",
    },
    {
        "question": SCENARIO_QUESTIONS[2],
        "task_type": "anomaly_explain",
        "scenario_num": 2,
        "entity_types": ["GlobalSKU", "TrainingImage", "Pallet"],
        "traversal_depth": 3,
        "note": "cross-source weak signals below individual thresholds",
    },
    {
        "question": SCENARIO_QUESTIONS[3],
        "task_type": "risk_rank",
        "scenario_num": 3,
        "entity_types": ["GlobalSKU"],
        "traversal_depth": 1,
        "note": "proactive top-20 at-risk ranking before training",
    },
    {
        "question": SCENARIO_QUESTIONS[4],
        "task_type": "root_cause",
        "scenario_num": 4,
        "entity_types": ["Brand", "GlobalSKU"],
        "traversal_depth": 4,
        "note": "A/B: closed-world Cypher blind vs reflexive KG",
    },
    {
        "question": SCENARIO_QUESTIONS[5],
        "task_type": "anomaly_explain",
        "scenario_num": 5,
        "entity_types": ["GlobalSKU", "Customer"],
        "traversal_depth": 3,
        "note": "shared SKUs unsafe to change cross-customer",
    },
    {
        "question": SCENARIO_QUESTIONS[6],
        "task_type": "root_cause",
        "scenario_num": 6,
        "entity_types": ["TenantSKU", "GlobalSKU"],
        "traversal_depth": 3,
        "note": "vendor auto-map to wrong GlobalSKU",
    },
]


def _lifecycle_few_shot_block() -> str:
    lines = ["Hackathon lifecycle scenarios (Neo4j cohort ACME_ONBOARDING):"]
    for ex in LIFECYCLE_FEW_SHOTS:
        lines.append(
            f'  Q: "{ex["question"][:90]}..." → '
            f'{{"task_type":"{ex["task_type"]}","scenario_num":{ex["scenario_num"]},'
            f'"entity_types":{ex["entity_types"]},"traversal_depth":{ex["traversal_depth"]}}}  '
            f'({ex["note"]})'
        )
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a query parser for a reflexive knowledge graph (handbook Use Case 3 — SKU data lifecycle).

The graph has these node types:
- GlobalSKU: master catalog SKU (~14k rows in PostgreSQL master_data + Neo4j) — UPC, brand_name, package_category_name
- TenantSKU: customer-specific product record at import — maps to GlobalSKU via MAPS_TO
- Brand, Customer, PackageType, TrainingImage, MergeEvent, Pallet, Manufacturer, Supplier, ProductClass

Data stores:
- PostgreSQL master_data: authoritative master catalog (brand_name slugs like AQUA_WTR, GATBLT_WSTW_MB_TM)
- Neo4j: reflexive KG with self_emb / reflect_emb for ANN matching and lifecycle traversals

Given a natural-language question, extract and return ONLY valid JSON:

{{
  "task_type": "<root_cause|risk_rank|anomaly_explain|catalog_match|catalog_duplicate>",
  "scenario_num": <integer 1-6 or null — set when question matches a hackathon demo scenario>,
  "entity_types": ["<label1>", ...],
  "anchor_label": "<label or null>",
  "anchor_entity_id": "<specific ID string or null>",
  "time_window": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null,
  "traversal_depth": <integer 1-5>,
  "brand_name": "<string or null — required when task_type is catalog_match>",
  "package_type": "<string or null — required when task_type is catalog_match>",
  "weight": <number or null — optional unit/case weight hint>,
  "height": <number or null — optional case height in inches>,
  "length": <number or null — optional case length in inches>,
  "width": <number or null — optional case width in inches>
}}

Rules:
- task_type "catalog_match": user asks whether a product/SKU EXISTS in the master catalog (PostgreSQL master_data). Set brand_name and package_type. If the user mentions weight or case dimensions (length/width/height), set those numeric fields too. scenario_num=null.
- task_type "catalog_duplicate": user asks whether DUPLICATE SKUs exist in the master catalog/database (data quality). NOT brand duplicates during customer import. scenario_num=null.
- task_type "root_cause": trace WHY something went wrong (brand mismatch, import failure, wrong mapping). depth 3-5.
- task_type "risk_rank": ranked list of at-risk SKUs (depth 1-2). scenario_num=3 for proactive top-20 before training.
- task_type "anomaly_explain": explain weak signals or cross-customer risk (depth 2-3).
- scenario_num: set 1-6 when the question clearly matches a planted lifecycle demo (see examples below). Otherwise null.
- Human brand labels (e.g. "AQUA WATER") may differ from catalog slugs (AQUA_WTR) — pass the user's text as brand_name.
- Package descriptors look like: 28OZ PL 1/15, 16.9OZ PL 15/1, 20OZ PL 1/24, 1.5L PL 1/12
- Return ONLY the JSON object, no explanation.

{_lifecycle_few_shot_block()}

{few_shot_prompt_block()}
"""


def heuristic_parse(question: str) -> QuerySpec | None:
    """
    Offline intent classification for all hackathon flows (no Bedrock).

    Priority:
      1. Master catalog lookup (brand + package)
      2. Master duplicate scan (catalog_duplicate)
      3. Lifecycle demo scenarios 1–6 (keyword / canonical question match)
    """
    q = question.strip()
    if not q:
        return None

    catalog = parse_catalog_match(q)
    if catalog:
        return QuerySpec(
            question=q,
            task_type="catalog_match",
            entity_types=["GlobalSKU"],
            traversal_depth=1,
            brand_name=catalog["brand_name"],
            package_type=catalog["package_type"],
            query_dims=dict(catalog.get("query_dims") or {}),
        )

    if is_catalog_duplicate_question(q):
        return QuerySpec(
            question=q,
            task_type="catalog_duplicate",
            entity_types=["GlobalSKU"],
            traversal_depth=1,
        )

    scenario_num = detect_scenario(q)
    if scenario_num is not None:
        spec = spec_for_lifecycle_scenario(scenario_num, q)
        return spec

    return None


def _build_spec(question: str, parsed: dict) -> QuerySpec:
    task_type = parsed.get("task_type", "anomaly_explain")
    if task_type not in ("root_cause", "risk_rank", "anomaly_explain", "catalog_match", "catalog_duplicate"):
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

    scenario_num = parsed.get("scenario_num")
    if scenario_num is not None:
        try:
            scenario_num = int(scenario_num)
            if scenario_num not in range(1, 7):
                scenario_num = None
        except (TypeError, ValueError):
            scenario_num = None

    brand_name = (parsed.get("brand_name") or "").strip() or None
    package_type = (parsed.get("package_type") or "").strip() or None
    query_dims = merge_query_dims(
        parse_catalog_match(question) or {},
        {
            "weight": parsed.get("weight"),
            "height": parsed.get("height"),
            "length": parsed.get("length"),
            "width": parsed.get("width"),
        },
    )

    if task_type == "catalog_match" and (not brand_name or not package_type):
        heuristic = parse_catalog_match(question)
        if heuristic:
            brand_name = heuristic["brand_name"]
            package_type = heuristic["package_type"]
            query_dims = merge_query_dims(query_dims, heuristic)
        else:
            task_type = "anomaly_explain"

    if task_type == "catalog_duplicate":
        return QuerySpec(
            question=question,
            task_type="catalog_duplicate",
            entity_types=["GlobalSKU"],
            traversal_depth=1,
        )

    if scenario_num is None and task_type not in ("catalog_match", "catalog_duplicate"):
        detected = detect_scenario(question)
        if detected is not None:
            return spec_for_lifecycle_scenario(detected, question)

    if scenario_num is not None:
        base = spec_for_lifecycle_scenario(scenario_num, question)
        if anchor_label:
            base.anchor_label = anchor_label
        if parsed.get("anchor_entity_id"):
            base.anchor_entity_id = parsed.get("anchor_entity_id")
        if parsed.get("time_window"):
            base.time_window = parsed.get("time_window")
        return apply_anchor_heuristic(base, question)

    spec = QuerySpec(
        question=question,
        task_type=task_type,
        entity_types=entity_types,
        anchor_label=anchor_label,
        anchor_entity_id=parsed.get("anchor_entity_id"),
        time_window=parsed.get("time_window"),
        traversal_depth=depth,
        brand_name=brand_name,
        package_type=package_type,
        query_dims=query_dims,
        scenario_num=scenario_num,
    )
    return apply_anchor_heuristic(spec, question)


class SupervisorAgent:
    """
    Agent 1 — Query Orchestrator.

    Parses natural-language questions into QuerySpec for the Planner.
    Calls Bedrock first when use_llm=True; heuristics are fallback only.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            print(f"  [Supervisor] Using {bedrock_model_label()}")

    def parse(
        self,
        question: str,
        adjust_depth: int = 0,
        forced_scenario: int | None = None,
    ) -> QuerySpec:
        print(f"\n[Supervisor] Parsing: '{question}'")

        if forced_scenario is not None:
            if forced_scenario not in range(1, 7):
                raise ValueError(f"forced_scenario must be 1–6, got {forced_scenario}")
            spec = spec_for_lifecycle_scenario(forced_scenario, question)
            print(
                f"  [Supervisor] → scenario {forced_scenario} (forced) | "
                f"task_type={spec.task_type} | depth={spec.traversal_depth}"
            )
            self._trace_spec(spec, source="forced_scenario")
            return spec

        heuristic = heuristic_parse(question)

        if self.use_llm:
            try:
                parsed = get_llm().json(
                    f"Question: {question}",
                    system=SYSTEM_PROMPT,
                    max_tokens=512,
                )
                spec = _build_spec(question, parsed)
                if adjust_depth and spec.task_type not in ("catalog_match", "catalog_duplicate") and spec.scenario_num is None:
                    spec.traversal_depth = min(spec.traversal_depth + adjust_depth, 5)
                    print(f"  [Supervisor] Re-routing with depth {spec.traversal_depth}")
                extra = ""
                if spec.task_type == "catalog_match":
                    extra = f" | brand={spec.brand_name!r} package={spec.package_type!r}"
                if spec.scenario_num:
                    extra += f" | scenario={spec.scenario_num}"
                print(
                    f"  [Supervisor] → task_type={spec.task_type} (LLM) | "
                    f"entity_types={spec.entity_types} | "
                    f"anchor={spec.anchor_entity_id} ({spec.anchor_label}) | "
                    f"depth={spec.traversal_depth}{extra}"
                )
                self._trace_spec(spec, source="llm")
                return spec
            except LLMError as exc:
                print(f"  [Supervisor] Bedrock unavailable — {exc}")

        if heuristic and not adjust_depth:
            tag = (
                f"scenario {heuristic.scenario_num} (heuristic)"
                if heuristic.scenario_num
                else f"{heuristic.task_type} (heuristic)"
            )
            extra = ""
            if heuristic.task_type == "catalog_match":
                extra = f" | brand={heuristic.brand_name!r} | package={heuristic.package_type!r}"
            print(f"  [Supervisor] → {tag}{extra}")
            self._trace_spec(heuristic, source="heuristic")
            return apply_anchor_heuristic(heuristic, question)

        if adjust_depth and heuristic and heuristic.scenario_num is None:
            heuristic.traversal_depth = min(heuristic.traversal_depth + adjust_depth, 5)
            self._trace_spec(heuristic, source="heuristic_reroute")
            return apply_anchor_heuristic(heuristic, question)

        if heuristic:
            self._trace_spec(heuristic, source="heuristic_fallback")
            return apply_anchor_heuristic(heuristic, question)

        raise LLMError(
            "Supervisor could not classify the question. "
            f"Configure {bedrock_model_label()} or rephrase using a demo scenario."
        )

    def _trace_spec(self, spec: QuerySpec, *, source: str) -> None:
        label = TASK_TYPE_LABELS.get(spec.task_type, spec.task_type.replace("_", " "))
        detail_parts = [f"Mission: {label}"]
        if spec.task_type == "catalog_match":
            detail_parts.append(f"{spec.brand_name} · {spec.package_type}")
        elif spec.scenario_num:
            detail_parts.append(
                SCENARIO_TITLES.get(spec.scenario_num, "Knowledge graph analysis")
            )
        elif spec.anchor_entity_id:
            detail_parts.append(f"Anchor {spec.anchor_label} {spec.anchor_entity_id}")
        trace(
            "supervisor", "done",
            "Intent locked in",
            " · ".join(detail_parts),
            task_type=spec.task_type,
            source=source,
            scenario=spec.scenario_num,
            brand_name=spec.brand_name,
            package_type=spec.package_type,
        )
