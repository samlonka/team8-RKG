"""
agents/critic.py — Critic Agent

Responsibilities:
  - Receive candidate chains from the Doer
  - Validate each chain on three dimensions:
      1. Temporal validity   — timestamps increase monotonically
      2. Evidence density    — minimum entities per hop
      3. Anomaly signal      — average anomaly score across path
  - Compute confidence = temporal × evidence_density × anomaly_signal
  - Reject chains below CRITIC_CONFIDENCE_THRESHOLD (default 0.65)
  - Return top-3 validated chains ranked by confidence

WHY A CRITIC:
  Without a Critic, the agent returns everything — including noise.
  The Critic is what makes the output trustworthy and the A/B comparison credible.
  A chain the Critic accepts at 0.81 confidence is meaningfully different from
  a closed-world query returning 0 results.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from agents.models import (
    CandidateChain, ValidatedChain, RejectedChain,
    CriticResult, EntityNode,
)
from agents.llm import LLMError, get_llm
from agents.dim_match import format_dim_comparison, format_dims, normalize_query_dims
from agents.product_risk import format_product_risk_for_prompt
from agents.pipeline_trace import trace
from config import (
    CRITIC_CONFIDENCE_THRESHOLD,
    MIN_ENTITIES_PER_HOP,
    CRITIC_TOP_N,
    ANOMALY_HIGH_RISK,
)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO-style timestamp strings. Returns None on failure."""
    if not ts:
        return None
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts.strip()[:26], fmt)
        except (ValueError, AttributeError):
            continue
    return None


def _temporal_validity(path: list[EntityNode]) -> float:
    """
    Fraction of consecutive node pairs where timestamps are valid and ordered.
    - If no nodes have timestamps: 0.5 (neutral — can't confirm or deny)
    - If all timestamps are ordered: 1.0
    - Penalty per inversion
    """
    timestamped = [n for n in path if n.has_timestamp()]
    if len(timestamped) < 2:
        return 0.5  # insufficient temporal data — neutral score

    parsed = [(_parse_timestamp(n.timestamp), n) for n in timestamped]
    parsed = [(dt, n) for dt, n in parsed if dt is not None]

    if len(parsed) < 2:
        return 0.5

    valid_pairs  = 0
    total_pairs  = len(parsed) - 1
    for i in range(total_pairs):
        if parsed[i][0] <= parsed[i + 1][0]:
            valid_pairs += 1

    return valid_pairs / total_pairs


def _evidence_density(path: list[EntityNode], min_per_hop: int) -> float:
    """
    Fraction of hops that meet the minimum entity count.
    For a POC with single-entity paths, this measures whether the
    chain has enough variety (multiple distinct labels).

    For multi-entity paths: checks entity count per distinct label group.
    """
    if not path:
        return 0.0

    # Group by label
    by_label: dict[str, int] = {}
    for node in path:
        by_label[node.label] = by_label.get(node.label, 0) + 1

    hops = len(by_label)  # number of distinct node types = conceptual hops

    if hops == 0:
        return 0.0

    # Count hops that meet minimum entity count
    valid_hops = sum(1 for count in by_label.values() if count >= min_per_hop)

    # Partial credit: also give credit for having multiple hops even if count is low
    diversity_bonus = min(hops / 3.0, 1.0)  # normalised to 3 expected hops

    raw = valid_hops / hops
    return min((raw * 0.7) + (diversity_bonus * 0.3), 1.0)


def _anomaly_signal(path: list[EntityNode]) -> float:
    """
    Average anomaly score across all path entities that have one.
    Normalised: 0.0 = all healthy, 1.0 = all maximally anomalous.
    Returns 0.0 if no entities have scores.
    """
    scores = [n.anomaly_score for n in path if n.anomaly_score is not None]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _confidence(
    temporal: float, density: float, anomaly: float
) -> float:
    """
    Composite confidence score.
    Weights: temporal (30%), evidence density (30%), anomaly signal (40%).
    Anomaly gets the highest weight because it is the core novelty signal.
    """
    return round(
        (temporal * 0.30) + (density * 0.30) + (anomaly * 0.40),
        4,
    )


def _catalog_master_nodes(chain: CandidateChain) -> list[EntityNode]:
    """Master-match SKUs in a catalog chain (graph search hits are supporting context)."""
    return [n for n in chain.path if n.source == "master_match"]


def _catalog_top_node(chain: CandidateChain) -> EntityNode | None:
    """Top ranked master-match node, not the first graph-search hit."""
    master = _catalog_master_nodes(chain)
    if master:
        return master[0]
    return chain.path[0] if chain.path else None


def _build_catalog_reasoning(chain: CandidateChain, confidence: float) -> str:
    if not chain.path:
        return (
            "No matching GlobalSKU found in the master catalog for the given "
            "brand and package. Recommend CREATE_NEW or manual review."
        )
    top = _catalog_top_node(chain)
    if top is None:
        return (
            "No matching GlobalSKU found in the master catalog for the given "
            "brand and package. Recommend CREATE_NEW or manual review."
        )
    props = top.properties
    status = props.get("match_status", "unknown")
    sku = top.entity_id
    brand = props.get("brand_name") or props.get("query_brand")
    pkg = props.get("package_category_name") or props.get("query_package")
    breakdown = props.get("score_breakdown") or {}
    query_dims = normalize_query_dims(props.get("query_dims"))
    parts = [
        f"Master catalog lookup: query brand '{props.get('query_brand')}' + "
        f"package '{props.get('query_package')}'.",
        f"Best match GlobalSKU {sku} ({brand} / {pkg}) — status={status}.",
    ]
    if query_dims:
        parts.append(f"Query dimensions: {format_dims(query_dims)}.")
        parts.append(format_dim_comparison(query_dims, props))
        if props.get("dim_applied"):
            mode = props.get("dim_mode", "nudge")
            parts.append(f"Dimension ranking applied ({mode}).")
    if breakdown:
        dim_note = ""
        if breakdown.get("dim_boost"):
            dim_note = f", dim={breakdown.get('dim_boost', 0):.2f}"
        parts.append(
            f"Signals — brand={breakdown.get('brand_match', 0):.2f}, "
            f"pkg={breakdown.get('pkg_match', 0):.2f}, "
            f"ANN={breakdown.get('ann_sim', 0):.2f}, "
            f"anomaly_attn={breakdown.get('anomaly_attn', 0):.2f}{dim_note}."
        )
    product_risk = props.get("product_risk")
    if product_risk and product_risk.get("anomaly_max") is not None:
        parts.append(
            f"Product ecosystem: {product_risk.get('classification')} risk "
            f"(max={product_risk.get('anomaly_max'):.2f}, "
            f"mean={product_risk.get('anomaly_mean'):.2f}). "
            f"{product_risk.get('summary', '')}"
        )
        drivers = product_risk.get("drivers") or []
        if drivers:
            top = drivers[0]
            parts.append(
                f"Top driver: {top.get('label')} {top.get('entity_id')} "
                f"(anomaly={top.get('anomaly'):.2f}, {top.get('context', '')})."
            )
    alt = _catalog_master_nodes(chain)[1:3]
    if alt:
        parts.append(
            "Alternates: "
            + ", ".join(f"{n.entity_id} ({n.display_name})" for n in alt)
        )
    parts.append(f"Match confidence: {confidence:.3f}.")
    return " ".join(parts)


def _validate_catalog_chain(chain: CandidateChain) -> tuple[float, str] | None:
    """Return (confidence, reasoning) for a master_match chain, or None to reject."""
    if not chain.path:
        return 0.0, _build_catalog_reasoning(chain, 0.0)

    top = _catalog_top_node(chain)
    if top is None:
        return None

    conf = float(top.properties.get("match_confidence") or 0.0)
    status = top.properties.get("match_status", "insert")

    if status == "merged" and conf >= 0.85:
        return conf, _build_catalog_reasoning(chain, conf)
    if status in ("merged", "updated") and conf >= 0.60:
        return conf, _build_catalog_reasoning(chain, conf)
    if conf >= 0.30:
        return conf, _build_catalog_reasoning(chain, conf)
    return None


def _build_graph_reasoning(
    chain: CandidateChain,
    temporal: float,
    density: float,
    anomaly: float,
    confidence: float,
) -> str:
    """Generate a human-readable explanation of a graph investigation chain."""
    path = chain.path
    labels  = list({n.label for n in path})
    high    = [n for n in path if (n.anomaly_score or 0) >= ANOMALY_HIGH_RISK]
    names   = [n.display_name for n in path[:3]]

    parts = []

    if chain.source == "union":
        parts.append("Cross-source evidence union (Cypher + self_emb ANN + reflect_emb ANN).")
    elif chain.source.startswith("ann_reflect"):
        parts.append("Neighbourhood-context ANN: these entities share a similar graph context "
                     "even if their own attributes differ — a signal no rule could encode.")
    elif chain.source.startswith("ann_self"):
        parts.append("Semantic ANN: these entities are attribute-similar.")
    else:
        parts.append("Graph traversal via Cypher.")

    parts.append(
        f"Path spans {len(path)} entities across {len(labels)} node type(s): "
        f"{', '.join(labels)}."
    )

    if names:
        parts.append(f"Top entities: {', '.join(str(n) for n in names)}.")

    if high:
        parts.append(
            f"{len(high)} entities are HIGH-RISK anomaly (score ≥ {ANOMALY_HIGH_RISK}): "
            f"{', '.join(n.display_name for n in high[:3])}."
        )

    parts.append(
        f"Scores — temporal: {temporal:.2f} | density: {density:.2f} | "
        f"anomaly: {anomaly:.2f} → confidence: {confidence:.3f}."
    )

    return " ".join(parts)


def _llm_verdict(
    chain: CandidateChain,
    question: str,
    task_type: str,
    rule_conf: float,
    temporal: float,
    density: float,
    anomaly: float,
    base_reasoning: str,
) -> dict | None:
    """
    Critic LLM: final accept/reject decision + analyst-facing reasoning.
    Returns dict with accept, confidence, reasoning, classification — or None on failure.
    """
    path = " → ".join(f"{n.label}:{n.display_name}" for n in chain.path[:12])
    if len(chain.path) > 12:
        path += f" (+{len(chain.path) - 12} more)"
    doer_note = f"\nDoer interpretation: {chain.llm_summary}" if chain.llm_summary else ""
    prompt = (
        f"User question: {question}\n"
        f"Task type: {task_type} | Chain source: {chain.source}\n"
        f"Rule-based scores — confidence: {rule_conf:.3f}, temporal: {temporal:.2f}, "
        f"density: {density:.2f}, anomaly: {anomaly:.2f}\n"
        f"Evidence path ({len(chain.path)} nodes): {path or '(empty)'}"
        f"{doer_note}\n\n"
        f"Rule-based summary: {base_reasoning}\n\n"
        "Return ONLY JSON:\n"
        '{"accept": true|false, "confidence": 0.0-1.0, "reasoning": "2-3 sentences", '
        '"classification": "Confirmed Anomaly|Needs Review|Healthy|Master Match|Not in Master"}'
    )
    try:
        return get_llm().json(
            prompt,
            system=(
                "You are the Critic agent for a reflexive SKU knowledge graph. "
                "Accept chains that credibly answer the question with sufficient evidence. "
                "For catalog matches, accept when a plausible GlobalSKU match exists."
            ),
            max_tokens=320,
        )
    except LLMError:
        return None


def _llm_catalog_verdict(
    chain: CandidateChain,
    question: str,
    rule_conf: float,
    base_reasoning: str,
) -> dict | None:
    top = _catalog_top_node(chain)
    props = top.properties if top else {}
    query_dims = normalize_query_dims(props.get("query_dims"))
    dim_block = ""
    if query_dims and top:
        dim_block = (
            f"\nQuery dimensions: {format_dims(query_dims)}\n"
            f"{format_dim_comparison(query_dims, props)}\n"
            f"Dimension tie-break applied: {bool(props.get('dim_applied'))} "
            f"(mode={props.get('dim_mode', 'none')})\n"
        )
    product_risk = props.get("product_risk")
    product_block = ""
    if product_risk:
        product_block = f"\n{format_product_risk_for_prompt(product_risk)}\n"
    prompt = (
        f"User question: {question}\n"
        f"Match confidence: {rule_conf:.3f} | status: {props.get('match_status', 'none')}\n"
        f"Best SKU: {top.entity_id if top else 'none'}\n"
        f"{dim_block}{product_block}"
        f"Rule summary: {base_reasoning}\n\n"
        "When physical dimensions were provided, explain whether the best SKU's "
        "weight/length/width/height support the match (especially after a tie-break).\n"
        "When product ecosystem risk is provided, explain whether neighbors (pallets, "
        "merges, tenant SKUs) make this product risky even if the GlobalSKU looks healthy.\n\n"
        "Return ONLY JSON:\n"
        '{"accept": true|false, "confidence": 0.0-1.0, "reasoning": "2-3 sentences", '
        '"classification": "Master Match|Needs Review|Not in Master"}'
    )
    try:
        return get_llm().json(
            prompt,
            system=(
                "You are the Critic agent validating master catalog SKU lookups. "
                "Use case dimensions (weight, length, width, height) when available "
                "to confirm or challenge close matches."
            ),
            max_tokens=320,
        )
    except LLMError:
        return None


def _llm_reasoning(chain: CandidateChain, base: str, confidence: float) -> str:
    """Legacy narrative polish — used only when LLM verdict did not supply reasoning."""
    path_summary = " → ".join(
        f"{n.label}:{n.display_name}" for n in chain.path[:8]
    )
    extra = ""
    if len(chain.path) > 8:
        extra = f" (+{len(chain.path) - 8} more hops)"
    try:
        return get_llm().complete(
            f"Evidence chain (confidence {confidence:.3f}):\n{path_summary}{extra}\n\n"
            f"Rule-based summary: {base}\n\n"
            "Write 2-3 sentences explaining why this chain is trustworthy or needs review. "
            "Mention temporal ordering, evidence density, and anomaly signal.",
            system="You are the Critic agent for a reflexive SKU knowledge graph.",
            max_tokens=256,
        )
    except LLMError:
        return base


# ─────────────────────────────────────────────────────────────────────────────
# CRITIC AGENT
# ─────────────────────────────────────────────────────────────────────────────

class CriticAgent:
    """
    Agent 4 — Evidence Validator.

    When use_llm=True, Bedrock makes the final accept/reject decision using
    rule-based scores as structured input. Rule-only fallback when Bedrock
    is unavailable (demo / offline mode).
    """

    def __init__(
        self,
        threshold: float = CRITIC_CONFIDENCE_THRESHOLD,
        top_n: int = CRITIC_TOP_N,
        min_per_hop: int = MIN_ENTITIES_PER_HOP,
        use_llm: bool = True,
    ):
        self.threshold  = threshold
        self.top_n      = top_n
        self.min_per_hop = min_per_hop
        self.use_llm    = use_llm

    def validate(
        self,
        candidates: list[CandidateChain],
        question: str = "",
        task_type: str = "root_cause",
    ) -> CriticResult:
        print(f"\n[Critic] Validating {len(candidates)} candidate chains "
              f"(threshold={self.threshold}) ...")

        validated: list[ValidatedChain] = []
        rejected:  list[RejectedChain]  = []

        for chain in candidates:
            if chain.source == "master_duplicate":
                if not chain.path:
                    rejected.append(RejectedChain(
                        chain_id=chain.chain_id,
                        reason="Empty duplicate scan result",
                        confidence=0.0,
                    ))
                    continue
                scores = [n.anomaly_score or 0.0 for n in chain.path]
                conf = max(scores) if scores else 0.7
                if not chain.path[0].entity_id.startswith("NO_DUPLICATES"):
                    conf = max(conf, 0.72)
                else:
                    conf = 0.88
                reasoning = chain.llm_summary or (
                    "Master catalog duplicate scan completed. "
                    + (
                        f"Found {len(chain.path)} duplicate group(s)."
                        if chain.path[0].entity_id != "NO_DUPLICATES"
                        else "No duplicate UPC or brand+package groups detected."
                    )
                )
                if self.use_llm and question:
                    verdict = _llm_verdict(
                        chain, question, task_type, conf, 1.0, 1.0,
                        conf, reasoning,
                    )
                    if verdict is not None:
                        conf = float(verdict.get("confidence", conf))
                        reasoning = verdict.get("reasoning") or reasoning
                        if not verdict.get("accept", True):
                            rejected.append(RejectedChain(
                                chain_id=chain.chain_id,
                                reason=f"LLM rejected: {reasoning[:120]}",
                                confidence=conf,
                            ))
                            continue
                validated.append(ValidatedChain(
                    chain_id=chain.chain_id,
                    path=chain.path,
                    confidence=conf,
                    temporal_validity=1.0,
                    evidence_density=min(1.0, len(chain.path) / 3.0),
                    avg_anomaly_score=sum(scores) / len(scores) if scores else 0.0,
                    reasoning=reasoning,
                    source=chain.source,
                ))
                continue

            if chain.source == "master_match":
                catalog = _validate_catalog_chain(chain)
                if catalog is None:
                    rejected.append(RejectedChain(
                        chain_id=chain.chain_id,
                        reason="Match confidence below minimum threshold",
                        confidence=0.0,
                    ))
                    continue
                conf, base_reasoning = catalog
                reasoning = base_reasoning
                final_conf = conf

                if self.use_llm and question:
                    verdict = _llm_catalog_verdict(chain, question, conf, base_reasoning)
                    if verdict is not None:
                        final_conf = float(verdict.get("confidence", conf))
                        reasoning = verdict.get("reasoning") or base_reasoning
                        if not verdict.get("accept", True):
                            rejected.append(RejectedChain(
                                chain_id=chain.chain_id,
                                reason=f"LLM rejected: {reasoning[:120]}",
                                confidence=final_conf,
                            ))
                            continue
                        print(f"  [Critic] LLM verdict chain={chain.chain_id} accept confidence={final_conf:.3f}")

                validated.append(ValidatedChain(
                    chain_id=chain.chain_id,
                    path=chain.path,
                    confidence=final_conf,
                    temporal_validity=1.0,
                    evidence_density=1.0 if chain.path else 0.0,
                    avg_anomaly_score=_anomaly_signal(chain.path),
                    reasoning=reasoning,
                    source=chain.source,
                ))
                continue

            if not chain.path:
                rejected.append(RejectedChain(
                    chain_id=chain.chain_id,
                    reason="Empty path",
                    confidence=0.0,
                ))
                continue

            temporal = _temporal_validity(chain.path)
            density  = _evidence_density(chain.path, self.min_per_hop)
            anomaly  = _anomaly_signal(chain.path)
            rule_conf = _confidence(temporal, density, anomaly)
            base_reasoning = _build_graph_reasoning(chain, temporal, density, anomaly, rule_conf)

            final_conf = rule_conf
            reasoning = base_reasoning
            accepted = rule_conf >= self.threshold

            if self.use_llm and question:
                verdict = _llm_verdict(
                    chain, question, task_type, rule_conf,
                    temporal, density, anomaly, base_reasoning,
                )
                if verdict is not None:
                    final_conf = float(verdict.get("confidence", rule_conf))
                    reasoning = verdict.get("reasoning") or base_reasoning
                    accepted = bool(verdict.get("accept", accepted))
                    print(
                        f"  [Critic] LLM verdict chain={chain.chain_id} "
                        f"accept={accepted} confidence={final_conf:.3f} "
                        f"(rule={rule_conf:.3f})"
                    )
                elif self.use_llm:
                    reasoning = _llm_reasoning(chain, base_reasoning, rule_conf)
            elif accepted:
                reasoning = _llm_reasoning(chain, base_reasoning, rule_conf)

            if not accepted or final_conf < self.threshold:
                rejected.append(RejectedChain(
                    chain_id=chain.chain_id,
                    reason=(
                        f"Confidence {final_conf:.3f} below threshold {self.threshold} "
                        f"(temporal={temporal:.2f}, density={density:.2f}, anomaly={anomaly:.2f})"
                    ),
                    confidence=final_conf,
                ))
                continue

            validated.append(ValidatedChain(
                chain_id=chain.chain_id,
                path=chain.path,
                confidence=final_conf,
                temporal_validity=temporal,
                evidence_density=density,
                avg_anomaly_score=anomaly,
                reasoning=reasoning,
                source=chain.source,
            ))

        # Sort by confidence descending, return top-N
        validated.sort(key=lambda v: v.confidence, reverse=True)
        top_validated = validated[: self.top_n]

        total      = len(candidates)
        accepted   = len(validated)
        accept_rate = accepted / total if total > 0 else 0.0

        print(f"  [Critic] Accepted: {accepted}/{total} "
              f"({accept_rate:.0%}) | Returning top-{self.top_n}")

        for i, v in enumerate(top_validated, 1):
            print(
                f"    #{i} chain={v.chain_id} confidence={v.confidence:.3f} "
                f"temporal={v.temporal_validity:.2f} density={v.evidence_density:.2f} "
                f"anomaly={v.avg_anomaly_score:.2f} source={v.source}"
            )

        if rejected:
            print(f"  [Critic] Rejected {len(rejected)} chains:")
            for r in rejected[:5]:  # show first 5
                print(f"    chain={r.chain_id} ({r.reason[:80]})")

        best_conf = top_validated[0].confidence if top_validated else 0.0
        if accepted:
            verdict = "Evidence accepted"
            detail = f"{accepted}/{total} chain(s) passed · best confidence {best_conf:.0%}"
        else:
            verdict = "No chain met the bar"
            detail = f"0/{total} chain(s) passed the quality threshold"
        trace(
            "critic", "done",
            verdict,
            detail,
            accepted=accepted,
            total=total,
            best_confidence=round(best_conf, 4),
        )

        return CriticResult(
            validated=top_validated,
            rejected=rejected,
            acceptance_rate=accept_rate,
        )

    def classify(self, chain: ValidatedChain) -> str:
        """
        Classify a validated chain for display.
        """
        if chain.source == "master_duplicate":
            if chain.path and chain.path[0].entity_id == "NO_DUPLICATES":
                return "Clean Master Catalog"
            return "Master Duplicates Found"

        if chain.source == "master_match":
            if not chain.path:
                return "Not in Master"
            top = _catalog_top_node(chain)
            if top is None:
                return "Not in Master"
            status = top.properties.get("match_status", "insert")
            if status == "merged" and chain.confidence >= 0.85:
                return "Master Match"
            if status in ("merged", "updated"):
                return "Needs Review"
            return "Not in Master"

        if chain.confidence >= 0.80 and chain.avg_anomaly_score >= ANOMALY_HIGH_RISK:
            return "Confirmed Anomaly"
        elif chain.confidence >= self.threshold:
            return "Needs Review"
        else:
            return "Healthy"
