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


def _build_reasoning(
    chain: CandidateChain,
    temporal: float,
    density: float,
    anomaly: float,
    confidence: float,
) -> str:
    """Generate a human-readable explanation of the chain."""
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


def _llm_reasoning(chain: CandidateChain, base: str, confidence: float) -> str:
    """Bedrock narrative for validated chains; falls back to rule-based summary if unavailable."""
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

    Scores every candidate chain on temporal validity, evidence density,
    and anomaly signal. Rejects below threshold. Returns top-N.

    The Critic is also what produces the "Needs Review" classification:
    chains that scored close to the threshold get a different label
    than chains that scored well above it.
    """

    def __init__(
        self,
        threshold: float = CRITIC_CONFIDENCE_THRESHOLD,
        top_n: int = CRITIC_TOP_N,
        min_per_hop: int = MIN_ENTITIES_PER_HOP,
    ):
        self.threshold  = threshold
        self.top_n      = top_n
        self.min_per_hop = min_per_hop

    def validate(self, candidates: list[CandidateChain]) -> CriticResult:
        print(f"\n[Critic] Validating {len(candidates)} candidate chains "
              f"(threshold={self.threshold}) ...")

        validated: list[ValidatedChain] = []
        rejected:  list[RejectedChain]  = []

        for chain in candidates:
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
            conf     = _confidence(temporal, density, anomaly)

            if conf < self.threshold:
                rejected.append(RejectedChain(
                    chain_id=chain.chain_id,
                    reason=(
                        f"Confidence {conf:.3f} below threshold {self.threshold} "
                        f"(temporal={temporal:.2f}, density={density:.2f}, anomaly={anomaly:.2f})"
                    ),
                    confidence=conf,
                ))
                continue

            reasoning = _build_reasoning(chain, temporal, density, anomaly, conf)
            reasoning = _llm_reasoning(chain, reasoning, conf)

            validated.append(ValidatedChain(
                chain_id=chain.chain_id,
                path=chain.path,
                confidence=conf,
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

        return CriticResult(
            validated=top_validated,
            rejected=rejected,
            acceptance_rate=accept_rate,
        )

    def classify(self, chain: ValidatedChain) -> str:
        """
        Classify a validated chain for display:
        - 'Confirmed Anomaly'  : high confidence + high anomaly score
        - 'Needs Review'       : passed threshold but low anomaly signal
        - 'Healthy'            : low anomaly, probably not a problem
        """
        if chain.confidence >= 0.80 and chain.avg_anomaly_score >= ANOMALY_HIGH_RISK:
            return "Confirmed Anomaly"
        elif chain.confidence >= self.threshold:
            return "Needs Review"
        else:
            return "Healthy"
