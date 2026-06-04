"""Streamlit rendering helpers for pipeline results."""

from __future__ import annotations

import html

import streamlit as st

from agents.critic import CriticAgent
from agents.models import PipelineResult, ValidatedChain, EntityNode
from config import CRITIC_CONFIDENCE_THRESHOLD


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp { background-color: #0a0c10; }
        section[data-testid="stSidebar"] { background-color: #111318; }
        h1, h2, h3, label, p, span, .stMarkdown { color: #e8eaf0 !important; }
        .rkg-hero {
            font-family: ui-monospace, monospace;
            color: #4fffb0;
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            border: 1px solid rgba(79,255,176,0.25);
            padding: 0.35rem 0.75rem;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        .rkg-card {
            background: #181c24;
            border: 1px solid #232838;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        .rkg-muted { color: #6b7280; font-size: 0.85rem; }
        .badge-high { color: #ff6b6b; font-weight: 700; }
        .badge-med { color: #ffd166; font-weight: 700; }
        .badge-low { color: #4fffb0; font-weight: 700; }
        .badge-confirmed { background: rgba(255,107,107,0.15); color: #ff6b6b;
            padding: 0.2rem 0.5rem; border-radius: 4px; }
        .badge-review { background: rgba(255,209,102,0.15); color: #ffd166;
            padding: 0.2rem 0.5rem; border-radius: 4px; }
        .timeline-node {
            border-left: 3px solid #7c6dfa;
            padding-left: 0.75rem;
            margin: 0.35rem 0;
        }
        .ab-left { border: 1px solid #232838; background: #111318;
            padding: 1rem; min-height: 200px; }
        .ab-right { border: 1px solid rgba(79,255,176,0.35); background: #111318;
            padding: 1rem; min-height: 200px; }
        .ab-zero { font-size: 2.5rem; color: #6b7280; font-weight: 800; }
        .ab-win { font-size: 1.25rem; color: #4fffb0; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def classification_badge(chain: ValidatedChain) -> str:
    label = CriticAgent().classify(chain)
    css = "badge-confirmed" if label == "Confirmed Anomaly" else "badge-review"
    return f'<span class="{css}">{html.escape(label)}</span>'


def render_confidence_bars(chain: ValidatedChain):
    cols = st.columns(3)
    cols[0].metric("Temporal", f"{chain.temporal_validity:.2f}")
    cols[1].metric("Evidence density", f"{chain.evidence_density:.2f}")
    cols[2].metric("Anomaly signal", f"{chain.avg_anomaly_score:.2f}")
    st.progress(min(chain.confidence, 1.0), text=f"Confidence {chain.confidence:.3f}")


def render_timeline(path: list[EntityNode]):
    for i, node in enumerate(path):
        score_txt = f"{node.anomaly_score:.3f}" if node.anomaly_score is not None else "—"
        ts = node.timestamp or "—"
        st.markdown(
            f'<div class="timeline-node">'
            f"<strong>[{html.escape(node.label)}]</strong> "
            f"{html.escape(node.display_name)}<br/>"
            f'<span class="rkg-muted">score={score_txt} · {html.escape(ts)}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_chain(chain: ValidatedChain, idx: int = 1):
    st.markdown(classification_badge(chain), unsafe_allow_html=True)
    st.markdown(
        f"**Chain #{idx}** · confidence **{chain.confidence:.3f}** · "
        f"{len(chain.path)} entities · source `{chain.source}`"
    )
    render_confidence_bars(chain)
    st.markdown("#### Causal path")
    render_timeline(chain.path)
    st.markdown("#### Reasoning")
    st.info(chain.reasoning)


def render_pipeline_result(result: PipelineResult, show_trace: bool = True):
    st.caption(
        f"Task: `{result.spec.task_type}` · depth {result.spec.traversal_depth} · "
        f"{result.latency_seconds}s · "
        f"{len(result.critic_result.validated)} accepted / "
        f"{len(result.candidates)} candidates"
    )

    if show_trace:
        with st.expander("Agent trace (Supervisor → Planner → Doer → Critic)", expanded=False):
            st.write(f"**Question:** {result.question}")
            st.write(f"**Entity types:** {', '.join(result.spec.entity_types)}")
            if result.tasks.tasks:
                for t in result.tasks.tasks:
                    st.code(f"Step {t.step} [{t.task_type}] {t.description}", language=None)
            else:
                st.caption("Lifecycle scenario — curated Cypher chain (no generic task list).")

    best = result.critic_result.best()
    if best:
        render_chain(best, 1)
        for i, chain in enumerate(result.critic_result.validated[1:], 2):
            with st.expander(f"Alternate chain #{i} ({chain.confidence:.3f})"):
                render_chain(chain, i)
    else:
        st.warning("No validated chains. Try a demo scenario or ensure reflect_emb is computed.")

    if result.critic_result.rejected:
        with st.expander(
            f"Rejected chains ({len(result.critic_result.rejected)}) — Critic threshold "
            f"{CRITIC_CONFIDENCE_THRESHOLD}",
            expanded=False,
        ):
            for r in result.critic_result.rejected[:8]:
                st.caption(f"`{r.chain_id}` conf={r.confidence:.3f}: {r.reason}")


def render_ab_panel(closed_rows: list[dict], result: PipelineResult):
    from ui.data_service import CLOSED_WORLD_CYPHER

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="ab-left">', unsafe_allow_html=True)
        st.markdown("##### Closed-world Cypher")
        st.code(CLOSED_WORLD_CYPHER, language="cypher")
        if closed_rows:
            st.dataframe(closed_rows, use_container_width=True, hide_index=True)
        else:
            st.markdown('<p class="ab-zero">0 results</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="rkg-muted">The flag <code>b.flag = duplicate</code> was never set. '
                "The cascade is invisible to rule-based queries.</p>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="ab-right">', unsafe_allow_html=True)
        st.markdown("##### Reflexive KG agent")
        best = result.critic_result.best()
        if best:
            st.markdown('<p class="ab-win">Validated causal chain</p>', unsafe_allow_html=True)
            st.metric("Confidence", f"{best.confidence:.3f}")
            st.metric("Avg anomaly", f"{best.avg_anomaly_score:.3f}")
            st.caption(best.reasoning[:400] + ("…" if len(best.reasoning) > 400 else ""))
            render_timeline(best.path[:6])
            st.success("No rule was written. Geometry detected it.")
        else:
            st.error("No validated chain")
        st.markdown("</div>", unsafe_allow_html=True)
