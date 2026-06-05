"""
Reflexive KG — Analyst Workbench (Streamlit UI)

Run from project root:
    streamlit run ui/app.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from agents.lifecycle_doer import COHORT, SCENARIO_QUESTIONS
from ui.data_service import (
    check_connection,
    cohort_stats,
    load_auto_map_skus,
    load_blast_radius,
    load_cohort_rankings,
    load_manifest,
    load_score_histogram,
    load_shared_skus,
    load_training_gate,
    rel_weights_for_ui,
    run_closed_world,
)
from ui.render import inject_styles, render_ab_panel, render_pipeline_result

st.set_page_config(
    page_title="Reflexive KG",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

DEMO_PROMPTS = {
    "Brand mismatch (Scenario 1)": SCENARIO_QUESTIONS[1],
    "Weak multi-signal (Scenario 2)": SCENARIO_QUESTIONS[2],
    "A/B import degradation (Scenario 4)": SCENARIO_QUESTIONS[4],
    "Shared SKU boundary (Scenario 5)": SCENARIO_QUESTIONS[5],
    "Auto-map error (Scenario 6)": SCENARIO_QUESTIONS[6],
}


@contextlib.contextmanager
def quiet_pipeline():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def run_investigation(question: str, use_llm: bool):
    import importlib

    pipeline = importlib.import_module("04_agent_pipeline")
    with quiet_pipeline():
        if st.session_state.get("force_scenario"):
            return pipeline.run_lifecycle_pipeline(
                st.session_state.force_scenario,
                question=question,
                use_llm=use_llm,
            )
        anchor = st.session_state.get("selected_sku")
        if anchor and st.session_state.get("investigate_sku_chain"):
            return pipeline.run_sku_pipeline(anchor, question, use_llm=use_llm)
        return pipeline.run_pipeline(question, use_llm=use_llm)


def init_session():
    defaults = {
        "selected_sku": None,
        "investigate_sku_chain": False,
        "last_result": None,
        "last_ab_result": None,
        "last_closed": None,
        "force_scenario": None,
        "use_llm": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sidebar():
    st.sidebar.markdown('<p class="rkg-hero">Reflexive KG</p>', unsafe_allow_html=True)
    st.sidebar.caption(f"Cohort **{COHORT}**")

    ok, msg = check_connection()
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(f"Neo4j: {msg}")
        st.sidebar.info("Start Neo4j and run `python 05_synthesize_lifecycle.py`")

    st.sidebar.caption("Agents use **Bedrock Claude Opus 4.7** (required).")
    st.session_state.use_llm = True

    try:
        stats = cohort_stats()
        st.sidebar.metric("SKUs scored", stats["n_scored"])
        st.sidebar.metric("Top-decile threshold", f"{stats['threshold']:.3f}")
        st.sidebar.metric("High-risk SKUs", stats["high_risk"])
        st.sidebar.metric("Shared SKUs", stats["shared_count"])
    except Exception as e:
        st.sidebar.warning(f"Stats unavailable: {e}")

    st.sidebar.divider()
    st.sidebar.markdown("**Quick scenarios**")
    for label, num in [
        ("Run Scenario 1", 1),
        ("Run Scenario 4 (A/B)", 4),
        ("Run Scenario 5", 5),
    ]:
        if st.sidebar.button(label, use_container_width=True):
            st.session_state.force_scenario = num
            st.session_state.pending_question = SCENARIO_QUESTIONS[num]
            st.session_state.goto_tab = "Investigate"


def page_risk_inbox():
    st.markdown("## Risk inbox")
    st.caption("Proactive ranking before training — Scenario 3 / AC 8")

    col_chart, col_table = st.columns([1, 2])
    with col_chart:
        try:
            import numpy as np
            import pandas as pd

            scores = load_score_histogram()
            counts, edges = np.histogram(scores, bins=20)
            mids = [round((edges[i] + edges[i + 1]) / 2, 3) for i in range(len(counts))]
            st.bar_chart(
                pd.DataFrame({"SKUs": counts}, index=mids),
                height=350,
            )
            st.caption("Anomaly score distribution (cohort)")
        except Exception as e:
            st.error(str(e))

    with col_table:
        try:
            rows = load_cohort_rankings(30)
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "SKU": r.sku_id,
                        "Brand": r.brand_family,
                        "Anomaly": r.anomaly_score,
                        "Band": r.band,
                        "Planted": r.planted_type or "",
                    }
                    for r in rows
                ]
            )
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Anomaly": st.column_config.ProgressColumn(
                        "Anomaly",
                        min_value=0,
                        max_value=1,
                        format="%.3f",
                    ),
                },
            )

            sku_pick = st.selectbox(
                "Open in Investigation",
                options=[r.sku_id for r in rows],
                index=0,
            )
            if st.button("Investigate selected SKU", type="primary"):
                st.session_state.selected_sku = sku_pick
                st.session_state.investigate_sku_chain = True
                st.session_state.pending_question = (
                    f"Explain why GlobalSKU {sku_pick} has a high anomaly score "
                    f"and trace the distributed failure chain."
                )
                st.session_state.force_scenario = None
                st.session_state.goto_tab = "Investigate"
                st.rerun()
        except Exception as e:
            st.error(str(e))


def page_investigate():
    st.markdown("## Investigation")
    st.caption("NL root-cause and anomaly explain — Scenarios 1–2, 6 / AC 6, 11")

    prompt_col, btn_col = st.columns([4, 1])
    with prompt_col:
        default_q = st.session_state.get("pending_question", "")
        question = st.text_area(
            "Question",
            value=default_q,
            height=80,
            placeholder="Why are brands duplicated after customer import?",
        )
    with btn_col:
        st.write("")
        st.write("")
        run_btn = st.button("Run agents", type="primary", use_container_width=True)

    chip_cols = st.columns(len(DEMO_PROMPTS))
    for col, (label, q) in zip(chip_cols, DEMO_PROMPTS.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.pending_question = q
                st.session_state.force_scenario = {
                    "Brand mismatch (Scenario 1)": 1,
                    "Weak multi-signal (Scenario 2)": 2,
                    "A/B import degradation (Scenario 4)": 4,
                    "Shared SKU boundary (Scenario 5)": 5,
                    "Auto-map error (Scenario 6)": 6,
                }.get(label)
                st.rerun()

    if st.session_state.selected_sku:
        st.info(f"Anchor SKU: **{st.session_state.selected_sku}** (from Risk inbox)")

    if run_btn and question.strip():
        with st.spinner("Supervisor → Planner → Doer → Critic …"):
            try:
                result = run_investigation(question.strip(), st.session_state.use_llm)
                st.session_state.last_result = result
                st.session_state.pending_question = question
                st.session_state.force_scenario = None
                st.session_state.investigate_sku_chain = bool(st.session_state.selected_sku)
            except Exception as e:
                st.exception(e)

    if st.session_state.last_result:
        render_pipeline_result(st.session_state.last_result)


def page_ab_compare():
    st.markdown("## A/B — Closed world vs Reflexive KG")
    st.caption("Mandatory demo moment — Scenario 4 / AC 9")

    if st.button("Run comparison", type="primary"):
        import importlib

        pipeline = importlib.import_module("04_agent_pipeline")
        with quiet_pipeline():
            with st.spinner("Running closed-world query and reflexive agent …"):
                closed = run_closed_world()
                result = pipeline.run_lifecycle_pipeline(4, use_llm=st.session_state.use_llm)
        st.session_state.last_closed = closed
        st.session_state.last_ab_result = result

    if st.session_state.last_ab_result is not None:
        render_ab_panel(
            st.session_state.last_closed or [],
            st.session_state.last_ab_result,
        )
    else:
        st.info("Click **Run comparison** to load the side-by-side panel.")


def page_boundaries():
    st.markdown("## Boundaries & mapping")
    st.caption("Shared-SKU and auto-map guardrails — Scenarios 5–6 / AC 15")

    blast_sku = st.selectbox(
        "Blast-radius lookup (shared SKU)",
        options=[""] + [r.sku_id for r in load_cohort_rankings(100)],
        format_func=lambda x: x or "— select SKU —",
        key="blast_sku_pick",
    )
    if blast_sku:
        try:
            br = load_blast_radius(blast_sku)
            if br.get("found"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Customers affected", br["customer_count"])
                c2.metric("Tenant mappings", br["tenant_mappings"])
                c3.metric("Scan failures", br["scan_failures"])
                c4.metric("Unsafe to change", "Yes" if br["unsafe_to_change"] else "No")
                st.markdown(
                    f"**Customers:** {', '.join(br['customers']) or '—'}  \n"
                    f"**Brand / package:** {br.get('linked_brand') or br.get('brand')} / "
                    f"{br.get('package') or '—'}"
                )
            else:
                st.warning(f"SKU {blast_sku} not found in graph")
        except Exception as e:
            st.error(str(e))

    left, right = st.columns(2)
    with left:
        st.markdown("### Shared SKUs (multi-customer)")
        try:
            shared = load_shared_skus()
            if shared:
                import pandas as pd

                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "SKU": r["sku"],
                                "Brand": r.get("brand", ""),
                                "Customers": len(r["customers"]),
                                "Safe to change": "No",
                            }
                            for r in shared
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                if st.button("Investigate shared-SKU scenario", key="s5"):
                    st.session_state.force_scenario = 5
                    st.session_state.pending_question = SCENARIO_QUESTIONS[5]
                    st.session_state.goto_tab = "Investigate"
                    st.rerun()
            else:
                st.warning("No shared SKUs found — run 05_synthesize_lifecycle.py")
        except Exception as e:
            st.error(str(e))

    with right:
        st.markdown("### Auto-map errors (fuzzy tenant → global)")
        try:
            auto = load_auto_map_skus()
            if auto:
                import pandas as pd

                st.dataframe(pd.DataFrame(auto), use_container_width=True, hide_index=True)
                if st.button("Investigate auto-map scenario", key="s6"):
                    st.session_state.force_scenario = 6
                    st.session_state.pending_question = SCENARIO_QUESTIONS[6]
                    st.session_state.goto_tab = "Investigate"
                    st.rerun()
            else:
                st.warning("No auto-map SKUs tagged")
        except Exception as e:
            st.error(str(e))


def page_preflight():
    st.markdown("## Cohort pre-flight")
    st.caption("One-page readiness checklist for new customer onboarding")

    manifest = load_manifest()
    try:
        stats = cohort_stats()
        rankings = load_cohort_rankings(20)
        top_decile_n = max(1, stats["n_scored"] // 10)
        top_skus = rankings[:top_decile_n]
    except Exception as e:
        st.error(str(e))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cohort size", stats["n_scored"])
    c2.metric("Top-decile SKUs", top_decile_n)
    c3.metric("Shared SKUs", stats["shared_count"])
    c4.metric("Auto-map flagged", stats["auto_map_count"])

    checks = []

    def add_check(title, ok, detail, action=None):
        checks.append((title, ok, detail, action))

    add_check(
        "Top-decile SKUs identified",
        len(top_skus) > 0,
        f"{len(top_skus)} SKUs above threshold {stats['threshold']:.3f}",
    )
    add_check(
        "Shared-SKU boundaries surfaced",
        stats["shared_count"] >= 1,
        f"{stats['shared_count']} SKUs used by >1 customer",
    )
    add_check(
        "Evaluation manifest present",
        manifest is not None,
        f"{len(manifest.get('ground_truth', []))} ground-truth labels"
        if manifest
        else "Run 05_synthesize_lifecycle.py",
    )

    try:
        gate = load_training_gate()
        gate_ok = gate.get("ok", False)
        gate_detail = (
            f"Top decile has {gate.get('top_decile_count', 0)} SKUs "
            f"(threshold {gate.get('threshold', 0):.3f}); "
            f"{gate.get('shared_in_top_decile', 0)} shared-SKU in top decile"
        )
        if gate.get("warn"):
            gate_detail += " — review before training"
    except Exception as e:
        gate_ok = False
        gate_detail = str(e)
    add_check(
        "Training gate (top-decile clear)",
        gate_ok,
        gate_detail,
    )

    brand_ok = False
    brand_detail = "Not run yet"
    if st.button("Verify brand-mismatch chain (Scenario 1)", key="pf_s1"):
        import importlib

        pipeline = importlib.import_module("04_agent_pipeline")
        with quiet_pipeline():
            r = pipeline.run_lifecycle_pipeline(1, use_llm=True)
        brand_ok = bool(r.critic_result.validated)
        brand_detail = (
            f"Confidence {r.critic_result.validated[0].confidence:.3f}"
            if brand_ok
            else "No validated chain"
        )
        st.session_state.pf_brand_ok = brand_ok
        st.session_state.pf_brand_detail = brand_detail

    if "pf_brand_ok" in st.session_state:
        brand_ok = st.session_state.pf_brand_ok
        brand_detail = st.session_state.pf_brand_detail

    add_check("Brand-mismatch cascade detectable", brand_ok, brand_detail)

    for title, ok, detail, _ in checks:
        icon = "✅" if ok else "⬜"
        st.markdown(f"{icon} **{title}** — {detail}")

    if manifest:
        with st.expander("Ground-truth labels (judge benchmark)"):
            import pandas as pd

            st.dataframe(pd.DataFrame(manifest["ground_truth"]), hide_index=True)


def page_settings():
    st.markdown("## Settings")
    st.caption("REL_WEIGHTS (AC 5) — read-only in UI; edit config.py to tune")

    weights = rel_weights_for_ui()
    import pandas as pd

    st.dataframe(
        pd.DataFrame([{"Relationship": k, "Weight": v} for k, v in sorted(weights.items())]),
        hide_index=True,
        use_container_width=True,
    )
    st.code("python 05_synthesize_lifecycle.py  # re-plant + refresh reflect_emb", language="bash")
    st.code("python 06_evaluate.py  # acceptance metrics", language="bash")
    st.code("python 06_scale_evaluate.py --sample 5000  # catalog-scale eval", language="bash")


def main():
    init_session()
    sidebar()

    st.markdown('<p class="rkg-hero">VOR Hackathon · UseCase 3 · Analyst Workbench</p>', unsafe_allow_html=True)
    st.title("Reflexive Knowledge Graph")

    tab_names = [
        "Risk inbox",
        "Investigate",
        "A/B Compare",
        "Boundaries",
        "Pre-flight",
        "Settings",
    ]
    goto = st.session_state.pop("goto_tab", None)
    if goto in tab_names:
        st.session_state.nav_page = goto
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = tab_names[0]

    page = st.sidebar.radio(
        "Navigate",
        tab_names,
        index=tab_names.index(st.session_state.nav_page),
        key="nav_page",
    )

    {
        "Risk inbox": page_risk_inbox,
        "Investigate": page_investigate,
        "A/B Compare": page_ab_compare,
        "Boundaries": page_boundaries,
        "Pre-flight": page_preflight,
        "Settings": page_settings,
    }[page]()


if __name__ == "__main__":
    main()
