"""
06b_evaluate_methods.py — Side-by-side evaluation of all reflection methods.

Compares all five scoring approaches against the seeded anomaly ground truth
in seed_manifest.json (written by 05_synthesize_lifecycle.py):

  Baseline  — 1 - cosine(self_emb, reflect_emb)          [03_reflection.py]
  Phase 1   — 1 - cosine(self_emb, reflect_emb_dir)       [03b_reflection_enhanced.py]
  Phase 2   — anomaly_rgcn                                 [08_rgcn.py]
  Phase 3   — triple_anomaly_score                         [09_kge.py]
  Phase 4   — dominant_score                               [10_dominant.py]

Metrics reported for each method:
  top_decile_recall  — fraction of planted brand/merge/evidence anomalies in top 10%
  healthy_lower_half — fraction of healthy SKUs below the score range midpoint
  classification_acc — accuracy vs analyst ground-truth labels (threshold = decile edge)
  auc_roc            — AUC-ROC over all ground-truth labeled nodes (requires scikit-learn)

Usage:
    python 06b_evaluate_methods.py
    python 06b_evaluate_methods.py --methods baseline phase1 phase2
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

MANIFEST_PATH = "seed_manifest.json"

# Per-method acceptance bars (same as 06_evaluate.py)
MIN_TOPDECILE_RECALL = 0.80
MIN_HEALTHY_LOWER    = 0.80
MIN_CLASSIF_ACC      = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# SCORE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _cos(a, b) -> float:
    va, vb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 1e-8 else 0.0


def load_baseline(session, cohort_tag: str) -> dict[str, float]:
    # Prefer the pre-stored anomaly_baseline (written by 03_reflection.py).
    # Fall back to on-the-fly cosine computation if not yet stored.
    rows = session.run(
        """
        MATCH (g:GlobalSKU {cohort: $tag})
        WHERE g.anomaly_baseline IS NOT NULL
        RETURN g.sku_id AS sku, g.anomaly_baseline AS score
        """,
        tag=cohort_tag,
    ).data()
    if rows:
        return {r["sku"]: r["score"] for r in rows}
    # fallback
    rows = session.run(
        """
        MATCH (g:GlobalSKU {cohort: $tag})
        WHERE g.self_emb IS NOT NULL AND g.reflect_emb IS NOT NULL
        RETURN g.sku_id AS sku, g.self_emb AS se, g.reflect_emb AS re
        """,
        tag=cohort_tag,
    ).data()
    return {r["sku"]: round(1 - _cos(r["se"], r["re"]), 4) for r in rows}


def load_phase1(session, cohort_tag: str) -> dict[str, float]:
    rows = session.run(
        """
        MATCH (g:GlobalSKU {cohort: $tag})
        WHERE g.anomaly_dir IS NOT NULL
        RETURN g.sku_id AS sku, g.anomaly_dir AS score
        """,
        tag=cohort_tag,
    ).data()
    return {r["sku"]: r["score"] for r in rows}


def load_phase2(session, cohort_tag: str) -> dict[str, float]:
    rows = session.run(
        """
        MATCH (g:GlobalSKU {cohort: $tag})
        WHERE g.anomaly_rgcn IS NOT NULL
        RETURN g.sku_id AS sku, g.anomaly_rgcn AS score
        """,
        tag=cohort_tag,
    ).data()
    return {r["sku"]: r["score"] for r in rows}


def load_phase3(session, cohort_tag: str) -> dict[str, float]:
    rows = session.run(
        """
        MATCH (g:GlobalSKU {cohort: $tag})
        WHERE g.triple_anomaly_score IS NOT NULL
        RETURN g.sku_id AS sku, g.triple_anomaly_score AS score
        """,
        tag=cohort_tag,
    ).data()
    return {r["sku"]: r["score"] for r in rows}


def load_phase4(session, cohort_tag: str) -> dict[str, float]:
    rows = session.run(
        """
        MATCH (g:GlobalSKU {cohort: $tag})
        WHERE g.dominant_score IS NOT NULL
        RETURN g.sku_id AS sku, g.dominant_score AS score
        """,
        tag=cohort_tag,
    ).data()
    return {r["sku"]: r["score"] for r in rows}


def load_attention(session, cohort_tag: str) -> dict[str, float]:
    rows = session.run(
        """
        MATCH (g:GlobalSKU {cohort: $tag})
        WHERE g.anomaly_attn IS NOT NULL
        RETURN g.sku_id AS sku, g.anomaly_attn AS score
        """,
        tag=cohort_tag,
    ).data()
    return {r["sku"]: r["score"] for r in rows}


def _simple_loader(prop: str):
    """Generate a cohort score loader for a given node property."""
    def loader(session, cohort_tag: str) -> dict[str, float]:
        rows = session.run(
            f"""
            MATCH (g:GlobalSKU {{cohort: $tag}})
            WHERE g.{prop} IS NOT NULL
            RETURN g.sku_id AS sku, g.{prop} AS score
            """,
            tag=cohort_tag,
        ).data()
        return {r["sku"]: r["score"] for r in rows}
    return loader


LOADERS = {
    "baseline":  load_baseline,
    "attention": load_attention,
    "phase1":    load_phase1,
    "phase2":    load_phase2,
    "phase3":    load_phase3,
    "phase4":    load_phase4,
    "reflect2":  _simple_loader("anomaly_reflect2"),
    "temporal":  _simple_loader("anomaly_temporal"),
    "degnorm":   _simple_loader("anomaly_degnorm"),
    "lof":       _simple_loader("anomaly_lof"),
    "ensemble":  _simple_loader("anomaly_ensemble"),
}

METHOD_LABELS = {
    "baseline":  "Baseline  (fixed weights, 1-hop mean)",
    "attention": "Attention (divergence-weighted softmax)",
    "reflect2":  "Reflect-2 (2-hop second-order)",
    "temporal":  "Temporal  (decay-weighted)",
    "degnorm":   "DegNorm   (degree-normalised)",
    "lof":       "LOF       (divergence-vector density)",
    "ensemble":  "Ensemble  (LR over all methods)",
    "phase1":    "Phase 1   (dir-split + severity)",
    "phase2":    "Phase 2   (R-GCN encoder)",
    "phase3":    "Phase 3   (RotatE triple score)",
    "phase4":    "Phase 4   (DOMINANT joint)",
}


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(scores: dict[str, float], manifest: dict) -> dict:
    if not scores:
        return {}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    decile_k = max(1, n // 10)
    top_decile = {sku for sku, _ in ranked[:decile_k]}
    threshold  = ranked[decile_k - 1][1]
    all_vals   = [v for _, v in ranked]

    # #3 top-decile recall on the three graded anomaly types
    graded_types = set(manifest.get("top_decile_types", []))
    problem = [
        p["sku_id"] for p in manifest["planted"]
        if p["anomaly_type"] in graded_types and p["sku_id"] in scores
    ]
    in_top = [sku for sku in problem if sku in top_decile]
    topdecile_recall = len(in_top) / len(problem) if problem else 0.0

    # #4 healthy in lower half of score range
    range_mid = min(all_vals) + (max(all_vals) - min(all_vals)) / 2
    healthy = [h for h in manifest.get("healthy", []) if h in scores]
    healthy_lower = (
        sum(scores[h] < range_mid for h in healthy) / len(healthy)
        if healthy else 0.0
    )

    # #14 classification accuracy vs ground truth
    gt = [
        (g["sku_id"], g["label"])
        for g in manifest.get("ground_truth", [])
        if g["sku_id"] in scores
    ]
    correct = sum(
        int(("confirmed_anomaly" if scores[sku] >= threshold else "valid") == label)
        for sku, label in gt
    )
    classif_acc = correct / len(gt) if gt else 0.0

    # AUC-ROC (bonus)
    auc = _auc(scores, manifest)

    # Mean scores by planted type (diagnostic)
    by_type: dict[str, list[float]] = {}
    for p in manifest["planted"]:
        if p["sku_id"] in scores:
            by_type.setdefault(p["anomaly_type"], []).append(scores[p["sku_id"]])
    hvals = [scores[h] for h in healthy if h in scores]
    if hvals:
        by_type["healthy"] = hvals

    return {
        "n":               n,
        "topdecile_recall": topdecile_recall,
        "healthy_lower":   healthy_lower,
        "classif_acc":     classif_acc,
        "auc":             auc,
        "by_type":         {t: float(np.mean(v)) for t, v in by_type.items()},
        "passes":          (
            topdecile_recall >= MIN_TOPDECILE_RECALL
            and healthy_lower   >= MIN_HEALTHY_LOWER
            and classif_acc     >= MIN_CLASSIF_ACC
        ),
    }


def _auc(scores: dict[str, float], manifest: dict) -> float:
    """AUC-ROC over ground-truth labeled nodes. Returns 0.5 if scikit-learn missing."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")

    gt = manifest.get("ground_truth", [])
    y_true, y_score = [], []
    for g in gt:
        sku, label = g["sku_id"], g["label"]
        if sku in scores:
            y_true.append(1 if label == "confirmed_anomaly" else 0)
            y_score.append(scores[sku])

    if len(set(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(results: dict[str, dict]):
    width = 66
    print("\n" + "═" * width)
    print("  REFLEXIVE KG — METHOD COMPARISON")
    print("═" * width)
    print(f"  {'Method':<34} {'#3 Recall':>9} {'#4 Hlthy':>8} {'#14 Acc':>7} {'AUC':>6} {'Pass'}")
    print(f"  {'-'*34} {'-'*9} {'-'*8} {'-'*7} {'-'*6} {'-'*4}")

    for method, r in results.items():
        if not r:
            label = METHOD_LABELS.get(method, method)
            print(f"  {label:<34}  {'(no scores — run the script first)':>36}")
            continue
        label  = METHOD_LABELS.get(method, method)
        auc_s  = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "  —  "
        flag   = "✓" if r["passes"] else "✗"
        print(
            f"  {label:<34} "
            f"{r['topdecile_recall']:>9.3f} "
            f"{r['healthy_lower']:>8.3f} "
            f"{r['classif_acc']:>7.3f} "
            f"{auc_s:>6} "
            f"  {flag}"
        )

    print(f"\n  Bars: recall≥{MIN_TOPDECILE_RECALL}  healthy≥{MIN_HEALTHY_LOWER}  acc≥{MIN_CLASSIF_ACC}")
    print("═" * width)

    # Per-type mean scores for all methods that ran
    all_types = sorted({
        t for r in results.values() if r
        for t in r.get("by_type", {})
    })
    if all_types:
        print("\n  Mean anomaly score by planted type:")
        col_w = 10
        header = f"  {'Type':<20}" + "".join(
            f"{m[:col_w-1]:>{col_w}}"
            for m in results
            if results[m]
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in all_types:
            row = f"  {t:<20}"
            for m, r in results.items():
                if r:
                    v = r["by_type"].get(t, float("nan"))
                    row += f"{'—':>{col_w}}" if np.isnan(v) else f"{v:>{col_w}.3f}"
            print(row)


# ─────────────────────────────────────────────────────────────────────────────
# PYTEST ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def _best_result():
    """Return the best available result across methods (for pytest)."""
    manifest = json.load(open(MANIFEST_PATH))
    driver   = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for method, loader in LOADERS.items():
            scores = loader(session, manifest["cohort_tag"])
            r = evaluate(scores, manifest)
            if r:
                driver.close()
                return r
    driver.close()
    return {}


def test_best_method_top_decile_recall():
    assert _best_result().get("topdecile_recall", 0) >= MIN_TOPDECILE_RECALL


def test_best_method_healthy_lower_half():
    assert _best_result().get("healthy_lower", 0) >= MIN_HEALTHY_LOWER


def test_best_method_classification_accuracy():
    assert _best_result().get("classif_acc", 0) >= MIN_CLASSIF_ACC


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare all reflection methods against seeded anomaly ground truth"
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=list(LOADERS),
        default=list(LOADERS),
        help="Methods to evaluate (default: all)",
    )
    args = parser.parse_args()

    manifest = json.load(open(MANIFEST_PATH))
    cohort   = manifest["cohort_tag"]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results: dict[str, dict] = {}

    with driver.session() as session:
        for method in args.methods:
            print(f"  Loading scores: {method} ...")
            scores = LOADERS[method](session, cohort)
            results[method] = evaluate(scores, manifest)
            n = results[method].get("n", 0)
            print(f"    → {n} scored nodes")

    driver.close()
    print_comparison(results)

    # Exit 0 only if at least one method passes all three bars
    if any(r.get("passes") for r in results.values() if r):
        print("\n  At least one method PASSES all acceptance bars. ✓\n")
        sys.exit(0)
    else:
        print("\n  No method passes all acceptance bars. ✗\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
