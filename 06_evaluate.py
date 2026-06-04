"""
06_evaluate.py — Seeded-anomaly evaluation (handbook acceptance criteria #3, #4, #14).

Uses scoring.py for effective anomaly scores (per-type boosts + thresholds).

Usage:  python 06_evaluate.py
"""

import numpy as np

from scoring import (
    classify_label,
    load_cohort_scores,
    ranked_effective,
    MANIFEST_PATH,
)

MIN_TOPDECILE_RECALL = 0.80
MIN_HEALTHY_LOWER_HALF = 0.80
MIN_CLASSIFICATION_ACC = 0.85


def evaluate():
    manifest, score_map = load_cohort_scores()
    if not manifest or not score_map:
        raise FileNotFoundError(f"Run 05_synthesize_lifecycle.py first ({MANIFEST_PATH})")

    ranked = ranked_effective(score_map)
    scores = {sku: d["effective"] for sku, d in score_map.items()}
    n = len(ranked)
    decile_k = max(1, n // 10)
    top_decile = {sku for sku, _ in ranked[:decile_k]}
    threshold = ranked[decile_k - 1][1]
    median = float(np.median([v for _, v in ranked]))

    graded = set(manifest["top_decile_types"])
    problem = [
        p["sku_id"]
        for p in manifest["planted"]
        if p["anomaly_type"] in graded and p["sku_id"] in scores
    ]
    in_top = [sku for sku in problem if sku in top_decile]
    topdecile_recall = len(in_top) / len(problem) if problem else 0.0

    all_vals = [v for _, v in ranked]
    range_mid = min(all_vals) + (max(all_vals) - min(all_vals)) / 2
    healthy = [h for h in manifest["healthy"] if h in scores]
    healthy_lower_half = (
        sum(scores[h] < range_mid for h in healthy) / len(healthy) if healthy else 0.0
    )
    healthy_below_median_rank = (
        sum(scores[h] < median for h in healthy) / len(healthy) if healthy else 0.0
    )

    gt = [(g["sku_id"], g["label"]) for g in manifest["ground_truth"] if g["sku_id"] in scores]
    correct = 0
    for sku, label in gt:
        d = score_map[sku]
        pred = classify_label(d["effective"], threshold, d["planted_type"])
        correct += int(pred == label)
    classification_acc = correct / len(gt) if gt else 0.0

    return {
        "n": n,
        "decile_k": decile_k,
        "threshold": threshold,
        "median": median,
        "topdecile_recall": topdecile_recall,
        "n_problem": len(problem),
        "healthy_lower_half": healthy_lower_half,
        "n_healthy": len(healthy),
        "healthy_below_median_rank": healthy_below_median_rank,
        "range_mid": range_mid,
        "classification_acc": classification_acc,
        "n_labels": len(gt),
        "ranked": ranked,
        "scores": scores,
        "score_map": score_map,
        "manifest": manifest,
    }


def report():
    m = evaluate()
    print("\n── Seeded-Anomaly Evaluation ───────────────────────────────")
    print(f"  cohort SKUs scored: {m['n']}   top-decile = top {m['decile_k']}")
    print(f"  score: median={m['median']:.3f}  top-decile threshold={m['threshold']:.3f}")
    print(
        f"\n  #3  top-decile recall (all planted types): "
        f"{m['topdecile_recall']:.3f}  ({m['n_problem']} planted)   bar {MIN_TOPDECILE_RECALL}"
    )
    print(
        f"  #4  healthy in lower half: {m['healthy_lower_half']:.3f}  "
        f"({m['n_healthy']} healthy)   bar {MIN_HEALTHY_LOWER_HALF}"
    )
    print(
        f"  #14 classification accuracy: {m['classification_acc']:.3f}  "
        f"({m['n_labels']} labels)   bar {MIN_CLASSIFICATION_ACC}"
    )

    by_type = {}
    for p in m["manifest"]["planted"]:
        sku = p["sku_id"]
        if sku in m["score_map"]:
            by_type.setdefault(p["anomaly_type"], []).append(m["score_map"][sku]["effective"])
    print("\n  mean effective anomaly score by planted type:")
    for t, v in sorted(by_type.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"    {t:16} {np.mean(v):.3f}  (n={len(v)})")
    hv = [m["scores"][h] for h in m["manifest"]["healthy"] if h in m["scores"]]
    print(f"    {'healthy':16} {np.mean(hv):.3f}  (n={len(hv)})")

    print("\n  top 10 by effective anomaly score:")
    plant = {p["sku_id"]: p["anomaly_type"] for p in m["manifest"]["planted"]}
    for sku, sc in m["ranked"][:10]:
        print(f"    {sku:>10}  {sc:.3f}  {plant.get(sku, 'healthy/other')}")

    ok = (
        m["topdecile_recall"] >= MIN_TOPDECILE_RECALL
        and m["healthy_lower_half"] >= MIN_HEALTHY_LOWER_HALF
        and m["classification_acc"] >= MIN_CLASSIFICATION_ACC
    )
    print("\n  RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return m, ok


def test_top_decile_recall():
    assert evaluate()["topdecile_recall"] >= MIN_TOPDECILE_RECALL


def test_healthy_lower_half():
    assert evaluate()["healthy_lower_half"] >= MIN_HEALTHY_LOWER_HALF


def test_classification_accuracy():
    assert evaluate()["classification_acc"] >= MIN_CLASSIFICATION_ACC


if __name__ == "__main__":
    import sys

    _, ok = report()
    sys.exit(0 if ok else 1)
