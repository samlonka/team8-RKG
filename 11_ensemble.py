"""
11_ensemble.py — LOF anomaly detection + logistic regression score ensemble.

Runs after all reflection methods have been computed.  Combines every available
anomaly score into one calibrated ensemble probability per entity.

──────────────────────────────────────────────────────────────────────────────
Step 1 — LOF  (anomaly_lof)

  Local Outlier Factor on the divergence vector: self_emb − reflect_emb_attn.
  The difference captures HOW an entity diverges from its neighbourhood, not
  just how much.  Entities with rare divergence patterns score high even if
  their cosine distance is only moderate.

  Requires: self_emb + reflect_emb_attn on each node.

──────────────────────────────────────────────────────────────────────────────
Step 2 — Ensemble  (anomaly_ensemble)

  Logistic regression trained on ALL available method scores using the analyst
  ground-truth labels from seed_manifest.json.

  Features used (missing methods imputed with median):
    baseline · attention · phase1 · phase2 · phase3 · phase4
    lof · reflect2 · temporal · degnorm

  The model output is a calibrated probability [0, 1] that the entity is a
  confirmed anomaly.  Coefficients printed after training show which method
  contributes most per anomaly type.

  Requires: seed_manifest.json (written by 05_synthesize_lifecycle.py).

Usage:
    python 11_ensemble.py                              # run both steps on GlobalSKU
    python 11_ensemble.py --label ALL                 # run on all base labels
    python 11_ensemble.py --skip-lof                  # ensemble only
    python 11_ensemble.py --skip-ensemble             # LOF only
    python 11_ensemble.py --scores-only               # print current scores
"""

from __future__ import annotations

import argparse
import json
import uuid

import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase

try:
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required for 11_ensemble.py.  "
        "Install with: pip install scikit-learn"
    ) from e

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    LOF_N_NEIGHBORS, ENSEMBLE_C,
    ANOMALY_HIGH_RISK, ANOMALY_MEDIUM_RISK,
)
from score_log import log_batch_run, ensure_indexes as _ensure_log_indexes

MANIFEST_PATH = "seed_manifest.json"

ALL_LABELS = [
    "GlobalSKU", "TenantSKU", "Brand", "PackageType",
    "Manufacturer", "Supplier", "ProductClass",
]

PK_MAP = {
    "GlobalSKU":    "sku_id",   "TenantSKU":  "tenant_sku_id",
    "Brand":        "brand_id", "PackageType": "package_type_id",
    "Manufacturer": "name",     "Supplier":    "name",
    "ProductClass": "name",
}

# Anomaly score properties fed into the ensemble.
# anomaly_reflect2 excluded: circular on leaf-node topology (AUC 0.385).
# anomaly_degnorm  excluded: degree normalisation suppresses the anomaly signal.
ALL_METHODS = [
    "anomaly_baseline",
    "anomaly_attn",
    "anomaly_dir",
    "anomaly_rgcn",
    "triple_anomaly_score",
    "dominant_score",
    "anomaly_lof",
    "anomaly_temporal",
]

METHOD_SHORT = {
    "anomaly_baseline":       "baseline",
    "anomaly_attn":           "attention",
    "anomaly_dir":            "phase1",
    "anomaly_rgcn":           "phase2",
    "triple_anomaly_score":   "phase3",
    "dominant_score":         "phase4",
    "anomaly_lof":            "lof",
    "anomaly_reflect2":       "reflect2",
    "anomaly_temporal":       "temporal",
    "anomaly_degnorm":        "degnorm",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOCAL OUTLIER FACTOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_lof(session, label: str,
                cohort_filter: str | None = None) -> dict[str, float]:
    """
    Compute LOF on the divergence vector: self_emb − reflect_emb_attn.

    cohort_filter: when provided, restrict to nodes where n.cohort = cohort_filter.
    This is critical — running LOF on the full graph makes cohort SKUs (which have
    unique lifecycle neighbours) look like global outliers regardless of health,
    destroying within-cohort discrimination.  Always pass the cohort tag when
    evaluating against seed_manifest ground truth.

    Returns {entity_id: lof_score} normalised to [0, 1].
    """
    pk = PK_MAP[label]

    # Prefer attention embedding; fall back to plain reflect_emb
    attn_count = session.run(
        f"MATCH (n:{label}) WHERE n.reflect_emb_attn IS NOT NULL RETURN count(n) AS c"
    ).single()["c"]
    reflect_field = "reflect_emb_attn" if attn_count > 0 else "reflect_emb"

    cohort_clause = f"AND n.cohort = $cohort" if cohort_filter else ""
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.self_emb IS NOT NULL AND n.{reflect_field} IS NOT NULL
        {cohort_clause}
        RETURN n.{pk} AS id, n.self_emb AS se, n.{reflect_field} AS re
        """,
        cohort=cohort_filter,
    ).data()

    if len(rows) < LOF_N_NEIGHBORS + 1:
        print(f"  {label}: only {len(rows)} nodes — need ≥ {LOF_N_NEIGHBORS + 1} for LOF")
        return {}

    ids      = [r["id"] for r in rows]
    se_arr   = np.array([r["se"] for r in rows], dtype=np.float32)
    re_arr   = np.array([r["re"] for r in rows], dtype=np.float32)

    # Divergence vector: self − reflect (normalised per entity)
    diff     = se_arr - re_arr                                  # (N, 768)
    norms    = np.linalg.norm(diff, axis=1, keepdims=True)
    norms    = np.where(norms < 1e-8, 1.0, norms)
    diff_n   = diff / norms                                     # unit divergence direction

    k = min(LOF_N_NEIGHBORS, len(ids) - 1)
    lof = LocalOutlierFactor(n_neighbors=k, metric="cosine", contamination="auto")
    lof.fit_predict(diff_n)

    # negative_outlier_factor_: more negative = more anomalous
    raw  = -lof.negative_outlier_factor_
    lo, hi = raw.min(), raw.max()
    norm_scores = ((raw - lo) / (hi - lo + 1e-8)).tolist()

    print(f"  {label}: LOF on {len(ids):,} nodes  (k={k}, field={reflect_field})")
    return dict(zip(ids, norm_scores))


def write_lof(session, label: str, scores: dict[str, float]):
    pk = PK_MAP[label]
    rows = [{"id": str(eid), "score": round(float(s), 4)}
            for eid, s in scores.items()]
    for i in range(0, len(rows), 500):
        session.run(
            f"""
            UNWIND $rows AS r
            MATCH (n:{label} {{{pk}: r.id}})
            SET n.anomaly_lof = r.score
            """,
            rows=rows[i : i + 500],
        )
    print(f"    → {len(rows):,} LOF scores written")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def _load_scores(session, label: str) -> tuple[list[str], np.ndarray]:
    """
    Load all available anomaly scores for every entity of `label`.
    Returns (entity_ids, feature_matrix) where missing values are np.nan.
    """
    pk     = PK_MAP[label]
    fields = ", ".join(f"n.{m} AS {m.replace('.','_')}" for m in ALL_METHODS)

    rows = session.run(
        f"""
        MATCH (n:{label}) WHERE n.anomaly_baseline IS NOT NULL
        RETURN n.{pk} AS id, {fields}
        """
    ).data()

    ids = [str(r["id"]) for r in rows]
    X   = np.array(
        [[r.get(m.replace(".", "_")) or np.nan for m in ALL_METHODS]
         for r in rows],
        dtype=np.float32,
    )
    return ids, X


def _load_ground_truth(manifest_path: str) -> dict[str, int]:
    """Return {entity_id: 1_or_0} from seed_manifest ground truth labels."""
    try:
        manifest = json.load(open(manifest_path))
    except FileNotFoundError:
        return {}
    return {
        str(g["sku_id"]): (1 if g["label"] == "confirmed_anomaly" else 0)
        for g in manifest.get("ground_truth", [])
    }


def train_ensemble(
    session, label: str, manifest_path: str
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Train a logistic regression on ground-truth labeled entities, then apply
    to all entities of `label`.

    Returns:
        scores      : {entity_id: ensemble_probability}
        importances : {method_short_name: coefficient}
    """
    gt = _load_ground_truth(manifest_path)
    if not gt:
        print(f"  No ground truth found in {manifest_path}")
        return {}, {}

    ids, X = _load_scores(session, label)
    if len(ids) == 0:
        print(f"  No entities with anomaly_baseline for {label}")
        return {}, {}

    # ── Drop all-NaN columns (methods not yet run) ───────────────────────────
    # SimpleImputer silently drops columns with zero observed values, which
    # shrinks X and makes clf.coef_ shorter than ALL_METHODS.  We track the
    # surviving column mask explicitly so importances stay aligned.
    has_data    = ~np.all(np.isnan(X), axis=0)          # (n_methods,) bool
    active_methods = [m for m, ok in zip(ALL_METHODS, has_data) if ok]
    X_active    = X[:, has_data]

    active_short = [METHOD_SHORT.get(m, m) for m in active_methods]
    skipped_short = [METHOD_SHORT.get(m, m)
                     for m, ok in zip(ALL_METHODS, has_data) if not ok]
    print(f"  Active  ({len(active_methods)}): {', '.join(active_short)}")
    if skipped_short:
        print(f"  Skipped (no data yet): {', '.join(skipped_short)}")

    # ── Impute remaining NaN values within active columns ────────────────────
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_active)

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    # ── Build training set from labeled entities only ─────────────────────────
    labeled_mask = [i for i, eid in enumerate(ids) if eid in gt]
    if len(labeled_mask) < 4:
        print(f"  {label}: only {len(labeled_mask)} labeled entities — need ≥ 4 for ensemble")
        return {}, {}

    X_train = X_sc[labeled_mask]
    y_train = [gt[ids[i]] for i in labeled_mask]
    n_pos   = sum(y_train)
    n_neg   = len(y_train) - n_pos

    print(f"  {label}: training on {len(y_train)} labels  "
          f"({n_pos} anomalous, {n_neg} healthy)")

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = LogisticRegression(
        C=ENSEMBLE_C,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    # Training AUC
    train_proba = clf.predict_proba(X_train)[:, 1]
    try:
        train_auc = roc_auc_score(y_train, train_proba)
        print(f"  Training AUC: {train_auc:.4f}")
    except ValueError:
        pass

    # ── Feature importances — aligned to active_methods only ─────────────────
    coefs      = clf.coef_[0]                            # len == len(active_methods)
    importances = {
        METHOD_SHORT.get(m, m): round(float(coefs[i]), 4)
        for i, m in enumerate(active_methods)
    }
    sorted_imp = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)

    print(f"\n  Feature importances (logistic regression coefficients):")
    for name, coef in sorted_imp:
        bar = "█" * max(1, int(abs(coef) * 20))
        sign = "+" if coef >= 0 else "-"
        print(f"    {name:<12} {sign}{abs(coef):.4f}  {bar}")

    # ── Apply to all entities ─────────────────────────────────────────────────
    proba  = clf.predict_proba(X_sc)[:, 1]
    scores = {eid: round(float(p), 4) for eid, p in zip(ids, proba)}

    return scores, importances


def write_ensemble(session, label: str, scores: dict[str, float]):
    pk = PK_MAP[label]
    rows = [{"id": str(eid), "score": s} for eid, s in scores.items()]
    for i in range(0, len(rows), 500):
        session.run(
            f"""
            UNWIND $rows AS r
            MATCH (n:{label} {{{pk}: r.id}})
            SET n.anomaly_ensemble = r.score
            """,
            rows=rows[i : i + 500],
        )
    print(f"    → {len(rows):,} ensemble scores written")


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_top(session, label: str, score_prop: str, title: str, top_n: int):
    pk   = PK_MAP[label]
    rows = session.run(
        f"""
        MATCH (n:{label}) WHERE n.{score_prop} IS NOT NULL
        RETURN n.{pk} AS id, n.{score_prop} AS score
        ORDER BY score DESC LIMIT $n
        """,
        n=top_n,
    ).data()
    if not rows:
        return
    print(f"\n── {title}: {label} {'─'*35}")
    print(f"  {'ID':<18} {'Score':<8} Risk")
    print(f"  {'-'*18} {'-'*8} {'-'*6}")
    for r in rows:
        risk = ("HIGH"   if r["score"] >= ANOMALY_HIGH_RISK   else
                "MEDIUM" if r["score"] >= ANOMALY_MEDIUM_RISK else "LOW")
        print(f"  {str(r['id']):<18} {r['score']:<8.4f} {risk}")


def compare_ensemble_vs_best(session, label: str, top_n: int = 15):
    """Side-by-side: best individual method (attention) vs ensemble."""
    pk   = PK_MAP[label]
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.anomaly_ensemble IS NOT NULL AND n.anomaly_attn IS NOT NULL
        RETURN n.{pk} AS id,
               n.anomaly_attn     AS attn,
               n.anomaly_ensemble AS ensemble
        ORDER BY ensemble DESC LIMIT $n
        """,
        n=top_n,
    ).data()
    if not rows:
        return
    print(f"\n── Attention vs Ensemble: {label} (top {top_n} by ensemble) ──────────")
    print(f"  {'ID':<18} {'Attention':>9} {'Ensemble':>9} {'Delta':>8}")
    print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*8}")
    for r in rows:
        delta = r["ensemble"] - r["attn"]
        flag  = " ← ensemble finds more" if delta > 0.05 else ""
        print(f"  {str(r['id']):<18} {r['attn']:>9.4f} {r['ensemble']:>9.4f} "
              f"{delta:>+8.4f}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LOF anomaly detection + logistic regression score ensemble"
    )
    parser.add_argument("--label",         default="GlobalSKU",
                        help="Node label (default: GlobalSKU; use ALL for all base labels)")
    parser.add_argument("--top",           type=int, default=20)
    parser.add_argument("--skip-lof",      action="store_true",
                        help="Skip LOF computation")
    parser.add_argument("--skip-ensemble", action="store_true",
                        help="Skip ensemble training")
    parser.add_argument("--scores-only",   action="store_true",
                        help="Skip computation, just print current scores")
    parser.add_argument("--compare",       action="store_true",
                        help="Show attention vs ensemble side-by-side")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    labels = ALL_LABELS if args.label == "ALL" else [args.label]

    # Load cohort tag from manifest so LOF runs on the labeled population only
    cohort_tag: str | None = None
    try:
        import json as _json
        cohort_tag = _json.load(open(MANIFEST_PATH)).get("cohort_tag")
    except (FileNotFoundError, KeyError):
        pass

    with driver.session() as session:
        _ensure_log_indexes(session)
        run_id = str(uuid.uuid4())

        if not args.scores_only:

            # ── LOF ───────────────────────────────────────────────────────────
            if not args.skip_lof:
                print("\n── Step 1: Local Outlier Factor ─────────────────────────────────")
                if cohort_tag:
                    print(f"  Restricting to cohort '{cohort_tag}' (avoids global-population bias)")
                for label in labels:
                    lof_scores = compute_lof(session, label, cohort_filter=cohort_tag)
                    if lof_scores:
                        write_lof(session, label, lof_scores)
                        _, n = log_batch_run(session, label, "lof", run_id=run_id)
                        print(f"    score log: {n:,} entries  run={run_id[:8]}…")

            # ── Ensemble ──────────────────────────────────────────────────────
            if not args.skip_ensemble:
                print("\n── Step 2: Score Ensemble ───────────────────────────────────────")
                for label in labels:
                    ens_scores, _ = train_ensemble(session, label, MANIFEST_PATH)
                    if ens_scores:
                        write_ensemble(session, label, ens_scores)
                        _, n = log_batch_run(session, label, "ensemble", run_id=run_id)
                        print(f"    score log: {n:,} entries  run={run_id[:8]}…")

        # ── Report ────────────────────────────────────────────────────────────
        print(f"\n── Top anomalies ─────────────────────────────────────────────────")
        for label in labels:
            if not args.skip_lof:
                print_top(session, label, "anomaly_lof",      "LOF",      args.top)
            if not args.skip_ensemble:
                print_top(session, label, "anomaly_ensemble", "Ensemble", args.top)
            if args.compare:
                compare_ensemble_vs_best(session, label, top_n=args.top)

    driver.close()
    print("\nEnsemble pipeline complete.\n")


if __name__ == "__main__":
    main()
