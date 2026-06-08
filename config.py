"""
config.py — Central configuration for Reflexive KG
All tuneable parameters live here. Domain team can adjust REL_WEIGHTS
without touching any algorithm code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j connection ──────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── Embedding model ───────────────────────────────────────────────────────────
# all-mpnet-base-v2: better semantic quality for short brand/supplier phrases
# all-MiniLM-L6-v2: faster, lighter — swap here if GPU memory is tight
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM   = 768

# ── PostgreSQL (AWS RDS) — set all values in .env ─────────────────────────────
POSTGRES_HOST             = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT             = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB               = os.getenv("POSTGRES_DB", "")
POSTGRES_USER             = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD         = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_SECRET_ID        = os.getenv("POSTGRES_SECRET_ID", "")
POSTGRES_CONNECT_TIMEOUT  = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "15"))
POSTGRES_SSLMODE          = os.getenv("POSTGRES_SSLMODE", "").strip() or None

# ── Data file paths ───────────────────────────────────────────────────────────
# Place your source files in the data/ folder (or update these paths)
# Master global catalog (handbook §2 — GlobalSKU / master SKU list)
GLOBAL_SKU_CSV    = "data/vor_sku_data.csv"
# One tenant's product list at import (handbook §2 — TenantSKU); also used by 02_seed_data.py
TENANT_SKU_XLSX   = "data/SKU_Export.xlsx"
VENDOR_SKU_XLSX   = TENANT_SKU_XLSX   # backward-compatible alias

# ── Relation-type weights ─────────────────────────────────────────────────────
# Higher weight = this relationship type has stronger influence on reflect_emb.
# Tune these to match domain importance without changing any algorithm code.
REL_WEIGHTS = {
    # ── SKU-lifecycle relationships (handbook §3.2 — these drive the anomaly) ──
    "MERGED_INTO":      3.0,   # merge history = strongest data-quality signal
    "SCANNED_ON":       2.5,   # production scan failures
    "MAPS_TO":          2.0,   # TenantSKU → global SKU (auto-map errors)
    "TRAINED_WITH":     2.0,   # TrainingImage → SKU (missing = evidence gap)
    "FUZZY_MATCH":      1.8,   # near-duplicate brand records
    "BELONGS_TO_BRAND": 1.5,   # SKU → Brand (brand mismatch cascade)
    "HAS_PACKAGE":      1.5,   # SKU → PackageType
    "USED_BY":          1.0,   # SKU → Customer (shared-SKU boundary)
    # ── Vendor-catalog relationships kept from the original build ──
    "SUPPLIED_BY":      1.5,
    "MADE_BY":          1.0,
    "IN_CLASS":         1.0,
    # Default for any relationship type not listed above
    "_DEFAULT":         1.0,
}

# ── Anomaly scoring ───────────────────────────────────────────────────────────
# Critic rejects causal chains with confidence below this threshold
CRITIC_CONFIDENCE_THRESHOLD = 0.65

# Minimum entities per hop for a chain to be considered valid
MIN_ENTITIES_PER_HOP = 3   # relaxed from 5 for POC dataset size

# Top-N chains the Critic returns
CRITIC_TOP_N = 3

# ── Anomaly score thresholds ──────────────────────────────────────────────────
# Score = 1 - cosine_similarity(self_emb, reflect_emb)
# 0.0 = identical (healthy), 1.0 = completely divergent (anomalous)
ANOMALY_HIGH_RISK  = 0.75   # flag for immediate review
ANOMALY_MEDIUM_RISK = 0.50  # flag for monitoring
ANOMALY_LOW_RISK   = 0.25   # considered healthy

# ── Fuzzy brand matching ──────────────────────────────────────────────────────
# Brand text is just "brand <family>", which all-mpnet-base-v2 maps into a tiny,
# dense region of vector space — ~43% of ALL brand pairs score >= 0.98. A fixed
# similarity threshold therefore matches tens of millions of pairs and blows up
# the graph. We instead keep each brand's TOP-K nearest neighbours (bounded to
# ~N*K edges), with a floor to drop genuinely dissimilar matches.
BRAND_FUZZY_TOP_K  = 5      # max FUZZY_MATCH partners kept per brand
BRAND_FUZZY_MIN_SIM = 0.90  # floor: ignore neighbours below this cosine sim
# Legacy global threshold (kept for reference; no longer used for matching)
BRAND_FUZZY_MATCH_THRESHOLD = 0.80

# ── Phase 1: Edge-property severity multipliers ───────────────────────────────
# Applied on top of REL_WEIGHTS when a neighbour node signals an anomaly via
# its own properties (e.g. a Pallet with outcome='failure' is a stronger signal
# than a healthy one even though both carry the same SCANNED_ON rel type).
EDGE_SEVERITY = {
    "outcome:failure":          1.5,   # Pallet scan failure
    "status:conflicted":        2.0,   # MergeEvent conflict
    "rollback_available:False": 1.3,   # MergeEvent, no rollback option
    "match_method:fuzzy":       1.5,   # TenantSKU fuzzy (uncertain) match
}

# ── Advanced reflection variants (03d_advanced_reflection.py) ────────────────
# Temporal decay: exp(-λ × days_since_event).  λ = 0.01 → ~70-day half-life.
TEMPORAL_DECAY_LAMBDA = 0.01

# ── LOF + Ensemble (11_ensemble.py) ──────────────────────────────────────────
LOF_N_NEIGHBORS  = 50    # LocalOutlierFactor neighbourhood size — higher reduces duplicate-value warnings
ENSEMBLE_C       = 1.0   # logistic regression regularisation strength

# ── Attention reflection (03c_reflection_attention.py) ───────────────────────
# Temperature controls how sharply attention focuses on the most divergent
# neighbours.  τ < 1 → winner-takes-all.  τ > 1 → approaches uniform mean.
# τ = 1.0 is the standard softmax — a good starting point.
ATTN_TEMPERATURE = 1.0

# ── Phase 2: R-GCN hyperparameters ───────────────────────────────────────────
# Per-relation weight matrices replace the scalar REL_WEIGHTS.
# num_bases uses basis decomposition to share parameters across relation types,
# reducing overfitting when num_relations is small relative to training data.
RGCN_HIDDEN_DIM = 256
RGCN_OUT_DIM    = 768   # must equal EMBEDDING_DIM — keeps cosine comparison valid
RGCN_NUM_BASES  = 4     # basis vectors shared across 11 relation types
RGCN_EPOCHS     = 50
RGCN_LR         = 0.01
RGCN_DROPOUT    = 0.2

# ── Phase 3: KGE (RotatE) hyperparameters ────────────────────────────────────
KGE_MODEL         = "RotatE"
KGE_EMBEDDING_DIM = 64    # complex-space dimension per entity
KGE_EPOCHS        = 100
KGE_LR            = 1e-3

# ── Phase 4: DOMINANT hyperparameters ────────────────────────────────────────
# alpha=0 → structure-only; alpha=1 → attribute-only; 0.5 balances both.
DOMINANT_ALPHA      = 0.5
DOMINANT_HIDDEN_DIM = 256
DOMINANT_BOTTLENECK = 64   # encoder bottleneck to keep structure decoder tractable
DOMINANT_EPOCHS     = 100
DOMINANT_LR         = 1e-3

# ── Vendor ingestion pipeline (ingest_vendor.py) ─────────────────────────────
MATCH_AUTO_THRESHOLD      = 0.90   # confidence ≥ this → auto-match, no review needed
MATCH_REVIEW_THRESHOLD    = 0.65   # confidence ≥ this → human review queue
#                                    confidence < MATCH_REVIEW_THRESHOLD → GlobalSKUDraft
MATCH_ANN_TOP_K           = 10     # ANN candidates per TenantSKU
MATCH_ANOMALY_ALERT_DELTA = 0.10   # flag GlobalSKU if anomaly_attn rises by ≥ this after ingest

# ── Amazon Bedrock LLM (all four agents — Claude Opus 4.7) ───────────────────
AWS_REGION       = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-opus-4-7")
BEDROCK_REGION   = AWS_REGION
# When True, Supervisor/Planner/Doer/Critic call Bedrock first; rule-based logic is fallback.
AGENT_USE_LLM    = os.getenv("AGENT_USE_LLM", "true").lower() in ("1", "true", "yes")

# ── Batch processing ──────────────────────────────────────────────────────────
EMBED_BATCH_SIZE = 64   # sentence-transformer batch size

# ── Node labels (Neo4j) ───────────────────────────────────────────────────────
NODE_LABELS = {
    "global_sku":    "GlobalSKU",
    "tenant_sku":    "TenantSKU",
    "brand":         "Brand",
    "customer":      "Customer",
    "training_image": "TrainingImage",
    "merge_event":   "MergeEvent",
    "pallet":        "Pallet",
    "package_type":  "PackageType",
    "manufacturer":  "Manufacturer",
    "supplier":      "Supplier",
    "product_class": "ProductClass",
}

# ── Field mappings: Global SKU CSV ────────────────────────────────────────────
# Maps CSV column names to canonical property names used in Neo4j
GLOBAL_SKU_FIELDS = {
    "sku_id":                "sku_id",
    "upc":                   "upc",
    "status":                "status",
    "package_type_id":       "package_type_id",
    "package_category_name": "package_category_name",
    "short_description":     "short_description",
    "weight":                "weight",
    "height":                "height",
    "length":                "length",
    "width":                 "width",
    "manufacturer":          "manufacturer",
    "product_category":      "product_category",
    "units_per_case":        "units_per_case",
    "brand_id":              "brand_id",
    "brand_family":          "brand_family",
    "brand_name":            "brand_name",
    "is_imaged_on_training_station": "is_imaged_on_training_station",
    "is_imaged_on_wrapper":          "is_imaged_on_wrapper",
    "is_review_needed":              "is_review_needed",
    "vor_reference_number":          "vor_reference_number",
    "creation_date":                 "creation_date",
}

# ── Field mappings: Vendor SKU XLSX ──────────────────────────────────────────
VENDOR_SKU_FIELDS = {
    "• Warehouse":              "warehouse",
    "Product ID":               "product_id",
    "Product Description":      "product_description",
    "Brand":                    "brand",
    "Supplier":                 "supplier",
    "Product Class":            "product_class",
    "Selling Units per Case":   "units_per_case",
    "Unit Weight":              "unit_weight",
    "Case Length (Inches)":     "case_length",
    "Case Width (Inches)":      "case_width",
    "Case Height (Inches)":     "case_height",
    "Case UPC":                 "case_upc",
    "Retail UPC":               "retail_upc",
    "Eaches UPC":               "eaches_upc",
}
