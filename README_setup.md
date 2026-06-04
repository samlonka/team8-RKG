# Reflexive KG — Setup & Run Order

## Prerequisites

- Python 3.11+
- Neo4j 5.x running locally (Docker recommended)
- Your data files in the `data/` folder

## Step 0 — Neo4j via Docker

```bash
docker run -d \
  --name neo4j-rkg \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.20
```

Open http://localhost:7474 to verify it's running.

## Step 1 — Python environment

```bash
cd reflexive_kg
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2 — Environment variables

```bash
cp .env.example .env
# Edit .env with your Neo4j password
```

## Step 3 — Place data files

```
reflexive_kg/
└── data/
    ├── vor_sku_data.csv          # Master / Global SKU list
    ├── SKU_Export.xlsx           # Vendor SKU export
    ├── Global_sku.csv            # Legacy master (optional)
    └── sample_vendor_SKU_Export.xlsx
```

## Step 4 — Run pipeline in order

```bash
# Create Neo4j schema (run once)
python 01_schema.py

# Load data + generate self_emb (takes ~2-5 min for embeddings)
python 02_seed_data.py

# Compute reflect_emb (attention + edge severity) + print anomaly scores
python 03_reflection.py

# Show top-50 anomalies for GlobalSKU only
python 03_reflection.py --label GlobalSKU --top 50

# Skip recomputation, just print scores
python 03_reflection.py --scores-only
```

## Step 5 — Lifecycle anomalies + demo

```bash
# Plant handbook failure patterns + recompute reflect_emb for cohort
python 05_synthesize_lifecycle.py

# Verify acceptance criteria (effective scores + per-type thresholds)
python 06_evaluate.py

# Stratified catalog-scale check (1k–10k SKUs, UPC-missing strata)
python 06_scale_evaluate.py --sample 5000

python -m pytest test_agents.py test_full_criteria.py -v

# Run all 6 hackathon demo scenarios (unified pipeline)
python 04_agent_pipeline.py --demo --no-llm

# Jupyter demo notebook (AC 18 — assumes steps above already run)
jupyter notebook RKG_Demo.ipynb

# Analyst workbench UI (Streamlit): risk inbox → SKU chain, training gate, blast radius
pip install streamlit
streamlit run ui/app.py

# Vendor vs master evaluation report (UPC join; add --full for brand/package fuzzy match)
python scripts/eval_vendor_master.py
python ingest_vendor.py data/SKU_Export.xlsx
python ingest_vendor.py --report

# Synthetic 50-row vendor (new warehouse: SYNTH TEST VENDOR) for QA
python scripts/generate_synthetic_vendor.py
python scripts/test_synthetic_vendor.py --full
python scripts/eval_vendor_master.py --vendor data/synthetic_vendor_50.xlsx
python ingest_vendor.py data/synthetic_vendor_50.xlsx
```

Handbook-oriented modules: `reflection_core.py` (default reflection), `scoring.py` (per-type boosts/thresholds), `06_scale_evaluate.py`.

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| `all-mpnet-base-v2` over `all-MiniLM-L6-v2` | Better semantic quality for short brand/supplier phrases |
| `brand_family` for embedding, not `brand_name` | `brand_family` is human-readable (`BIG RED`); `brand_name` is encoded slug (`AQUA_WTR`) |
| `package_category_name` not `package_type_id` | `1.5L PL 1/12` is embeddable; `556` is not |
| UPC as boolean flag in text | `012000001598` has no semantic meaning to a sentence encoder |
| Neo4j for both graph + vectors | Neo4j 5.x native vector indexes — no PostgreSQL needed |
| Retail UPC as join key | 232 matches between Global and Vendor datasets |
