# Reflexive KG — Setup & Run Order

**Usage guide (vendor ingest, agents, API, scenarios):** see [README.md](README.md).

---

## Prerequisites

- Python 3.11+
- Neo4j 5.x running locally (Docker recommended)
- Amazon Bedrock access to **Claude Opus 4.7** (for agent pipeline — see [README.md](README.md#bedrock-agents))
- Data files in `data/` (master CSV + vendor Excel)

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

To wipe and reload:

```bash
# In Neo4j Browser or cypher-shell:
MATCH (n) DETACH DELETE n;
```

Then re-run Steps 4–5 below.

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
# Edit .env: Neo4j password, AWS region, optional BEDROCK_MODEL_ID
```

## Step 3 — Place data files

```
reflexive_kg/
└── data/
    ├── vor_sku_data.csv              # Master / Global SKU list (required)
    ├── SKU_Export.xlsx               # Production vendor export (example)
    ├── sample_vendor_SKU_Export.xlsx # Schema reference
    └── Global_sku.csv                # Legacy master (optional)
```

Point `GLOBAL_SKU_CSV` / `VENDOR_SKU_XLSX` in `config.py` if your paths differ.

## Step 4 — Run pipeline in order

```bash
python 01_schema.py
python 02_seed_data.py
python 03_reflection.py
python 03_reflection.py --label GlobalSKU --top 50
python 03_reflection.py --scores-only    # skip recompute, print scores only
```

## Step 5 — Lifecycle demo cohort + verification

```bash
python 05_synthesize_lifecycle.py --cohort 300
python 06_evaluate.py
python 06_scale_evaluate.py --sample 5000
python -m pytest test_agents.py test_full_criteria.py -v
```

## Next steps

| Goal | Command / doc |
|------|----------------|
| **New vendor Excel → status** | [README.md — Process a new vendor SKU Excel](README.md#1-process-a-new-vendor-sku-excel-primary-operations-path) |
| **LLM agents (Bedrock)** | [README.md — LLM agent pipeline](README.md#4-llm-agent-pipeline-bedrock-opus-47) |
| **Streamlit UI** | `streamlit run ui/app.py` |
| **API** | `uvicorn api.main:app --reload --port 8000` |
| **Synthetic QA vendor** | [README.md — Synthetic vendor QA](README.md#3-synthetic-vendor-qa-controlled-test-buckets) |

Handbook-oriented modules: `reflection_core.py`, `scoring.py`, `06_scale_evaluate.py`.
