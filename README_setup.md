# Reflexive KG — Setup & Run Order

**Usage guide (vendor ingest, agents, API, scenarios):** see [README.md](README.md).

---

## Prerequisites

- Python 3.11+
- Neo4j 5.x running locally (Docker recommended)
- PostgreSQL (team AWS RDS) — connection settings in `.env`
- Amazon Bedrock access to **Claude Opus 4.7** (for agent pipeline)
- Bootstrap data files in `data/` (for first Postgres load only)

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

To wipe and reload the graph:

```bash
# In Neo4j Browser or cypher-shell:
MATCH (n) DETACH DELETE n;
```

Then re-run Steps 5–6 below.

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
```

Edit `.env`:

- **Neo4j** — `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- **PostgreSQL** — `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` (or `POSTGRES_SECRET_ID`)
- **Bedrock** — `AWS_REGION`, `BEDROCK_MODEL_ID`

Fetch RDS password (zsh: use single quotes):

```bash
aws secretsmanager get-secret-value \
  --secret-id 'rds!db-259bad4d-76af-44ff-8967-aa765bb03770' \
  --region us-east-1 \
  --query SecretString --output text
```

## Step 3 — Bootstrap data files (first load only)

```
reflexive_kg/
└── data/
    ├── vor_sku_data.csv              # Master / Global SKU list
    ├── SKU_Export.xlsx               # Example tenant export
    └── sample_vendor_SKU_Export.xlsx # Schema reference
```

These files are loaded **into PostgreSQL** by `12_load_postgres.py`. After that, `02_seed_data.py` reads from Postgres by default.

## Step 4 — Load PostgreSQL catalog

```bash
python 12_load_postgres.py
```

Creates / upserts:

- `master_data` ← `data/vor_sku_data.csv`
- `tenant_sku_data` + `tenant_data` ← `data/SKU_Export.xlsx`

Options:

```bash
python 12_load_postgres.py --wipe                    # truncate + reload
python 12_load_postgres.py --tenant-only path.xlsx   # upsert one tenant file
```

Verify:

```bash
python -c "from data.postgres_store import check_connection, table_counts, pg_session; print(check_connection()); 
with pg_session() as c: print(table_counts(c))"
```

## Step 5 — Build Neo4j from PostgreSQL

```bash
python 01_schema.py
python 02_seed_data.py              # reads master_data + tenant_sku_data from Postgres
python 03_reflection.py
python 03_reflection.py --label GlobalSKU --top 50
```

Offline fallback (no RDS):

```bash
python 02_seed_data.py --from-csv
```

## Step 6 — Lifecycle demo cohort + verification

```bash
python 05_synthesize_lifecycle.py --cohort 300
python 06_evaluate.py
python 06_scale_evaluate.py --sample 5000
python -m pytest test_agents.py test_full_criteria.py -v
```

## Next steps

| Goal | Command / doc |
|------|----------------|
| **New vendor Excel → Postgres + graph** | `12_load_postgres.py --tenant-only …` then `ingest_vendor.py` |
| **LLM agents (Bedrock)** | [README.md — LLM agent pipeline](README.md#4-llm-agent-pipeline-bedrock-opus-47) |
| **Streamlit UI** | `streamlit run ui/app.py` |
| **API** | `uvicorn api.main:app --reload --port 8000` |
| **DBeaver** | RDS host + `team8_db` + SSL require + password from Secrets Manager |

Handbook-oriented modules: `reflection_core.py`, `scoring.py`, `06_scale_evaluate.py`.
