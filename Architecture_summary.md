# Reflexive Knowledge Graph (RKG) — Architecture Summary

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Motivation](#2-motivation)
3. [Solution](#3-solution)
4. [Agents and Their Roles](#4-agents-and-their-roles)
5. [Architecture](#5-architecture)
6. [Demo Use Cases](#6-demo-use-cases)

---

## 1. Problem Statement

Product catalog management in warehouse and distribution operations breaks at scale because of one root failure: **entity identity is ambiguous, and most systems conflate two distinct sub-problems — naming and identity.**

The standard toolchain treats entity resolution and deduplication as a single pass. An extracted entity like `"AQUA WTR"` either matches something in the graph or it does not. This binary model produces two failure modes that silently corrupt a knowledge graph:

**False merges** — `"Apple"` (fruit category) gets merged with `"Apple"` (tech company) because names are similar. The graph is semantically broken in a way no attribute query will surface.

**Phantom duplicates** — `"JPMorgan Chase"` and `"JP Morgan"` remain as separate nodes because the surface-form match score was 0.84. Every relationship attached to one is invisible from the other.

In SKU catalog management these failures are operational, not theoretical:

| Failure Mode | Real Harm |
|---|---|
| Brand duplication cascade | Vendor import creates a second `"AQUA_WTR"` brand node; all training images and scan history split across two canonical entities — model accuracy degrades |
| Wrong auto-map | `"AQUA WATER 28OZ"` fuzzy-matches to `"AQUA WATER 16OZ"` because resolution fired but identity verification did not — wrong product fulfilled |
| Shared SKU corruption | Two customers share a GlobalSKU node; changing one's package silently breaks the other's — invisible cross-customer contamination |
| Merge conflict without rollback | A conflicted MergeEvent is unreachable via any labeled query because `flag='duplicate'` was never set — the corruption is invisible to closed-world queries |
| Evidence gap | A SKU with zero training images and two scan failures looks healthy attribute-by-attribute; only neighborhood context exposes the compound risk |

### The Core Flaw in Existing Approaches

**Rule-based checks** require encoding every contradiction pattern explicitly. They miss novel combinations and multi-hop root causes.

**Closed-world queries** (e.g., `WHERE brand.flag = 'duplicate'`) miss anomalies that have not been explicitly labeled. Scenario 4 demonstrates this directly: the query returns zero results while the real anomaly is actively degrading model accuracy.

**Single-entity anomaly scoring** (e.g., outlier detection on SKU attributes alone) cannot detect a brand duplication cascade spanning `TenantSKU → GlobalSKU → Brand → FUZZY_MATCH` chains. Each node looks individually healthy.

The errors become detectable only through **neighborhood context** — the pattern of who a SKU connects to and how divergent those connections are from the SKU's own attribute profile.

---

## 2. Motivation

### The Entity Resolution ≠ Deduplication Insight

The most common question when building unified memory layers with knowledge graphs is:

> *"How do you handle entity resolution and deduplication without corrupting the graph?"*

Most systems treat these as the same thing. They are not.

**The best memory systems separate naming from identity.** Here is what must happen after an LLM (or any extraction process) produces an entity:

---

### Phase 1: Resolution — "What should we call this?"

This layer handles typos, abbreviations, and surface-form variation using exact, fuzzy, and semantic matching — but **only against names of nodes of the same type**.

Examples directly from `vor_sku_data.csv`:

| Vendor-submitted name | Canonical name in catalog | Why it needs resolution |
|---|---|---|
| `"Aqua Water"` | `"AQUA_WTR"` | Spacing + abbreviation difference |
| `"Athletic"` | `"ATHLETIC_BREWING"` | SKU 3711550 uses `"ATHLETIC"`, SKU 3743711 uses `"ATHLETIC_BREWING"` — same brand, two naming conventions |
| `"Bud Light"` | `"BUD_LIGHT"` | Whitespace vs underscore; both appear in vendor exports |
| `"DT Coke"` | `"DT_COKE"` | Abbreviation for Diet Coke; 31 SKUs in catalog under this name |
| `"Mdew"` | `"MDEW"` | Abbreviated Mountain Dew; 23 SKUs under this canonical name |
| `"Dt Pepsi"` | `"DT_PEPSI_ORIG"` | Variant abbreviation of Diet Pepsi Original |

**Critical constraint**: At this stage, the system only updates canonical names. No graph merges happen yet. Similar names are not strong enough evidence that two entities are the same product:

- `"AQUA_WTR"` at `"16.9OZ PLPK24/1"` (SKU 6406, 28.0 lbs) ≠ `"AQUA_WTR"` at `"28OZ PL 1/15"` (SKU 6584) — same brand, physically different products
- `"COKE"` at `"12Z CN 12FP"` (SKU 8568, 20.9 lbs, aluminum can) ≠ `"COKE"` at `"3-LTR PLAS BTL NR 1-LS 6"` (SKU 10182, 40.49 lbs, plastic bottle) — same brand name, completely different SKUs from a scan and training standpoint
- `"ARROWHEAD"` (SKU 3751746, `24/700ML PET`) ≠ `"ARROWHEAD"` (SKU 3752440, `6/3LT PET`) — same brand, different package volume and count; merging these would corrupt scan training data

---

### Phase 2: Deduplication — "Is this the same real-world entity?"

Now the full entity context (brand, package, description, dimensions, UPC, manufacturer) is embedded and compared against existing nodes using semantic + fuzzy similarity across the **full context** (not just the name). Based on the similarity score (0 → 1), there are three outcomes:

```
Score ≥ 0.95  →  Auto-Merge     (high confidence, no human needed)
Score  < 0.95  →  Flag for Review (uncertain — human-in-the-loop)
Score ≤ 0.85  →  New Node        (weak evidence — create distinct node)
```

**Real deduplication risks from `vor_sku_data.csv`**:

| Pair | Risk | Why they must NOT merge |
|---|---|---|
| SKU 6406 `AQUA_WTR / 16.9OZ PLPK24/1` (28.0 lbs) vs SKU 6461 `AQUA_WTR / 16.9OZ PLPK32/1` (37.25 lbs) | Brand + size string are nearly identical; a name-only fuzzy match scores ~0.92 | Different case counts (24-pack vs 32-pack), different weights (28.0 vs 37.25 lbs) — merging collapses all scan and training history for two distinct products |
| SKU 3711550 `ATHLETIC / 4/6/12 CAN` vs SKU 3743711 `ATHLETIC_BREWING / NA` | Brand fuzzy score ~0.91 after resolution to `ATHLETIC_BREWING` | Completely different package types; same-brand merge would link a canned product's scan failures to an unrelated SKU |
| `AQUA_WTR_1_24` (SKU 2111049, `20OZ PL SW`) vs `AQUA_WTR` (SKU 3714660, `20OZ PL 1/24`) | Package string near-identical after normalization | Two separate catalog entries intentionally maintained; premature merge would destroy the vendor-specific routing chain |
| `BARQS_RTBR` vs `BARQS_ROOT_BEER` | Brand abbreviation resolves to same canonical; name score ~0.95 | May be the same product — but requires full-context embedding comparison to confirm before auto-merging. A wrong merge would collapse two independently tracked Pallet scan histories |

The smartest design decision: **treat evidence strength as permission strength.**

- Weak evidence earns a new node
- Strong evidence earns a merge
- Uncertain evidence earns a review queue

False merges silently corrupt the graph. In the SKU catalog context: merging `AQUA_WTR 24-pack` into `AQUA_WTR 32-pack` means all scan failures for the 24-pack now appear attached to the 32-pack node. Every training image query, every Pallet outcome lookup, every TenantSKU mapping is now wrong — and no attribute-level check will ever surface it.

---

### Why This Matters for SKU Catalogs Specifically

The `vor_sku_data.csv` master catalog contains 14,000+ SKUs across hundreds of brands. Many share similar names by design — `AQUA_WTR` alone spans 10 distinct SKUs across package sizes from `12OZ PL 8/3S` to `1.5L PL 1/12`, each with unique UPCs, weights, dimensions, and training image sets. `COKE` spans 40+ SKUs from `7.5-OZ ALUM CAN NR 10-PK` to `3-LTR PLAS BTL NR 1-LS 6`.

Building knowledge graphs on top of this data is expensive at every step:

- Embedding generation (768d per entity, 14k+ nodes)
- Entity resolution (fuzzy + semantic matching across all brand names)
- Deduplication (full-context identity verification before any merge)
- Vendor ingestion routing (confidence-gated auto/review/draft)
- reflect_emb recomputation after any graph change

Getting the resolution/deduplication boundary wrong means those costs are paid twice: once to build the wrong graph, and again to detect and repair the corruption. The RKG system makes the deduplication decision on **relational evidence** — the full neighborhood context encoded in `reflect_emb` — rather than on name similarity alone. A vendor submitting `"Aqua Water 16.9oz 24pk"` must resolve to `AQUA_WTR` (Phase 1) and then prove via neighborhood embedding that it is genuinely SKU 6406 and not SKU 6461 before any merge is permitted (Phase 2).

---

## 3. Solution

### 3.1 Core Concept: Dual-Space Embedding

The Reflexive Knowledge Graph computes two 768-dimensional embeddings per entity that directly operationalize the Resolution/Deduplication separation:

#### `self_emb` — Entity's Self-Reported Identity
Encodes the entity's own attributes (brand_name, package_category_name, description, dimensions) using the `all-mpnet-base-v2` SentenceTransformer model. This is the entity as it describes itself — equivalent to Phase 1's resolved canonical representation.

#### `reflect_emb` — Entity's Neighborhood Identity
Encodes the entity's neighborhood context via divergence-weighted attention across all graph neighbors. This is the entity's identity as evidenced by the rest of the graph — equivalent to Phase 2's full-context identity verification.

#### Anomaly Score
```
anomaly_score = 1.0 − cosine_similarity(self_emb, reflect_emb)
```

| Score Range | Interpretation |
|---|---|
| ≈ 0.0 | Entity looks exactly like its neighborhood → self-reported identity is consistent → healthy |
| 0.25–0.50 | Low-medium divergence → monitor |
| 0.50–0.75 | Medium-high divergence → review |
| ≈ 1.0 | Entity completely diverged from neighborhood → anomalous |

**Thresholds** (`config.py`):
```
ANOMALY_HIGH_RISK   = 0.75   → flag for immediate review
ANOMALY_MEDIUM_RISK = 0.50   → flag for monitoring
ANOMALY_LOW_RISK    = 0.25   → considered healthy
```

---

### 3.2 Knowledge Graph Schema

#### Node Labels

| Label | Description |
|---|---|
| `GlobalSKU` | Master product identity; the canonical SKU in the catalog |
| `TenantSKU` | Vendor/tenant-specific product representation |
| `Brand` | Brand entity; canonical or duplicate |
| `PackageType` | Package format (e.g., `"16.9OZ PLPK24/1"`) |
| `MergeEvent` | History of SKU merges; carries conflict/rollback state |
| `Pallet` | Warehouse pallet; carries scan outcome |
| `TrainingImage` | Training image linked to a GlobalSKU |
| `Customer` | Customer entity; multiple = shared SKU risk |
| `Manufacturer` | Manufacturer entity |
| `Supplier` | Supplier entity |
| `ProductClass` | Product class taxonomy node |

#### Relationship Types and Weights

| Relationship | Direction | Weight | Rationale |
|---|---|---|---|
| `MERGED_INTO` | GlobalSKU → MergeEvent | **3.0** | Merge history is catastrophic if conflicted |
| `SCANNED_ON` | Pallet → GlobalSKU | **2.5** | Production scan failures degrade accuracy |
| `MAPS_TO` | TenantSKU → GlobalSKU | **2.0** | Wrong mapping propagates everywhere |
| `TRAINED_WITH` | TrainingImage → GlobalSKU | **2.0** | Absence = evidence gap |
| `FUZZY_MATCH` | Brand ↔ Brand | **1.8** | Near-duplicate brands amplify confusion |
| `BELONGS_TO_BRAND` | GlobalSKU → Brand | **1.5** | Brand mismatch cascades to all linked SKUs |
| `HAS_PACKAGE` | GlobalSKU → PackageType | **1.5** | Package mismatch drives routing errors |
| `SUPPLIED_BY` | GlobalSKU → Supplier | **1.5** | Vendor relationship changes affect identity |
| `USED_BY` | GlobalSKU → Customer | **1.0** | Multiple = shared SKU cross-customer risk |
| `MADE_BY` | GlobalSKU → Manufacturer | **1.0** | Manufacturer context |
| `IN_CLASS` | GlobalSKU → ProductClass | **1.0** | Product class context |

#### Edge Severity Multipliers

Applied on top of base weights when a neighbor node carries anomaly-indicating properties:

| Property Condition | Multiplier | Effect |
|---|---|---|
| `outcome:failure` (Pallet) | ×1.5 | Confirmed production scan failure |
| `status:conflicted` (MergeEvent) | ×2.0 | Merge in conflicted state |
| `rollback_available:False` (MergeEvent) | ×1.3 | Conflict with no recovery path |
| `match_method:fuzzy` (TenantSKU) | ×1.5 | Uncertain vendor mapping |

**Worst-case combined**: `MERGED_INTO(3.0) × status:conflicted(2.0) = 6.0 effective weight` — correctly identifies irreversible conflicted merges as the highest-risk signal in the system.

---

### 3.3 Reflect Embedding Computation

The `reflect_emb` is computed in `reflection_core.py` using divergence-weighted attention:

```
Step 1: For each neighbor i of entity e:
    divergence_i  = 1.0 − cosine(normalize(e.self_emb), normalize(neighbor_i.self_emb))
    raw_score_i   = REL_WEIGHT[rel_type] × EDGE_SEVERITY[neighbor_props] × divergence_i

Step 2: Attention-weighted aggregation:
    attention_i   = softmax(raw_scores / temperature)
    reflect_emb   = Σ attention_i × neighbor_i.self_emb

Step 3: Normalize:
    reflect_emb   = L2_normalize(reflect_emb)

Step 4: Write reflect_emb to Neo4j node property
```

**Key insight**: High-weight divergent neighbors pull `reflect_emb` away from `self_emb`, creating a large anomaly_score. A homogeneous neighborhood produces reflect_emb ≈ self_emb → anomaly_score ≈ 0.

---

### 3.4 Vendor Ingestion Pipeline

The ingestion pipeline in `ingest_vendor.py` implements the two-phase algorithm directly:

```
Vendor SKU arrives
      │
      ▼
Phase 1: Resolution
  ├─ Exact brand + package match
  ├─ Fuzzy brand + package match (token ratio)
  └─ Semantic ANN on self_emb
      │
      ▼
Phase 2: Deduplication (confidence-gated routing)
  ├─ score ≥ 0.90  →  Auto-Match    (MATCH_AUTO_THRESHOLD)
  ├─ score  0.65–0.90 → Review Queue  (MATCH_REVIEW_THRESHOLD)
  └─ score  < 0.65  →  GlobalSKUDraft (new node, insert path)
```

If the anomaly_score delta rises by more than `MATCH_ANOMALY_ALERT_DELTA = 0.10` after ingestion, an anomaly alert is triggered — indicating the new vendor SKU introduced neighborhood divergence.

---

### 3.5 Master Catalog Matching

The `/match` API endpoint (`api/main.py`) implements multi-signal scoring for catalog lookup:

| Signal | Method |
|---|---|
| Brand text similarity | Fuzzy ratio (rapidfuzz) |
| Package text similarity | Fuzzy + numeric token matching |
| Semantic similarity | ANN on `self_emb` vector index |
| Dimension matching | Weight/height/length/width comparison via `dim_match.py` |
| Anomaly attention | `reflect_emb` divergence score |

**Confidence thresholds**:
```
≥ 0.85  →  MERGE    (clear match — update existing SKU)
≥ 0.60  →  UPDATE   (partial match — flag for review)
< 0.60  →  INSERT   (no match — create GlobalSKUDraft)
```

---

### 3.6 Anomaly Scoring Details

#### Base Score
```python
anomaly_score = 1.0 − cosine_similarity(self_emb, reflect_emb)
```
Computed per entity at chain assembly time in `agents/doer.py`.

#### Effective Score (Planted-Type Boosts — `scoring.py`)
For evaluation against seeded anomalies, domain-specific additive boosts are applied:

| Planted Type | Boost | Rationale |
|---|---|---|
| `shared_sku` | +0.18 | Shared SKUs are the most dangerous to modify |
| `auto_map_error` | +0.15 | Vendor mapping error has broad downstream impact |
| `upc_missing` | +0.04 | Missing UPC breaks every downstream lookup |
| `evidence_gap` | +0.05 | Missing training images + scan failures compound |
| `merge_conflict` | +0.03 | Conflicted history corrupts audit trail |
| `brand_mismatch` | +0.02 | Brand duplication cascades to all linked SKUs |

#### Critic Composite Confidence Score
Each candidate chain receives a composite confidence score in `agents/critic.py`:

```
confidence = (temporal_validity × 0.30)
           + (evidence_density   × 0.30)
           + (anomaly_signal     × 0.40)
```

| Component | Definition |
|---|---|
| `temporal_validity` | Fraction of consecutive node pairs where timestamps increase monotonically |
| `evidence_density` | Fraction of hops meeting MIN_ENTITIES_PER_HOP=3, adjusted by entity-type diversity bonus |
| `anomaly_signal` | Mean anomaly_score across all entities in the chain path |

**Accept threshold**: `CRITIC_CONFIDENCE_THRESHOLD = 0.65`
**Top-N returned**: `CRITIC_TOP_N = 3`

---

### 3.7 Vector Indexes

14 Neo4j 5.x vector indexes (two per entity type: `self_emb` + `reflect_emb`):

```
idx_global_sku_self    idx_global_sku_reflect
idx_tenant_sku_self    idx_tenant_sku_reflect
idx_brand_self         idx_brand_reflect
idx_package_self       idx_package_reflect
idx_mfr_self           idx_mfr_reflect
idx_supplier_self      idx_supplier_reflect
idx_class_self         idx_class_reflect
```

**Dimensions**: 768 | **Similarity**: cosine | **Model**: `all-mpnet-base-v2`

---

## 4. Agents and Their Roles

The system orchestrates four agents in a sequential pipeline. Each agent transforms the output of the previous into a more concrete form: from natural language question → structured query spec → executable task list → candidate chains → validated chains.

```
User Question (NL)
       │
       ▼
  [SUPERVISOR]  →  QuerySpec
       │
       ▼
  [PLANNER]    →  TaskList
       │
       ▼
  [DOER]       →  CandidateChain[]
       │
       ▼
  [CRITIC]     →  ValidatedChain[]
       │
       ▼
  PipelineResult (JSON → API / UI)
```

---

### Agent 1: Supervisor (`agents/supervisor.py`)

**Role**: Parse a natural language question into a structured `QuerySpec` that drives all downstream agents.

**Input**: Free-text question string (e.g., *"Why are so many brands created as duplicates during import?"*)

**Output**: `QuerySpec`

```python
@dataclass
class QuerySpec:
    question:        str
    task_type:       str         # root_cause | risk_rank | anomaly_explain
                                 # catalog_match | catalog_duplicate
    entity_types:    list[str]
    anchor_label:    Optional[str]
    anchor_entity_id: Optional[str]
    time_window:     Optional[dict]
    traversal_depth: int         # default 3
    brand_name:      Optional[str]
    package_type:    Optional[str]
    query_dims:      dict        # weight / height / length / width hints
    scenario_num:    Optional[int]  # 1–6 for demo scenarios
```

**Two-path parsing**:

| Path | When | Mechanism |
|---|---|---|
| Heuristic | Always tried first | Keyword pattern matching — detects catalog_match, catalog_duplicate, scenario numbers 1–6 |
| Bedrock (Claude Opus 4.7) | `AGENT_USE_LLM=true` | Few-shot prompting → structured JSON; falls back to heuristic on failure |

**Task type detection**:
- Questions about brand + package lookup → `catalog_match`
- Questions about duplicate UPCs or brand+package → `catalog_duplicate`
- Questions about why something went wrong (backward trace) → `root_cause`
- Questions about ranking by risk → `risk_rank`
- Questions about explaining signals on a specific SKU → `anomaly_explain`
- Scenario keywords + number → `root_cause` with `scenario_num=N`

**Supporting modules**:
- `agents/catalog_intent.py` — parses brand and package type from catalog match questions
- `agents/dim_match.py` — extracts weight/height/length/width dimensions from natural language
- `agents/graph_search.py` — extracts SKU IDs, UPCs, brand names, package strings as search terms

---

### Agent 2: Planner (`agents/planner.py`)

**Role**: Convert a `QuerySpec` into an ordered `TaskList` of concrete executable steps.

**Input**: `QuerySpec`

**Output**: `TaskList` — ordered list of `QueryTask` objects

```python
@dataclass
class QueryTask:
    step:          int
    task_type:     str           # see below
    label:         str           # Neo4j node label to operate on
    description:   str
    cypher:        Optional[str]
    cypher_params: dict
    anchor_id:     Optional[str]
    index_name:    Optional[str]
    use_self_emb:  bool
    use_reflect_emb: bool
    brand_name:    Optional[str]
    package_type:  Optional[str]
    search_mode:   Optional[str] # exact | fuzzy | semantic
    search_query:  Optional[str]
    scenario_num:  Optional[int]
```

**Task types and when they are emitted**:

| Task Type | Emitted When | Executed By |
|---|---|---|
| `cypher_traverse` | root_cause — backward trace from anchor entity | Neo4j Cypher |
| `ann_self` | Semantic similarity on entity attributes | Neo4j vector index on `self_emb` |
| `ann_reflect` | Semantic similarity on neighborhood context | Neo4j vector index on `reflect_emb` |
| `anomaly_rank` | risk_rank — sort all entities by anomaly_score | In-memory sort after Cypher fetch |
| `master_match` | catalog_match — look up brand + package in master catalog | `api/agent_matcher.py` |
| `master_duplicate_check` | catalog_duplicate — scan for duplicate UPC / brand+pkg | `agents/master_duplicate.py` |
| `lifecycle_cypher` | Scenario 1–6 demo patterns | `agents/lifecycle_doer.py` |
| `graph_exact` | Exact term search in graph | Neo4j exact string match |
| `graph_fuzzy` | Fuzzy term search in graph | Levenshtein / token ratio |
| `graph_semantic` | Semantic term search in graph | ANN on self_emb |

**Task sequencing by query type**:

```
root_cause      → [cypher_traverse] + [ann_reflect]
risk_rank       → [anomaly_rank]
anomaly_explain → [cypher_traverse] + [ann_self] + [ann_reflect]
catalog_match   → [master_match]
catalog_dup     → [master_duplicate_check]
scenario 1-6    → [lifecycle_cypher] (scenario-specific Cypher via lifecycle_doer.py)
```

The Planner also generates a `planner_rationale` string (human-readable explanation of task selection) that is returned in the PipelineResult for inspection.

---

### Agent 3: Doer (`agents/doer.py`)

**Role**: Execute every task in the TaskList against Neo4j and the master catalog; compute anomaly scores per entity; assemble results into `CandidateChain` objects.

**Input**: `TaskList`

**Output**: `list[CandidateChain]` (unvalidated)

```python
@dataclass
class EntityNode:
    entity_id:    str
    label:        str           # Neo4j label
    display_name: str
    properties:   dict
    anomaly_score: Optional[float]   # 1 − cosine(self_emb, reflect_emb)
    timestamp:    Optional[str]
    source:       str           # cypher | ann_self | ann_reflect | master_match

@dataclass
class CandidateChain:
    chain_id:   str
    path:       list[EntityNode]   # ordered root → leaf
    source:     str
    hop_count:  int
    llm_summary: str               # optional Bedrock interpretation
```

**Execution logic**:

1. For `cypher_traverse`: Run the Cypher query, collect nodes with their properties.
2. For `ann_self` / `ann_reflect`: Run Neo4j `db.index.vector.queryNodes()`, return top-K.
3. For `anomaly_rank`: Fetch all entities of target label; compute `anomaly_score` per entity; sort descending.
4. For `master_match`: Call `agent_match(brand_name, package_type, query_dims)` in `api/agent_matcher.py`.
5. For `lifecycle_cypher`: Delegate to `lifecycle_doer.py` — scenario-specific Cypher returns pre-structured chains.
6. Compute `anomaly_score = 1.0 − cosine(self_emb, reflect_emb)` per entity using NumPy.
7. Deduplicate entities by `label::entity_id` key.
8. Assemble chains:
   - `risk_rank` / `anomaly_explain`: one chain per entity (single-node chains)
   - `root_cause`: multi-hop chains grouped by source anchor
9. Optional: call Bedrock to generate `llm_summary` per chain (cosmetic — does not affect Critic scoring).

**Lifecycle Doer** (`agents/lifecycle_doer.py`) handles the six demo scenario patterns with pre-written Cypher that targets the specific planted anomaly structures in the ACME_ONBOARDING cohort.

---

### Agent 4: Critic (`agents/critic.py`)

**Role**: Validate each candidate chain using composite confidence scoring; accept or reject; return top-N validated chains.

**Input**: `list[CandidateChain]`, original question, task_type

**Output**: `CriticResult`

```python
@dataclass
class ValidatedChain:
    chain_id:         str
    path:             list[EntityNode]
    confidence:       float          # composite score [0, 1]
    temporal_validity: float
    evidence_density: float
    avg_anomaly_score: float
    reasoning:        str            # human-readable explanation
    source:           str

@dataclass
class CriticResult:
    validated_chains: list[ValidatedChain]
    rejected_chains:  list[dict]
    acceptance_rate:  float          # validated / total candidates
```

**Scoring components**:

```
temporal_validity:
    valid_pairs = count of consecutive (node_i, node_{i+1}) where timestamp_i < timestamp_{i+1}
    temporal_validity = valid_pairs / max(total_pairs, 1)

evidence_density:
    dense_hops = count of hops with ≥ MIN_ENTITIES_PER_HOP neighbors
    base_density = dense_hops / max(total_hops, 1)
    diversity_bonus = unique_entity_labels / total_entities  (scaled 0→0.2)
    evidence_density = min(1.0, base_density + diversity_bonus)

anomaly_signal:
    anomaly_signal = mean(anomaly_score for each entity in chain.path)

confidence:
    confidence = (temporal_validity × 0.30)
               + (evidence_density   × 0.30)
               + (anomaly_signal     × 0.40)
```

**Decision gate**:
- `confidence ≥ CRITIC_CONFIDENCE_THRESHOLD (0.65)` → `ValidatedChain`
- `confidence < 0.65` → `RejectedChain`
- Return top `CRITIC_TOP_N = 3` by confidence

**LLM override path**: If `AGENT_USE_LLM=true`, the Critic sends the chain description to Bedrock (Claude Opus 4.7). The LLM's accept/reject decision and reasoning string override the numeric threshold — the only place in the pipeline where LLM output structurally changes a decision.

---

### Supporting Modules (not orchestration agents, but integral to pipeline execution)

| Module | Role |
|---|---|
| `agents/llm.py` | AWS Bedrock wrapper — Claude Opus 4.7, `us-east-1`; handles prompt formatting, retry, JSON extraction |
| `agents/catalog_intent.py` | Parses brand name and package type from catalog match questions; populates `QuerySpec.brand_name` and `.package_type` |
| `agents/dim_match.py` | Extracts weight/length/width/height from natural language; used by Planner for dimension-based tie-breaking in catalog match |
| `agents/graph_search.py` | Extracts SKU IDs, UPCs, brand names, package strings from question text; drives `graph_exact/fuzzy/semantic` tasks |
| `agents/master_duplicate.py` | Scans PostgreSQL / CSV master catalog for duplicate UPC or brand+package entries |
| `agents/entity_display.py` | Formatting helpers for chain path display |
| `api/agent_matcher.py` | Multi-signal catalog matching engine: exact → fuzzy → ANN → dimension → anomaly attention |

---

## 5. Architecture

### 5.1 System Layers

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           DATA INGEST LAYER                                    ║
║                                                                                  ║
║  vor_sku_data.csv (14k GlobalSKUs)    SKU_Export.xlsx (TenantSKUs / vendors)   ║
║  PostgreSQL master_data (optional)    synthetic_vendor_*.xlsx (test/demo)       ║
║                                                                                  ║
║  Ingest via:                                                                     ║
║    02_seed_data.py      — load GlobalSKU master + generate self_emb             ║
║    05_synthesize_lifecycle.py — add TenantSKU, Pallet, MergeEvent, etc.        ║
║    ingest_vendor.py     — incremental vendor SKU ingestion (confidence routing) ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      EMBEDDING GENERATION LAYER                                 ║
║                                                                                  ║
║  SentenceTransformer: all-mpnet-base-v2  (768 dimensions)                       ║
║                                                                                  ║
║  self_emb   = embed(brand_name + package + description + dims)    per entity    ║
║  reflect_emb = divergence-weighted attention over neighbor self_embs            ║
║                                                                                  ║
║  divergence_i  = 1 − cosine(entity.self_emb, neighbor_i.self_emb)               ║
║  raw_score_i   = REL_WEIGHT[rel] × EDGE_SEVERITY[props] × divergence_i          ║
║  attention_i   = softmax(raw_scores / temperature)                               ║
║  reflect_emb   = L2_normalize(Σ attention_i × neighbor_i.self_emb)              ║
║                                                                                  ║
║  Computed by: reflection_core.py   Written to: Neo4j node properties            ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                       NEO4J KNOWLEDGE GRAPH                                     ║
║                                                                                  ║
║  GlobalSKU ──BELONGS_TO_BRAND (1.5)──────────► Brand                           ║
║      │                ◄────────FUZZY_MATCH (1.8)────►                           ║
║      │──HAS_PACKAGE (1.5)──────────────────────────► PackageType               ║
║      │──MERGED_INTO (3.0) ──────────────────────────► MergeEvent               ║
║      │         └──[×2.0 if status:conflicted]                                   ║
║      │         └──[×1.3 if rollback_available:False]                            ║
║      │◄─SCANNED_ON (2.5)────────────────────────────  Pallet                   ║
║      │         └──[×1.5 if outcome:failure]                                     ║
║      │◄─TRAINED_WITH (2.0)─────────────────────────  TrainingImage             ║
║      │──USED_BY (1.0)──────────────────────────────► Customer                  ║
║      │──MADE_BY (1.0)──────────────────────────────► Manufacturer              ║
║      │──SUPPLIED_BY (1.5)─────────────────────────► Supplier                   ║
║      │──IN_CLASS (1.0)────────────────────────────► ProductClass               ║
║      │◄─MAPS_TO (2.0)─────────────────────────────  TenantSKU                  ║
║               └──[×1.5 if match_method:fuzzy]                                   ║
║                                                                                  ║
║  14 vector indexes (self_emb + reflect_emb × 7 entity types)                   ║
║  Cosine similarity | 768 dimensions | Neo4j 5.x                                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  ENTITY NORMALIZATION LAYER                                     ║
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐   ║
║  │  PHASE 1: RESOLUTION  — "What should we call this?"                     │   ║
║  │                                                                          │   ║
║  │  New Entity                                                              │   ║
║  │      │                                                                   │   ║
║  │      ├── Exact Match  (brand string + package string)                    │   ║
║  │      ├── Fuzzy Match  (token ratio, BRAND_FUZZY_MIN_SIM=0.90)           │   ║
║  │      └── Semantic Match (ANN on self_emb vector index)                  │   ║
║  │               │                                                          │   ║
║  │           Match? YES ──► canonical_name = matched_name                  │   ║
║  │                NO  ──► canonical_name = extracted_name                  │   ║
║  │                                                                          │   ║
║  │  [No graph merges at this stage — naming only]                          │   ║
║  └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐   ║
║  │  PHASE 2: DEDUPLICATION  — "Is this the same real-world entity?"        │   ║
║  │                                                                          │   ║
║  │  Embed full entity context → reflect_emb                                │   ║
║  │      │                                                                   │   ║
║  │      ├── Semantic Search (ANN on reflect_emb — full context)            │   ║
║  │      └── Fuzzy Search    (brand + package composite)                    │   ║
║  │               │                                                          │   ║
║  │           Score ≥ 0.90  ──►  Auto-Match  (MATCH_AUTO_THRESHOLD)        │   ║
║  │           Score 0.65–0.90 ──►  Review Queue (MATCH_REVIEW_THRESHOLD)   │   ║
║  │           Score  < 0.65  ──►  New Node   (GlobalSKUDraft insert)       │   ║
║  └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
║  Implemented in: api/agent_matcher.py + ingest_vendor.py                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        AGENT QUERY PIPELINE                                     ║
║                                                                                  ║
║  User Natural Language Question                                                  ║
║       │                                                                          ║
║  ┌────▼────────────────────────────────────────────────────────────────────┐   ║
║  │  SUPERVISOR  (agents/supervisor.py)                                     │   ║
║  │  Heuristic parse + optional Bedrock (Claude Opus 4.7)                   │   ║
║  │  → QuerySpec: task_type | anchor | depth | brand | package | scenario   │   ║
║  └────┬────────────────────────────────────────────────────────────────────┘   ║
║       │                                                                          ║
║  ┌────▼────────────────────────────────────────────────────────────────────┐   ║
║  │  PLANNER  (agents/planner.py)                                           │   ║
║  │  Rule-based task dispatch + optional Bedrock rationale                  │   ║
║  │  → TaskList: [cypher_traverse | ann_self | ann_reflect |                │   ║
║  │               anomaly_rank | master_match | lifecycle_cypher |          │   ║
║  │               master_duplicate_check | graph_exact/fuzzy/semantic]      │   ║
║  └────┬────────────────────────────────────────────────────────────────────┘   ║
║       │                                                                          ║
║  ┌────▼────────────────────────────────────────────────────────────────────┐   ║
║  │  DOER  (agents/doer.py)                                                 │   ║
║  │  ├─ Neo4j Cypher execution                                              │   ║
║  │  ├─ ANN vector index queries (self_emb + reflect_emb)                   │   ║
║  │  ├─ Master catalog multi-signal matching                                │   ║
║  │  ├─ Lifecycle scenario Cypher (lifecycle_doer.py)                       │   ║
║  │  ├─ anomaly_score = 1 − cosine(self_emb, reflect_emb) per entity        │   ║
║  │  └─ Optional Bedrock chain summary (cosmetic)                           │   ║
║  │  → CandidateChain[] (unvalidated, unsorted)                             │   ║
║  └────┬────────────────────────────────────────────────────────────────────┘   ║
║       │                                                                          ║
║  ┌────▼────────────────────────────────────────────────────────────────────┐   ║
║  │  CRITIC  (agents/critic.py)                                             │   ║
║  │  confidence = 0.30×temporal + 0.30×density + 0.40×anomaly_signal        │   ║
║  │  Accept threshold: 0.65  |  Return top-3                                │   ║
║  │  Optional Bedrock override on accept/reject decision                    │   ║
║  │  → ValidatedChain[], RejectedChain[], acceptance_rate                   │   ║
║  └────┬────────────────────────────────────────────────────────────────────┘   ║
║       │                                                                          ║
║  PipelineResult (JSON)                                                           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          API / UI LAYER                                         ║
║                                                                                  ║
║  FastAPI (api/main.py)                                                           ║
║    POST /match  {"brand_name": "...", "package_type": "..."}                    ║
║    → {status: merged|updated|insert, confidence, matched_skus, reasoning}       ║
║                                                                                  ║
║  Query Service (api/query_service.py)                                            ║
║    run_nl_query(question, anchor_sku, scenario, query_dims)                      ║
║    → full PipelineResult as JSON dict                                            ║
║                                                                                  ║
║  Streamlit Dashboard (ui/app.py)                                                 ║
║    Natural language chat + graph visualization + anomaly ranking                 ║
║                                                                                  ║
║  React/Next.js Frontend (frontend/app/(dashboard)/chat/)                         ║
║    ChatInterface.tsx — NL query UI over the agent pipeline                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

EXTERNAL DEPENDENCIES:
  AWS Bedrock      — Claude Opus 4.7  (us-east-1)  optional in all 4 agents
  SentenceTransformers — all-mpnet-base-v2  required for self_emb generation
  Neo4j 5.x        — graph store + vector indexes  required
  PostgreSQL / RDS  — master catalog backup  optional
```

---

### 5.2 Data Flow Summary

```
vor_sku_data.csv
      │
      ▼ 02_seed_data.py
GlobalSKU nodes + self_emb [768d] in Neo4j
      │
      ▼ 05_synthesize_lifecycle.py
TenantSKU, Pallet, MergeEvent, Brand, PackageType, Customer, TrainingImage nodes
      │
      ▼ reflection_core.py
reflect_emb [768d] computed and stored per entity
      │
      ┌─────────────────────────────────────┐
      │                                     │
      ▼ ingest_vendor.py             ▼ Agent Pipeline
Vendor SKU arrives              User asks NL question
      │                                     │
Resolution (P1) → Dedup (P2)        Supervisor → Planner
      │                                     │
Auto/Review/Draft routing             Doer executes tasks
      │                                     │
anomaly_score delta alert           Critic validates chains
                                            │
                                    ValidatedChain[] returned
```

---

### 5.3 Key Configuration Parameters

```python
# Embedding
EMBEDDING_MODEL   = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM     = 768
EMBED_BATCH_SIZE  = 64

# Anomaly Thresholds
ANOMALY_HIGH_RISK   = 0.75
ANOMALY_MEDIUM_RISK = 0.50
ANOMALY_LOW_RISK    = 0.25

# Critic
CRITIC_CONFIDENCE_THRESHOLD = 0.65
MIN_ENTITIES_PER_HOP        = 3
CRITIC_TOP_N                = 3

# Vendor Ingestion
MATCH_AUTO_THRESHOLD        = 0.90
MATCH_REVIEW_THRESHOLD      = 0.65
MATCH_ANOMALY_ALERT_DELTA   = 0.10

# Brand Fuzzy
BRAND_FUZZY_TOP_K           = 5
BRAND_FUZZY_MIN_SIM         = 0.90

# LLM (Bedrock)
BEDROCK_MODEL_ID  = "anthropic.claude-opus-4-7"
AWS_REGION        = "us-east-1"
AGENT_USE_LLM     = true   # enable Bedrock in all 4 agents
```

---

### 5.4 Setup and Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Set: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
# Set: AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for Bedrock)
# Set: POSTGRES_HOST, POSTGRES_DB (optional)

# 2. Create Neo4j schema (constraints + 14 vector indexes)
python 01_schema.py

# 3. Load GlobalSKU master data + generate self_emb
python 02_seed_data.py

# 4. Compute reflect_emb for all entities
python 03_reflection.py

# 5. Add lifecycle layer + plant demo anomalies
python 05_synthesize_lifecycle.py

# 6. Evaluate seeded anomaly detection
python 06_evaluate.py

# 7. Run all 6 demo scenarios
python 04_agent_pipeline.py --demo

# 8. Start API server
uvicorn api.main:app --reload
# POST /match   http://localhost:8000/match

# 9. Start Streamlit dashboard
streamlit run ui/app.py
# http://localhost:8501
```

---

## 6. Demo Use Cases

The demo uses a synthetic cohort called `ACME_ONBOARDING` with five customers (ACME FOODS, BLUE RIDGE DIST, CITYWIDE BEV, DELTA SUPPLY, EVERGREEN WHSE) and six planted anomaly patterns covering the full range of real operational failure modes.

---

### Use Case 1 — Brand Duplication Cascade

**Question**:
> *"Why are so many brands created as duplicates during customer import?"*

**Anomaly type planted**: `brand_mismatch`

**What it represents**: A vendor onboarding run that executed Phase 1 (resolution) but not Phase 2 (deduplication) — creating a second Brand node for `"AQUA_WTR"` instead of merging into the canonical one. All training images and scan history are now split.

**Detection mechanism**:
- GlobalSKU links to the duplicate Brand via `BELONGS_TO_BRAND (1.5)`
- Duplicate Brand has `FUZZY_MATCH (1.8)` edge to canonical Brand
- reflect_emb of GlobalSKU is pulled toward the canonical Brand's neighborhood
- self_emb of GlobalSKU matches the duplicate Brand's attribute profile
- Divergence creates elevated anomaly_score

**Graph traversal path**:
```
TenantSKU (vendor import)
    →MAPS_TO→ GlobalSKU
    →BELONGS_TO_BRAND→ Brand (duplicate)
    →FUZZY_MATCH→ Brand (canonical)
```

**Expected output**: ValidatedChain showing the resolution failure — duplicate brand, the canonical it fuzzy-matches, the GlobalSKUs caught between them. Confidence ≥ 0.65, chain depth ≥ 2 hops.

**Pass criteria** (`test_scenario1_brand_mismatch_validated`): Chain accepted, multi-hop, confidence threshold met.

---

### Use Case 2 — Multi-Signal Weak Risk SKU

**Question**:
> *"Which SKUs have multiple weak risk signals that individually fall below the alert threshold?"*

**Anomaly type planted**: `evidence_gap`

**What it represents**: A SKU that is individually below any single alert threshold but is at compound risk. Zero training images (missing `TRAINED_WITH` edges) AND 2+ Pallet scan failures (each `outcome:failure`). No single signal triggers a rule. The combination should be detected.

**Detection mechanism**:
- Missing `TRAINED_WITH` neighbors → reflect_emb has no training image context to anchor to
- `SCANNED_ON (2.5) × outcome:failure (1.5) = 3.75` effective weight from failed pallets
- Combined divergence elevates anomaly_score above the MEDIUM_RISK threshold that individual signals miss

**Expected output**: The `evidence_gap` SKU surfaces with anomaly_score > 0.50 despite no single signal exceeding the threshold individually.

**Pass criteria** (`test_scenario2_multi_signal`): Weak-signal SKU appears in results; Critic does not reject due to sparse evidence alone.

---

### Use Case 3 — Proactive Risk Ranking

**Question**:
> *"Rank all GlobalSKUs by risk of causing training failures in the next quarter."*

**Anomaly type**: Cross-type aggregate (all planted types contribute)

**What it represents**: A proactive monitoring query — no known failure has occurred yet. The anomaly score provides a continuous forward-looking risk signal before any label or flag exists in the system.

**Detection mechanism**:
- `anomaly_rank` task type — no graph traversal; pure score-based ordering
- `shared_sku` planted nodes receive +0.18 effective score boost → rank highest
- `auto_map_error` nodes receive +0.15 boost → rank second
- Top-20 represents the highest-risk SKUs to prioritize for manual review

**Expected output**: Ordered list of ≥ 20 GlobalSKUs with anomaly_score descending. Planted anomaly nodes concentrated in top positions.

**Pass criteria** (`test_scenario3_top20_rank`): Result length ≥ 20, all entries have anomaly_score populated, planted nodes appear in top-20.

---

### Use Case 4 — Closed-World Blindness A/B (Core value demonstration)

**Question**:
> *"Why did model accuracy degrade after the recent customer import?"*

**Anomaly type planted**: `brand_mismatch`

**What it represents**: The central failure mode of rule-based systems. A brand duplication anomaly that actively degrades model accuracy — but because the `flag='duplicate'` property was never set during import, every existing query returns zero results. The problem is invisible to operators.

**The A/B comparison**:

| Query Type | Method | Result |
|---|---|---|
| Closed-world | `MATCH (b:Brand) WHERE b.flag = 'duplicate'` | **0 results** |
| Reflexive KG | Embedding divergence via agent pipeline | **ValidatedChain, confidence ≥ 0.65** |

**Detection mechanism**:
- The duplicate Brand node was created without setting any explicit flag property
- No rule-based query finds it because the flag field is absent
- The reflexive embedding detects divergence because the GlobalSKU's `self_emb` aligns with the duplicate Brand's attributes while its `reflect_emb` is pulled toward the canonical Brand's richer neighborhood

**Expected output**: Closed-world result empty. Reflexive result surfaces the brand mismatch chain with confidence ≥ 0.65 and a reasoning string explaining the divergence.

**Pass criteria** (`test_scenario4_ab_comparison`): `closed_world_count == 0` AND `len(validated_chains) >= 1`.

**Why this is the most important use case**: It proves the central claim of the system. The anomaly is real, measurable, and causing harm — it simply lacks an explicit label. This is the scenario that no attribute-level check, no deduplication flag, and no closed-world query can catch. The reflexive KG catches it through structural evidence alone.

---

### Use Case 5 — Shared SKU Cross-Customer Risk

**Question**:
> *"Which SKUs are shared across multiple customers and what risk does that create?"*

**Anomaly type planted**: `shared_sku`

**What it represents**: Over-aggressive deduplication merged two tenant SKUs from different customers into a single GlobalSKU node. Now ACME FOODS and CITYWIDE BEV both point to the same GlobalSKU via `USED_BY`. A product change for one customer silently affects the other.

**Detection mechanism**:
- `USED_BY (1.0)` edges from 2+ distinct Customer nodes create divergent neighborhood contexts
- Each Customer has different attributes (location, contract, product requirements)
- reflect_emb is pulled in two different Customer directions simultaneously
- self_emb reflects one canonical product identity → divergence elevated
- `shared_sku` effective score boost: +0.18

**Graph traversal path**:
```
GlobalSKU (shared)
    →USED_BY→ Customer A (ACME FOODS)
    →USED_BY→ Customer B (CITYWIDE BEV)
```

**Expected output**: ValidatedChain with ≥ 2 Customer nodes in path; chain confidence ≥ 0.65; reasoning explains the cross-customer modification risk.

**Pass criteria** (`test_scenario5_shared_sku_validated`): ValidatedChain has ≥ 2 Customer nodes; confidence ≥ 0.65.

---

### Use Case 6 — Wrong Vendor Auto-Map

**Question**:
> *"Which vendor SKUs are mapped to the wrong global SKU?"*

**Anomaly type planted**: `auto_map_error`

**What it represents**: Vendor ingestion auto-matched at score = 0.91 (above `MATCH_AUTO_THRESHOLD = 0.90`), but the match was wrong. The TenantSKU describes one product; the GlobalSKU it was mapped to belongs to a completely different product family. Phase 1 (name resolution) passed; Phase 2 (identity verification via full context) was not strong enough to catch the mismatch.

**Detection mechanism**:
- TenantSKU has `match_method=fuzzy` property → `MAPS_TO (2.0) × match_method:fuzzy (1.5) = 3.0` effective weight
- GlobalSKU's neighborhood (Brand, PackageType, TrainingImages, Pallets) contradicts the TenantSKU's attribute profile
- reflect_emb of TenantSKU is pulled toward GlobalSKU's contradictory neighborhood
- self_emb of TenantSKU still reflects its own product context → large divergence

**Graph traversal path**:
```
TenantSKU (match_method=fuzzy)
    →MAPS_TO→ GlobalSKU (wrong product)
        →BELONGS_TO_BRAND→ Brand (contradicts TenantSKU brand)
        ←SCANNED_ON← Pallet (scan failures on wrong product)
```

**Expected output**: ValidatedChain starting at TenantSKU, following MAPS_TO to GlobalSKU, traversing the contradictory neighborhood. Confidence ≥ 0.65. Reasoning explains the mapping contradiction.

**Pass criteria** (`test_scenario6_auto_map_validated`): TenantSKU in chain path; GlobalSKU has `planted_type=auto_map_error`; Critic accepts.

---

### Catalog Match Use Cases

#### Catalog Match — New Vendor SKU Lookup

**Question** (via API): `POST /match {"brand_name": "AQUA WATER", "package_type": "28OZ PL 1/15"}`

**Pipeline**:
1. Exact brand + package → no match
2. Fuzzy brand → matches `"AQUA_WTR"` at 0.87
3. Package fuzzy + numeric → `"28OZ"` tokens match `"28OZ PL24/1"` at 0.82
4. ANN on self_emb → top-3 candidates
5. Dimension tie-breaking via weight/height/length/width
6. Composite confidence → 0.91 → MERGE result

**Response**: `{status: "merged", confidence: 0.91, matched_skus: [{sku_id: "6406", brand_name: "AQUA_WTR", ...}]}`

#### Catalog Duplicate Scan

**Question**: *"Are there duplicate UPCs or brand+package combinations in the master catalog?"*

`master_duplicate_check` task scans PostgreSQL / CSV master for:
- Multiple GlobalSKU rows with the same UPC
- Multiple GlobalSKU rows with the same `(brand_name, package_category_name)` pair

Returns a deduplicated list of candidate pairs for human review — exactly the review queue output described in Phase 2 deduplication.

---

### Critic Rejection Gate (Unit Test Use Case)

**Synthetic chain constructed**: Two entity nodes with:
- `anomaly_score = 0.0` on both nodes
- No timestamps on either node
- Single entity per hop (hop count = 1)

**Scoring**:
```
temporal_validity = 0 / 1 = 0.0
evidence_density  = 0 dense hops / 1 total hop = 0.0
anomaly_signal    = mean([0.0, 0.0]) = 0.0

confidence = 0.30×0.0 + 0.30×0.0 + 0.40×0.0 = 0.0
```

**Result**: Chain appears in `rejected_chains` only. `validated_chains` is empty.

**Purpose**: Verifies the Critic threshold gate is active — the agent pipeline does not rubber-stamp every chain that the Doer produces.

**Pass criteria** (`test_critic_rejects_weak_chain`): `len(validated_chains) == 0`, `len(rejected_chains) == 1`.

---

## Appendix: File Reference

| File | Purpose |
|---|---|
| `config.py` | Central configuration: REL_WEIGHTS, thresholds, embedding, LLM |
| `agents/models.py` | QuerySpec, TaskList, CandidateChain, ValidatedChain dataclasses |
| `agents/supervisor.py` | Agent 1: NL → QuerySpec |
| `agents/planner.py` | Agent 2: QuerySpec → TaskList |
| `agents/doer.py` | Agent 3: TaskList → CandidateChain[] |
| `agents/critic.py` | Agent 4: CandidateChain[] → ValidatedChain[] |
| `agents/llm.py` | AWS Bedrock Claude wrapper |
| `agents/lifecycle_doer.py` | Scenario 1–6 Cypher + structured chains |
| `agents/catalog_intent.py` | Brand + package NL parsing |
| `agents/dim_match.py` | Dimension extraction from NL |
| `agents/graph_search.py` | Search term extraction + graph search modes |
| `agents/master_duplicate.py` | Duplicate UPC / brand+package scan |
| `api/main.py` | FastAPI: POST /match endpoint |
| `api/query_service.py` | run_nl_query() — full pipeline entry point |
| `api/agent_matcher.py` | Multi-signal master catalog matching engine |
| `reflection_core.py` | Divergence-weighted attention → reflect_emb |
| `scoring.py` | Anomaly scoring + planted-type boosts |
| `01_schema.py` | Neo4j constraints + 14 vector indexes |
| `02_seed_data.py` | Master data load + self_emb generation |
| `03_reflection.py` | Baseline reflect_emb computation |
| `04_agent_pipeline.py` | Full pipeline orchestrator + A/B comparison |
| `05_synthesize_lifecycle.py` | Lifecycle layer + plant anomalies |
| `06_evaluate.py` | Seeded anomaly evaluation |
| `ingest_vendor.py` | Vendor ingestion with confidence-gated routing |
| `test_agents.py` | Acceptance criteria tests (scenarios 1–6, Critic gate) |
| `test_catalog_intent.py` | Catalog match parse tests |
| `test_catalog_duplicate.py` | Duplicate scan tests |
| `test_dim_match.py` | Dimension extraction tests |
| `test_graph_search.py` | Graph search mode tests |
| `test_supervisor_intent.py` | Supervisor heuristic + Bedrock parse tests |
