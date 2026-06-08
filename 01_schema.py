"""
01_schema.py — Neo4j schema: constraints, indexes, and vector indexes

Run this ONCE before seeding data.
Neo4j 5.x required for vector index support.

Usage:
    python 01_schema.py              # create schema (idempotent)
    python 01_schema.py --wipe       # delete all data + indexes, then recreate schema
"""

import argparse

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_DIM


# ── Constraint definitions ────────────────────────────────────────────────────
# Uniqueness constraints also create a backing B-tree index (faster lookups)
CONSTRAINTS = [
    ("GlobalSKU",    "sku_id",       "global_sku_id_unique"),
    ("TenantSKU",    "tenant_sku_id", "tenant_sku_id_unique"),
    ("Customer",     "customer_id",   "customer_id_unique"),
    ("Brand",        "brand_id",     "brand_id_unique"),
    ("PackageType",  "package_type_id", "package_type_id_unique"),
    ("Manufacturer", "name",         "manufacturer_name_unique"),
    ("Supplier",     "name",         "supplier_name_unique"),
    ("ProductClass", "name",         "product_class_name_unique"),
]

# ── Vector index definitions ──────────────────────────────────────────────────
# Two indexes per node type: self_emb (own attributes) + reflect_emb (neighbourhood view)
# similarity_function: cosine — anomaly score = 1 - cosine(self_emb, reflect_emb)
VECTOR_INDEXES = [
    # GlobalSKU
    ("idx_global_sku_self",    "GlobalSKU",    "self_emb"),
    ("idx_global_sku_reflect", "GlobalSKU",    "reflect_emb"),
    # TenantSKU (handbook — customer import layer)
    ("idx_tenant_sku_self",    "TenantSKU",    "self_emb"),
    ("idx_tenant_sku_reflect", "TenantSKU",    "reflect_emb"),
    # Brand
    ("idx_brand_self",         "Brand",        "self_emb"),
    ("idx_brand_reflect",      "Brand",        "reflect_emb"),
    # PackageType
    ("idx_package_self",       "PackageType",  "self_emb"),
    ("idx_package_reflect",    "PackageType",  "reflect_emb"),
    # Manufacturer
    ("idx_mfr_self",           "Manufacturer", "self_emb"),
    ("idx_mfr_reflect",        "Manufacturer", "reflect_emb"),
    # Supplier
    ("idx_supplier_self",      "Supplier",     "self_emb"),
    ("idx_supplier_reflect",   "Supplier",     "reflect_emb"),
    # ProductClass
    ("idx_class_self",         "ProductClass", "self_emb"),
    ("idx_class_reflect",      "ProductClass", "reflect_emb"),
]


_LEGACY_CONSTRAINTS = [
    "global_sku_id_unique", "tenant_sku_id_unique", "vendor_sku_id_unique",
    "customer_id_unique", "brand_id_unique", "package_type_id_unique",
    "manufacturer_name_unique", "supplier_name_unique", "product_class_name_unique",
    "scorelog_log_id_unique",
]
_LEGACY_INDEXES = [
    "idx_tenantsku_id", "idx_tenant_sku_id", "idx_customer_id",
    "idx_trainingimage_id", "idx_mergeevent_id", "idx_pallet_id",
    "idx_vendor_retail_upc", "idx_tenant_retail_upc",
    "idx_global_sku_upc", "idx_global_sku_brand_id",
    "idx_global_sku_self", "idx_global_sku_reflect",
    "idx_tenant_sku_self", "idx_tenant_sku_reflect",
    "idx_brand_self", "idx_brand_reflect", "idx_package_self", "idx_package_reflect",
    "idx_mfr_self", "idx_mfr_reflect", "idx_supplier_self", "idx_supplier_reflect",
    "idx_class_self", "idx_class_reflect",
    "idx_scorelog_entity_id", "idx_scorelog_run_id", "idx_scorelog_computed_at",
    "idx_match_candidate_vendor", "idx_match_candidate_status", "idx_global_sku_draft",
    "idx_ingestion_run", "idx_ingestion_run_at",
]


def _drop_schema_artifacts(session) -> None:
    """Drop known constraints/indexes without SHOW * (avoids OOM on large DBs)."""
    for name in _LEGACY_CONSTRAINTS:
        session.run(f"DROP CONSTRAINT {name} IF EXISTS")
        print(f"  ✓ Dropped constraint (if existed): {name}")
    for name in _LEGACY_INDEXES:
        session.run(f"DROP INDEX {name} IF EXISTS")
    print(f"  ✓ Dropped {len(_LEGACY_INDEXES)} known indexes (if existed)")


def wipe_database(session, batch_size: int = 10_000) -> None:
    """
    Full reset: drop schema artifacts, delete nodes in batches, recreate clean slate.

    Batched DELETE avoids Neo4j OOM on large graphs (~400k+ nodes).
    """
    print("\n── Wiping database ──────────────────────────")
    _drop_schema_artifacts(session)

    total = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    print(f"  Nodes to delete: {total:,}")

    deleted = 0
    while True:
        result = session.run(
            f"""
            MATCH (n)
            WITH n LIMIT {batch_size}
            DETACH DELETE n
            RETURN count(*) AS c
            """
        ).single()
        batch = result["c"]
        if batch == 0:
            break
        deleted += batch
        print(f"  … deleted {deleted:,} / {total:,}")

    remaining = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    print(f"  ✓ Wipe complete — {remaining:,} nodes remaining")


def create_constraints(session) -> None:
    for label, prop, name in CONSTRAINTS:
        cypher = (
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )
        session.run(cypher)
        print(f"  ✓ Constraint: {name}")


def create_vector_indexes(session) -> None:
    for idx_name, label, prop in VECTOR_INDEXES:
        cypher = f"""
            CREATE VECTOR INDEX {idx_name} IF NOT EXISTS
            FOR (n:{label}) ON n.{prop}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`:         {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """
        session.run(cypher)
        print(f"  ✓ Vector index: {idx_name} ({label}.{prop})")


def create_rel_indexes(session) -> None:
    """
    Composite index on MAPS_TO for fast UPC-based join lookups.
    """
    session.run(
        "CREATE INDEX idx_global_sku_upc IF NOT EXISTS "
        "FOR (n:GlobalSKU) ON (n.upc)"
    )
    session.run(
        "CREATE INDEX idx_tenant_retail_upc IF NOT EXISTS "
        "FOR (n:TenantSKU) ON (n.retail_upc)"
    )
    session.run(
        "CREATE INDEX idx_global_sku_brand_id IF NOT EXISTS "
        "FOR (n:GlobalSKU) ON (n.brand_id)"
    )
    print("  ✓ Property indexes: upc, retail_upc, brand_id")


def create_score_log_indexes(session) -> None:
    """Indexes for ScoreLog nodes — enable fast history and drift queries."""
    session.run(
        "CREATE CONSTRAINT scorelog_log_id_unique IF NOT EXISTS "
        "FOR (n:ScoreLog) REQUIRE n.log_id IS UNIQUE"
    )
    session.run(
        "CREATE INDEX idx_scorelog_entity_id IF NOT EXISTS "
        "FOR (n:ScoreLog) ON (n.entity_id)"
    )
    session.run(
        "CREATE INDEX idx_scorelog_run_id IF NOT EXISTS "
        "FOR (n:ScoreLog) ON (n.run_id)"
    )
    session.run(
        "CREATE INDEX idx_scorelog_computed_at IF NOT EXISTS "
        "FOR (n:ScoreLog) ON (n.computed_at)"
    )
    print("  ✓ ScoreLog indexes: entity_id, run_id, computed_at")


def main():
    parser = argparse.ArgumentParser(description="Neo4j schema setup")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete all graph data and indexes, then recreate schema from scratch",
    )
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print(f"\nConnecting to Neo4j at {NEO4J_URI} ...")

    with driver.session() as session:
        if args.wipe:
            wipe_database(session)

        print("\n── Constraints ──────────────────────────────")
        create_constraints(session)

        print("\n── Vector indexes ───────────────────────────")
        create_vector_indexes(session)

        print("\n── Property indexes ─────────────────────────")
        create_rel_indexes(session)

        print("\n── Score log indexes ────────────────────────")
        create_score_log_indexes(session)

    driver.close()
    print("\nSchema ready. Run 02_seed_data.py next.\n")


if __name__ == "__main__":
    main()
