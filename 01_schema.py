"""
01_schema.py — Neo4j schema: constraints, indexes, and vector indexes

Run this ONCE before seeding data.
Neo4j 5.x required for vector index support.

Usage:
    python 01_schema.py
"""

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_DIM


# ── Constraint definitions ────────────────────────────────────────────────────
# Uniqueness constraints also create a backing B-tree index (faster lookups)
CONSTRAINTS = [
    ("GlobalSKU",    "sku_id",       "global_sku_id_unique"),
    ("VendorSKU",    "product_id",   "vendor_sku_id_unique"),
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
    # VendorSKU
    ("idx_vendor_sku_self",    "VendorSKU",    "self_emb"),
    ("idx_vendor_sku_reflect", "VendorSKU",    "reflect_emb"),
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
        "CREATE INDEX idx_vendor_retail_upc IF NOT EXISTS "
        "FOR (n:VendorSKU) ON (n.retail_upc)"
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
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print(f"\nConnecting to Neo4j at {NEO4J_URI} ...")

    with driver.session() as session:
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
