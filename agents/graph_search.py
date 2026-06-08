"""
agents/graph_search.py — Question-driven graph retrieval (exact / fuzzy / semantic).

Used by Planner → Doer for Neo4j investigation queries. Extracts entity hints from
the user's NL question, then runs three complementary search modes:

  exact    — sku_id, UPC, brand_name, brand_family, package_category_name
  fuzzy    — difflib brand/package matching (same logic as agent_matcher)
  semantic — embed question phrase → ANN on self_emb vector index
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from agents.catalog_intent import PACKAGE_PATTERN, is_catalog_question, parse_catalog_match
from agents.models import QuerySpec, QueryTask

SKU_ID_RE = re.compile(
    r"\b(?:global\s*sku|globalsku|sku)\s*[#:\-]?\s*(\d{3,10})\b",
    re.IGNORECASE,
)
BARE_SKU_RE = re.compile(r"\bsku\s+(\d{3,10})\b", re.IGNORECASE)
UPC_RE = re.compile(r"\b(\d{8,14})\b")

# Strip common NL boilerplate before semantic embedding.
_BOILERPLATE_RE = re.compile(
    r"\b("
    r"why|what|which|how|when|where|who|show|list|find|rank|explain|trace|"
    r"investigate|tell|give|all|the|most|at|risk|before|after|during|"
    r"customer|import|training|failures|failure|model|accuracy|degrade|"
    r"skus|sku|globalsku|global|entities|related|about"
    r")\b",
    re.IGNORECASE,
)

LABEL_INDEX = {
    "GlobalSKU": "idx_global_sku_self",
    "TenantSKU": "idx_tenant_sku_self",
    "Brand": "idx_brand_self",
    "PackageType": "idx_package_self",
}

LABEL_REFLECT_INDEX = {
    "GlobalSKU": "idx_global_sku_reflect",
    "TenantSKU": "idx_tenant_sku_reflect",
    "Brand": "idx_brand_reflect",
    "PackageType": "idx_package_reflect",
}


@dataclass
class GraphSearchTerms:
    """Structured hints extracted from a user question for graph search."""
    brand_name: str | None = None
    package_type: str | None = None
    sku_ids: list[str] = field(default_factory=list)
    upcs: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    semantic_text: str = ""
    raw_question: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_searchable_terms(terms: GraphSearchTerms) -> bool:
    return bool(
        terms.brand_name
        or terms.package_type
        or terms.sku_ids
        or terms.upcs
        or terms.keywords
        or (terms.semantic_text and len(terms.semantic_text.strip()) > 3)
    )


def extract_search_terms(question: str, spec: QuerySpec | None = None) -> GraphSearchTerms:
    """Pull SKU/UPC/brand/package/keyword hints from the NL question."""
    q = (question or "").strip()
    spec = spec or QuerySpec(question=q, task_type="root_cause", entity_types=["GlobalSKU"])

    brand = spec.brand_name
    package = spec.package_type

    if is_catalog_question(q):
        catalog = parse_catalog_match(q)
        if catalog:
            brand = brand or catalog.get("brand_name")
            package = package or catalog.get("package_type")

    if not package:
        pkg_m = PACKAGE_PATTERN.search(q)
        if pkg_m:
            package = pkg_m.group(1).strip().upper()
            if not brand:
                before = q[: pkg_m.start()].strip()
                # Last token run before package descriptor ≈ brand name
                before = re.sub(
                    r"^(?:what|why|how|explain|trace|find|show|graph|evidence|for|about|exists)\s+",
                    "",
                    before,
                    flags=re.I,
                )
                brand = before.strip(" ,.?") or None

    sku_ids = list(dict.fromkeys(SKU_ID_RE.findall(q) + BARE_SKU_RE.findall(q)))
    if spec.anchor_entity_id and spec.anchor_label in (None, "GlobalSKU"):
        sku_ids.insert(0, str(spec.anchor_entity_id))

    upcs = [u for u in UPC_RE.findall(q) if len(u) >= 10]

    keywords: list[str] = []
    for token in re.findall(r"[A-Z][A-Z0-9_]{2,}|[A-Za-z]{4,}", q):
        upper = token.upper()
        if upper in ("GLOBALSKU", "TENANTSKU", "BRAND", "PACKAGE", "CUSTOMER"):
            continue
        if brand and upper in brand.upper().replace(" ", ""):
            continue
        keywords.append(token)

    if brand and package:
        semantic_text = f"brand {brand} package {package}"
    elif brand:
        semantic_text = f"brand {brand}"
    elif package:
        semantic_text = f"package {package}"
    else:
        cleaned = _BOILERPLATE_RE.sub(" ", q)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        semantic_text = cleaned[:200] if cleaned else q[:200]

    return GraphSearchTerms(
        brand_name=brand,
        package_type=package,
        sku_ids=sku_ids,
        upcs=upcs,
        keywords=keywords[:8],
        semantic_text=semantic_text,
        raw_question=q,
    )


def _node_from_row(
    row: dict,
    label: str,
    source: str,
    match_score: float | None = None,
) -> dict:
    return {
        "entity_id": str(row.get("entity_id") or row.get("sku_id") or ""),
        "label": label,
        "display_name": (
            row.get("display_name")
            or row.get("brand_family")
            or row.get("brand_name")
            or row.get("entity_id")
            or "?"
        ),
        "self_emb": row.get("self_emb"),
        "reflect_emb": row.get("reflect_emb"),
        "timestamp": row.get("timestamp") or row.get("creation_date") or "",
        "source": source,
        "match_score": match_score,
        "properties": {
            k: v
            for k, v in row.items()
            if k not in ("self_emb", "reflect_emb") and v is not None
        },
    }


def search_exact(session, terms: GraphSearchTerms, label: str = "GlobalSKU") -> list[dict]:
    """Exact / case-insensitive match on primary keys and catalog fields."""
    if label != "GlobalSKU":
        return _search_exact_brand(session, terms, label)

    rows: list[dict] = []
    params: dict[str, Any] = {
        "brand": terms.brand_name,
        "package": terms.package_type,
        "sku_ids": terms.sku_ids or None,
        "upcs": terms.upcs or None,
    }

    cypher = """
    MATCH (g:GlobalSKU)
    WHERE
      ($sku_ids IS NOT NULL AND g.sku_id IN $sku_ids)
      OR ($upcs IS NOT NULL AND g.upc IN $upcs)
      OR ($brand IS NOT NULL AND (
            toUpper(g.brand_name) = toUpper($brand)
         OR toUpper(g.brand_family) = toUpper($brand)
         OR toUpper(replace(g.brand_name, '_', ' ')) = toUpper(replace($brand, '_', ' '))
      ))
      OR ($package IS NOT NULL AND toUpper(g.package_category_name) CONTAINS toUpper($package))
    RETURN g.sku_id AS entity_id,
           'GlobalSKU' AS label,
           coalesce(g.brand_family, g.brand_name, g.sku_id) AS display_name,
           g.brand_name AS brand_name,
           g.brand_family AS brand_family,
           g.package_category_name AS package_type,
           g.upc AS upc,
           g.self_emb AS self_emb,
           g.reflect_emb AS reflect_emb,
           coalesce(g.creation_date, '') AS timestamp
    LIMIT 50
    """
    for rec in session.run(cypher, **params):
        rows.append(_node_from_row(dict(rec), "GlobalSKU", "graph_exact", match_score=1.0))

    if not rows and terms.keywords:
        kw = terms.keywords[0]
        for rec in session.run(
            """
            MATCH (g:GlobalSKU)
            WHERE toUpper(g.brand_name) CONTAINS toUpper($kw)
               OR toUpper(g.brand_family) CONTAINS toUpper($kw)
               OR toUpper(g.package_category_name) CONTAINS toUpper($kw)
            RETURN g.sku_id AS entity_id,
                   coalesce(g.brand_family, g.brand_name, g.sku_id) AS display_name,
                   g.brand_name AS brand_name,
                   g.brand_family AS brand_family,
                   g.package_category_name AS package_type,
                   g.self_emb AS self_emb,
                   g.reflect_emb AS reflect_emb,
                   coalesce(g.creation_date, '') AS timestamp
            LIMIT 20
            """,
            kw=kw,
        ):
            rows.append(_node_from_row(dict(rec), "GlobalSKU", "graph_exact", match_score=0.95))

    return rows


def _search_exact_brand(session, terms: GraphSearchTerms, label: str) -> list[dict]:
    if not terms.brand_name:
        return []
    rows = []
    for rec in session.run(
        f"""
        MATCH (b:{label})
        WHERE toUpper(b.brand_family) = toUpper($brand)
           OR toUpper(b.brand_id) = toUpper($brand)
        RETURN b.brand_id AS entity_id,
               b.brand_family AS display_name,
               b.self_emb AS self_emb,
               b.reflect_emb AS reflect_emb,
               coalesce(b.creation_date, '') AS timestamp
        LIMIT 20
        """,
        brand=terms.brand_name,
    ):
        rows.append(_node_from_row(dict(rec), label, "graph_exact", match_score=1.0))
    return rows


def search_fuzzy(session, terms: GraphSearchTerms, label: str = "GlobalSKU") -> list[dict]:
    """Fuzzy brand/package matching via agent_matcher graph blocks."""
    if label != "GlobalSKU":
        return []

    from api.agent_matcher import _brand_block, _package_block

    rows: list[dict] = []
    seen: set[str] = set()

    if terms.brand_name:
        for sid in _brand_block(session, terms.brand_name):
            if sid in seen:
                continue
            seen.add(sid)
            rec = session.run(
                """
                MATCH (g:GlobalSKU {sku_id: $sid})
                RETURN g.sku_id AS entity_id,
                       coalesce(g.brand_family, g.brand_name, g.sku_id) AS display_name,
                       g.brand_name AS brand_name,
                       g.brand_family AS brand_family,
                       g.package_category_name AS package_type,
                       g.self_emb AS self_emb,
                       g.reflect_emb AS reflect_emb,
                       coalesce(g.creation_date, '') AS timestamp
                """,
                sid=sid,
            ).single()
            if rec:
                rows.append(_node_from_row(dict(rec), "GlobalSKU", "graph_fuzzy", match_score=0.85))

    if terms.package_type:
        pkg_hits = _package_block(session, terms.package_type)
        for sid, quality in pkg_hits.items():
            if sid in seen:
                continue
            seen.add(sid)
            rec = session.run(
                """
                MATCH (g:GlobalSKU {sku_id: $sid})
                RETURN g.sku_id AS entity_id,
                       coalesce(g.brand_family, g.brand_name, g.sku_id) AS display_name,
                       g.brand_name AS brand_name,
                       g.brand_family AS brand_family,
                       g.package_category_name AS package_type,
                       g.self_emb AS self_emb,
                       g.reflect_emb AS reflect_emb,
                       coalesce(g.creation_date, '') AS timestamp
                """,
                sid=sid,
            ).single()
            if rec:
                rows.append(
                    _node_from_row(dict(rec), "GlobalSKU", "graph_fuzzy", match_score=float(quality))
                )

    return rows[:50]


def search_semantic(
    session,
    terms: GraphSearchTerms,
    label: str = "GlobalSKU",
    top_k: int = 20,
) -> list[dict]:
    """Embed semantic_text and ANN-search the Neo4j vector index."""
    from api.agent_matcher import _embed_query, embeddings_available

    index_name = LABEL_INDEX.get(label)
    if not index_name or not embeddings_available():
        return []

    text = terms.semantic_text.strip()
    if terms.brand_name and terms.package_type:
        vec = _embed_query(terms.brand_name, terms.package_type)
    elif terms.brand_name:
        vec = _embed_query(terms.brand_name, terms.package_type or "")
    else:
        model = __import__("api.agent_matcher", fromlist=["_get_model"])._get_model()
        if model is None:
            return []
        vec = model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

    if vec is None:
        return []

    pk = {"GlobalSKU": "sku_id", "TenantSKU": "tenant_sku_id", "Brand": "brand_id"}.get(
        label, "sku_id"
    )
    display = {
        "GlobalSKU": "coalesce(n.brand_family, n.brand_name, n.sku_id)",
        "TenantSKU": "coalesce(n.brand, n.tenant_sku_id)",
        "Brand": "n.brand_family",
    }.get(label, "n.name")

    rows = []
    for rec in session.run(
        f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node AS n, score AS similarity
        RETURN n.{pk} AS entity_id,
               '{label}' AS label,
               {display} AS display_name,
               n.self_emb AS self_emb,
               n.reflect_emb AS reflect_emb,
               coalesce(n.creation_date, '') AS timestamp,
               similarity
        LIMIT $top_k
        """,
        index_name=index_name,
        top_k=top_k,
        query_vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
    ):
        d = dict(rec)
        rows.append(
            _node_from_row(d, label, "graph_semantic", match_score=float(d.get("similarity", 0)))
        )
    return rows


def terms_from_task(task: QueryTask) -> GraphSearchTerms:
    raw = dict(task.search_terms or {})
    if task.search_query:
        raw.setdefault("raw_question", task.search_query)
    return GraphSearchTerms(
        brand_name=raw.get("brand_name"),
        package_type=raw.get("package_type"),
        sku_ids=list(raw.get("sku_ids") or []),
        upcs=list(raw.get("upcs") or []),
        keywords=list(raw.get("keywords") or []),
        semantic_text=raw.get("semantic_text") or "",
        raw_question=raw.get("raw_question") or "",
    )


def run_graph_search_task(session, task: QueryTask) -> list[dict]:
    """Execute a graph_exact / graph_fuzzy / graph_semantic QueryTask."""
    terms = terms_from_task(task)
    mode = task.search_mode or task.task_type.replace("graph_", "")
    label = task.label or "GlobalSKU"

    if mode == "exact":
        return search_exact(session, terms, label)
    if mode == "fuzzy":
        return search_fuzzy(session, terms, label)
    if mode == "semantic":
        return search_semantic(session, terms, label, top_k=task.top_k)
    return []

