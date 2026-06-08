"""
agents/catalog_intent.py — Detect and parse master-catalog lookup questions.

Heuristic parser used before (or instead of) Bedrock Supervisor for questions like:
  "Is the product available in the master list: AQUA WATER 28OZ PL 1/15"
  "Does SKU exist — brand AQUA_WTR, package 20OZ PL 1/24"
"""

from __future__ import annotations

import re

from agents.dim_match import merge_query_dims, parse_dimensions_from_text

# Triggers suggesting a catalog / existence lookup (not graph investigation).
CATALOG_TRIGGERS: tuple[str, ...] = (
    "master list",
    "master catalog",
    "master data",
    "global sku",
    "global catalog",
    "in the catalog",
    "in master",
    "does this sku exist",
    "does sku exist",
    "sku exist",
    "product exist",
    "product available",
    "available in",
    "identify if",
    "identify the product",
    "identify the sku",
    "identify product",
    "identify sku",
    "can you identify",
    "find this sku",
    "find sku",
    "look up",
    "lookup",
    "match to master",
    "match against master",
    "do we have",
    "is there a",
    "is this product",
)

# Package descriptor: 28OZ PL 1/15, 16.9OZ PL 15/1, 1.5L PL 1/12, etc.
PACKAGE_PATTERN = re.compile(
    r"\b("
    r"\d+(?:\.\d+)?\s*(?:OZ|L|ML|GAL)\s+"
    r"(?:PL|CN|AL|BT|CAN|BTL|PK|PLPK)"
    r"(?:PK)?"
    r"[^\s,?.']*"
    r"(?:\s*\d+/\S+)?"
    r")",
    re.IGNORECASE,
)

_EXPLICIT_BRAND_RE = re.compile(
    r"brand\s*name\s*:?\s*['\"]?(.+?)['\"]?\s*(?:,|\||package|$)",
    re.IGNORECASE,
)
_EXPLICIT_PACKAGE_RE = re.compile(
    r"package\s*type\s*:?\s*['\"]?(.+?)['\"]?\s*$",
    re.IGNORECASE,
)

# Strip leading boilerplate from brand fragment.
_BRAND_PREFIX_RE = re.compile(
    r"^(?:"
    r"is the product(?: available)?(?: in the master list)?"
    r"|can you identify if this sku exists"
    r"|brand name"
    r"|product"
    r"|sku"
    r")\s*:?\s*",
    re.IGNORECASE,
)

# Few-shot examples (master_data / Neo4j GlobalSKU slugs + human labels).
FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "Is the product available in the master list: AQUA WATER 28OZ PL 1/15",
        "brand_name": "AQUA WATER",
        "package_type": "28OZ PL 1/15",
        "note": "maps to catalog slug AQUA_WTR, sku_id 6584",
    },
    {
        "question": "Can you identify if this SKU exists Brand name: AQUA_WTR, Package Type: 28OZ PL 1/15",
        "brand_name": "AQUA_WTR",
        "package_type": "28OZ PL 1/15",
    },
    {
        "question": "Does GATBLT WSTW MB TM 16.9OZ PL 15/1 exist in the master catalog?",
        "brand_name": "GATBLT WSTW MB TM",
        "package_type": "16.9OZ PL 15/1",
        "note": "catalog slug GATBLT_WSTW_MB_TM, sku_id 3714984",
    },
    {
        "question": "Look up brand BIG RED package 20OZ PL 1/24 in global SKU master",
        "brand_name": "BIG RED",
        "package_type": "20OZ PL 1/24",
    },
    {
        "question": "Identify SKU AQUA WATER 28OZ PL 1/15 weight 10.5 length 12 width 8 height 10",
        "brand_name": "AQUA WATER",
        "package_type": "28OZ PL 1/15",
        "query_dims": {"weight": 10.5, "length": 12.0, "width": 8.0, "height": 10.0},
        "note": "dimensions disambiguate when brand+package tie",
    },
    {
        "question": "Are there any duplicate sku exists in the master list/database",
        "task_type": "catalog_duplicate",
        "note": "master_data quality — NOT lifecycle brand-import scenario",
    },
    {
        "question": "Why are so many brands created as duplicates during customer import?",
        "task_type": "root_cause",
        "note": "NOT catalog_match — lifecycle scenario 1 brand-import cascade",
    },
    {
        "question": "Rank all GlobalSKUs by risk of causing training failures",
        "task_type": "risk_rank",
        "note": "NOT catalog_match — anomaly rank on cohort graph",
    },
]


# Master-catalog duplicate detection (Q2 demo — not lifecycle brand-import).
MASTER_CONTEXT_TRIGGERS: tuple[str, ...] = (
    "master list",
    "master catalog",
    "master data",
    "master database",
    "master db",
    "in master",
    "in the catalog",
    "global catalog",
    "database",
)

DUPLICATE_INDICATORS: tuple[str, ...] = (
    "duplicate",
    "duplicates",
    "duplicated",
    "dupes",
    "duplicate sku",
    "duplicate skus",
)

# Lifecycle phrasing — brand duplicates during import (scenarios 1 & 4), not master QA.
IMPORT_DUPLICATE_CONTEXT: tuple[str, ...] = (
    "during import",
    "customer import",
    "during customer import",
    "root cause",
    "trace the root",
    "brand mismatch",
    "brands were created",
    "brands created",
    "model accuracy",
    "accuracy degrade",
)


def is_catalog_duplicate_question(question: str) -> bool:
    """
    True when the user asks about duplicate SKUs in the master catalog/database,
    not about brand duplication during customer import (lifecycle scenarios 1/4).
    """
    ql = question.lower().strip()
    if not any(d in ql for d in DUPLICATE_INDICATORS):
        return False
    if any(x in ql for x in IMPORT_DUPLICATE_CONTEXT):
        return False
    if any(m in ql for m in MASTER_CONTEXT_TRIGGERS):
        return True
    if ("sku" in ql or "skus" in ql) and ("master" in ql or "database" in ql or "catalog" in ql):
        return True
    return False


def parse_catalog_duplicate(question: str) -> dict[str, str] | None:
    if is_catalog_duplicate_question(question):
        return {"task_type": "catalog_duplicate"}
    return None


def _normalize_package(raw: str) -> str:
    pkg = raw.strip().strip("'\"")
    pkg = re.sub(r"\s+", " ", pkg).upper()
    pkg = pkg.replace("/FIFTEEN", "/15").replace("/TWENTY", "/20").replace("/TWENTYFOUR", "/24")
    pkg = re.sub(r"/FIFTEEN\b", "/15", pkg, flags=re.I)
    pkg = re.sub(r"/TWENTY\b", "/20", pkg, flags=re.I)
    return pkg


def _clean_brand(raw: str) -> str:
    brand = _BRAND_PREFIX_RE.sub("", raw.strip())
    brand = re.sub(r"^(?:brand|package type)\s*:?\s*", "", brand, flags=re.I)
    brand = re.sub(r",?\s*package\s*type\s*:.*$", "", brand, flags=re.I)
    brand = brand.strip(" ,.'\"")
    brand = re.sub(r"\s+", " ", brand)
    return brand


def _parse_explicit_fields(question: str) -> dict[str, str] | None:
    """Parse 'Brand name: X, Package Type: Y' forms."""
    bm = _EXPLICIT_BRAND_RE.search(question)
    pm = _EXPLICIT_PACKAGE_RE.search(question)
    if not bm or not pm:
        return None
    brand = _clean_brand(bm.group(1))
    package_type = _normalize_package(pm.group(1))
    if brand and package_type:
        return {
            "task_type": "catalog_match",
            "brand_name": brand,
            "package_type": package_type,
        }
    return None


def is_catalog_question(question: str) -> bool:
    ql = question.lower()
    if any(t in ql for t in CATALOG_TRIGGERS):
        return True
    if _parse_explicit_fields(question):
        return True
    return bool(PACKAGE_PATTERN.search(question) and _clean_brand(question))


def _attach_query_dims(result: dict[str, str], question: str) -> dict:
    """Merge dimension hints parsed from the full question text."""
    dims = parse_dimensions_from_text(question)
    if dims:
        result = dict(result)
        result["query_dims"] = dims
    return result


def parse_catalog_match(question: str) -> dict[str, str] | None:
    """
    Extract brand_name and package_type from a natural-language catalog question.
    Returns None if a package descriptor cannot be found.
    """
    q = question.strip()
    if not q:
        return None

    explicit = _parse_explicit_fields(q)
    if explicit:
        return _attach_query_dims(explicit, q)

    pkg_m = PACKAGE_PATTERN.search(q)
    if not pkg_m:
        return None

    package_type = _normalize_package(pkg_m.group(1))
    brand: str | None = None

    if ":" in q:
        tail = q.split(":", 1)[1].strip()
        if pkg_m.group(0).upper() in tail.upper():
            brand = tail.upper().replace(pkg_m.group(0).upper(), "").strip()
        else:
            brand = tail

    if not brand:
        before = q[: pkg_m.start()].strip()
        for trigger in sorted(CATALOG_TRIGGERS, key=len, reverse=True):
            idx = before.lower().find(trigger)
            if idx >= 0:
                before = before[idx + len(trigger) :].strip()
        brand = before

    brand = _clean_brand(brand or "")
    if not brand:
        return None

    return _attach_query_dims(
        {
            "task_type": "catalog_match",
            "brand_name": brand,
            "package_type": package_type,
        },
        q,
    )


def few_shot_prompt_block() -> str:
    lines = ["Few-shot intent examples (PostgreSQL master_data + Neo4j GlobalSKU):"]
    for ex in FEW_SHOT_EXAMPLES:
        if ex.get("task_type") == "catalog_duplicate":
            lines.append(
                f'  Q: "{ex["question"]}" → {{"task_type":"catalog_duplicate"}}  '
                f'({ex.get("note", "")})'
            )
        elif ex.get("task_type") in ("root_cause", "risk_rank", "anomaly_explain"):
            lines.append(f'  Q: "{ex["question"]}" → {{"task_type":"{ex["task_type"]}"}}  ({ex.get("note", "")})')
        else:
            note = f"  // {ex['note']}" if ex.get("note") else ""
            lines.append(
                f'  Q: "{ex["question"]}" → '
                f'{{"task_type":"catalog_match","brand_name":"{ex["brand_name"]}",'
                f'"package_type":"{ex["package_type"]}"}}{note}'
            )
    return "\n".join(lines)
