"""Unit tests for question-driven graph search term extraction."""

from agents.graph_search import extract_search_terms, has_searchable_terms
from agents.models import QuerySpec


def test_extract_brand_and_package():
    spec = QuerySpec(
        question="Is AQUA WATER 28OZ PL 1/15 in the master list?",
        task_type="catalog_match",
        entity_types=["GlobalSKU"],
        brand_name="AQUA WATER",
        package_type="28OZ PL 1/15",
    )
    terms = extract_search_terms(spec.question, spec)
    assert terms.brand_name == "AQUA WATER"
    assert terms.package_type == "28OZ PL 1/15"
    assert has_searchable_terms(terms)


def test_extract_sku_id():
    terms = extract_search_terms(
        "Explain anomaly for GlobalSKU 6584 and trace import chain",
        QuerySpec(question="", task_type="root_cause", entity_types=["GlobalSKU"]),
    )
    assert "6584" in terms.sku_ids


def test_semantic_text_from_brand():
    terms = extract_search_terms(
        "Why is GATBLT WSTW MB TM failing training?",
        QuerySpec(question="", task_type="root_cause", entity_types=["GlobalSKU"]),
    )
    assert terms.brand_name == "GATBLT WSTW MB TM" or "GATBLT" in (terms.semantic_text or "")


def test_no_terms_on_empty_boilerplate_only():
    terms = extract_search_terms(
        "Why?",
        QuerySpec(question="Why?", task_type="root_cause", entity_types=["GlobalSKU"]),
    )
    assert not has_searchable_terms(terms) or len(terms.semantic_text) <= 3
