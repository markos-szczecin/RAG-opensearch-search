"""
Unit tests for FilterBuilder.

Verifies that the correct OpenSearch filter clauses are generated for
various role + metadata filter combinations.
"""

import pytest

from app.search.filters import FilterBuilder


@pytest.fixture
def builder(test_settings) -> FilterBuilder:
    return FilterBuilder(test_settings)


def test_status_filter_applied(builder: FilterBuilder) -> None:
    from app.models.search import SearchFilters
    filters = SearchFilters(status="approved")
    clause = builder.build(filters, user_role="customer")
    filter_list = clause.get("bool", {}).get("filter", [])
    assert {"term": {"status": "approved"}} in filter_list


def test_no_filters_returns_match_all(builder: FilterBuilder) -> None:
    from app.models.search import SearchFilters
    filters = SearchFilters(status=None)   # all None
    clause = builder.build(filters, user_role="customer")
    assert clause == {"match_all": {}}


def test_department_filter_applied(builder: FilterBuilder) -> None:
    from app.models.search import SearchFilters
    filters = SearchFilters(department="compliance")
    clause = builder.build(filters, user_role="customer")
    filter_list = clause.get("bool", {}).get("filter", [])
    assert {"term": {"department": "compliance"}} in filter_list


def test_language_filter_applied(builder: FilterBuilder) -> None:
    from app.models.search import SearchFilters
    filters = SearchFilters(language="en", status=None)
    clause = builder.build(filters, user_role="customer")
    filter_list = clause.get("bool", {}).get("filter", [])
    assert {"term": {"language": "en"}} in filter_list
