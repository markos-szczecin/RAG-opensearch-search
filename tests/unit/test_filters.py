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


def test_no_user_filters_still_applies_security_filters(builder: FilterBuilder) -> None:
    """Security filters (access_level + date validity) are always present.

    A bare match_all would be a security vulnerability — every search must
    be scoped to the caller's access level regardless of user-provided filters.
    """
    from app.models.search import SearchFilters
    filters = SearchFilters(status=None)   # no user-defined filters
    clause = builder.build(filters, user_role="customer")
    filter_list = clause.get("bool", {}).get("filter", [])
    assert any("access_level" in str(f) for f in filter_list), "access_level filter must always be present"
    assert any("valid_from" in str(f) for f in filter_list), "date validity filter must always be present"


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
