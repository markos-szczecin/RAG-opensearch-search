import datetime

from app.config import Settings
from app.models.search import SearchFilters


class FilterBuilder:
    """
    Converts SearchFilters + user_role into an OpenSearch bool filter clause.

    Keeping this logic in one class means adding a new metadata field (e.g.
    'region') only requires changes here, not in every search service.

    TODO (Phase 1 — implement before any real search queries):
      - date_range: valid_from ≤ today AND (valid_to IS NULL OR valid_to ≥ today)
      - role_access: map user_role → allowed access_levels via Settings.role_access_levels
      - Combine all active filters with "must" (AND semantics in OpenSearch filter context)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self, filters: SearchFilters, user_role: str) -> dict:
        """
        Returns the OpenSearch bool.filter clause as a Python dict.

        Example output:
        {
          "bool": {
            "filter": [
              {"terms": {"access_level": ["public", "internal"]}},
              {"term": {"status": "approved"}},
              {"range": {"valid_from": {"lte": "now/d"}}},
              ...
            ]
          }
        }
        """
        must_filters: list[dict] = []

        # --- Access level ---
        # TODO: look up role → access_levels from settings.role_access_levels
        # allowed_levels = self._settings.role_access_levels.get(user_role, ["public"])
        # must_filters.append({"terms": {"access_level": allowed_levels}})

        # --- Explicit filter overrides ---
        if filters.status:
            must_filters.append({"term": {"status": filters.status}})
        if filters.language:
            must_filters.append({"term": {"language": filters.language}})
        if filters.doc_type:
            must_filters.append({"term": {"doc_type": filters.doc_type}})
        if filters.department:
            must_filters.append({"term": {"department": filters.department}})
        if filters.access_level:
            must_filters.append({"term": {"access_level": filters.access_level}})

        # --- Document freshness ---
        # TODO: add valid_from ≤ today
        # TODO: add valid_to ≥ today OR valid_to does not exist

        return {"bool": {"filter": must_filters}} if must_filters else {"match_all": {}}
