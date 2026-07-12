"""Public search-mode configuration and resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """User-facing vector-search strategies."""

    AUTO = "auto"
    EXACT_COSINE = "exact_cosine"
    HNSW_COSINE = "hnsw_cosine"


@dataclass(frozen=True)
class SearchModeResolution:
    """Requested and effective search modes, including any auto fallback."""

    requested: SearchMode
    resolved: SearchMode
    fallback_reason: Optional[str] = None


def resolve_search_mode(
    search_mode: SearchMode | str,
) -> tuple[SearchModeResolution, str]:
    """Resolve a public mode to the internal similarity backend name."""

    try:
        requested = SearchMode(search_mode)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(mode.value for mode in SearchMode)
        raise ValueError(f"search_mode must be one of: {choices}.") from exc

    if requested is SearchMode.EXACT_COSINE:
        return SearchModeResolution(requested, requested), "exact"
    if requested is SearchMode.HNSW_COSINE:
        return SearchModeResolution(requested, requested), "hnsw"

    try:
        from usearch.index import Index  # noqa: F401
    except ImportError:
        reason = (
            "The optional 'usearch' dependency is unavailable; auto mode "
            "selected exact cosine search."
        )
        logger.info(reason)
        return (
            SearchModeResolution(
                requested=requested,
                resolved=SearchMode.EXACT_COSINE,
                fallback_reason=reason,
            ),
            "exact",
        )

    return SearchModeResolution(requested, SearchMode.HNSW_COSINE), "hnsw"
