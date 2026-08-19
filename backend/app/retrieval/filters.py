from __future__ import annotations

from typing import Any

from backend.app.core.config import settings
from backend.app.core.models import Chunk


def build_scope_filter(
    domain: str | None = None,
    knowledge_base_id: str | None = None,
    extra_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filters = dict(extra_filters or {})
    filters["domain"] = domain or settings.default_domain
    filters["knowledge_base_id"] = knowledge_base_id or settings.default_knowledge_base_id
    return filters


def matches_filters(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if expected is None:
            continue
        value = chunk.metadata.get(key)
        if isinstance(expected, list):
            if value not in expected:
                return False
            continue
        if value != expected:
            return False
    return True
