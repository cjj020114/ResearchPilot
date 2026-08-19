from __future__ import annotations

from backend.app.ingestion.loaders.base import BaseLoader

__all__ = ["BaseLoader", "build_loader_registry"]


def build_loader_registry() -> dict[str, BaseLoader]:
    from backend.app.ingestion.loaders.registry import build_loader_registry as _build

    return _build()
