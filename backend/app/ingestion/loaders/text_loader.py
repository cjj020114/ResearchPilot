from __future__ import annotations

from pathlib import Path

from backend.app.core.models import Document, Element
from backend.app.ingestion.loaders.base import BaseLoader
from backend.app.ingestion.loaders.common import document_from_elements, normalize_text
from backend.app.ingestion.types import RouteDecision


class TextLoader(BaseLoader):
    def load(self, path: Path, route: RouteDecision, title: str | None = None) -> Document:
        text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
        element = Element.create(type="text", text=text)
        return document_from_elements(
            path=path,
            route=route,
            elements=[element],
            title=title,
            extra={"actual_parser": "text", "loader_status": "ready", "element_type": "text"},
        )
