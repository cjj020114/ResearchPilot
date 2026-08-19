from __future__ import annotations

from pathlib import Path

from backend.app.core.models import Document
from backend.app.ingestion.loaders.common import normalize_text
from backend.app.ingestion.pipeline import IngestionPipeline


class DocumentParser:
    """Compatibility facade over the routed ingestion pipeline."""

    def __init__(self) -> None:
        self.pipeline = IngestionPipeline()

    def parse_file(self, path: Path, title: str | None = None) -> Document:
        document, _route = self.pipeline.ingest_file(path, title=title)
        return document

    def parse_text(self, text: str, title: str, source: str = "manual") -> Document:
        return Document.create(
            title=title,
            source=source,
            text=normalize_text(text),
            metadata={
                "file_type": "text",
                "route_layer1": "document",
                "route_layer2": "txt",
                "loader": "text",
                "modality": "text",
                "element_type": "text",
                "page": None,
                "section": None,
                "image_path": None,
                "table_id": None,
                "pdf_mode": None,
                "route_reason": "manual text ingest",
            },
            elements=None,
        )
