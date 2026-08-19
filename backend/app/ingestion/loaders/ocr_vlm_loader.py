from __future__ import annotations

from pathlib import Path

from backend.app.core.models import Document, Element
from backend.app.ingestion.loaders.base import BaseLoader
from backend.app.ingestion.loaders.common import document_from_elements, infer_title, normalize_text
from backend.app.ingestion.types import RouteDecision
from backend.app.ingestion.vision.vlm_client import VlmBudget, get_vlm_client


class OcrVlmLoader(BaseLoader):
    """Standalone image -> one Document via a single VLM call (OCR + caption)."""

    def load(self, path: Path, route: RouteDecision, title: str | None = None) -> Document:
        budget = VlmBudget()
        client = get_vlm_client()
        enrichment = client.enrich_image(path, budget=budget, fallback_label=path.name)
        combined = enrichment.combined_text or normalize_text(
            f"[vlm_fallback] {path.name}"
        )
        element = Element.create(
            type="image",
            text=combined,
            ocr_text=enrichment.ocr_text or None,
            vlm_caption=enrichment.vlm_caption or None,
            image_path=str(path),
            metadata={
                "vlm_status": enrichment.status,
                "vlm_error": enrichment.error,
            },
        )
        status = "ready" if enrichment.status == "ready" else enrichment.status
        return document_from_elements(
            path=path,
            route=route,
            elements=[element],
            title=title or infer_title(combined, path.stem),
            extra={
                "image_path": str(path),
                "element_type": "image",
                "actual_parser": "vlm" if enrichment.status == "ready" else f"vlm_{enrichment.status}",
                "loader_status": status,
                "vlm_calls": budget.used,
            },
        )
