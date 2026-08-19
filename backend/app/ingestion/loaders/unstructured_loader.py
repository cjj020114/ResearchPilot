from __future__ import annotations

from pathlib import Path

from backend.app.core.models import Document, Element, ElementType
from backend.app.ingestion.loaders.base import BaseLoader
from backend.app.ingestion.loaders.common import document_from_elements, normalize_text
from backend.app.ingestion.types import RouteDecision
from backend.app.ingestion.vision.vlm_client import VlmBudget, get_vlm_client


class UnstructuredLoader(BaseLoader):
    """docx / pptx / html -> Element[] via unstructured, images enriched via VLM."""

    def load(self, path: Path, route: RouteDecision, title: str | None = None) -> Document:
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise RuntimeError("unstructured is required for office/html parsing") from exc

        budget = VlmBudget()
        client = get_vlm_client()
        partitioned = partition(filename=str(path))
        elements: list[Element] = []
        statuses: list[str] = []
        current_section: str | None = None

        for item in partitioned:
            category = (getattr(item, "category", None) or "NarrativeText").lower()
            text = normalize_text(str(getattr(item, "text", "") or ""))
            metadata = getattr(item, "metadata", None)
            page = getattr(metadata, "page_number", None) if metadata else None
            image_path = getattr(metadata, "image_path", None) if metadata else None

            element_type = _map_category(category)
            if element_type == "heading" and text:
                current_section = text

            if element_type == "image":
                enrichment_context = f"section={current_section or ''}; category={category}"
                if image_path and Path(image_path).exists():
                    enrichment = client.enrich_image(
                        Path(image_path),
                        context=enrichment_context,
                        budget=budget,
                        fallback_label=Path(image_path).name,
                    )
                    statuses.append(enrichment.status)
                    elements.append(
                        Element.create(
                            type="image",
                            text=enrichment.combined_text or text or Path(image_path).name,
                            ocr_text=enrichment.ocr_text or None,
                            vlm_caption=enrichment.vlm_caption or None,
                            page=page,
                            section=current_section,
                            image_path=str(image_path),
                            metadata={"vlm_status": enrichment.status, "category": category},
                        )
                    )
                elif text:
                    elements.append(
                        Element.create(
                            type="image",
                            text=text,
                            page=page,
                            section=current_section,
                            metadata={"category": category, "vlm_status": "skipped"},
                        )
                    )
                continue

            if not text:
                continue
            elements.append(
                Element.create(
                    type=element_type,
                    text=text,
                    page=page,
                    section=current_section,
                    table_id=path.stem if element_type == "table" else None,
                    metadata={"category": category},
                )
            )

        if not elements:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            elements.append(Element.create(type="text", text=normalize_text(raw) or path.name))

        status = "ready"
        if any(item == "fallback" for item in statuses):
            status = "fallback"
        elif statuses and all(item == "disabled" for item in statuses):
            status = "disabled"

        return document_from_elements(
            path=path,
            route=route,
            elements=elements,
            title=title,
            extra={
                "element_type": "mixed",
                "actual_parser": "unstructured",
                "loader_status": status if statuses else "ready",
                "vlm_calls": budget.used,
            },
        )


def _map_category(category: str) -> ElementType:
    if "title" in category or "header" in category or "headline" in category:
        return "heading"
    if "table" in category:
        return "table"
    if "image" in category or "figure" in category:
        return "image"
    return "text"
