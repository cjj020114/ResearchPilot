from __future__ import annotations

import re
from pathlib import Path

from backend.app.core.models import Document, Element
from backend.app.ingestion.loaders.base import BaseLoader
from backend.app.ingestion.loaders.common import document_from_elements, normalize_text
from backend.app.ingestion.types import RouteDecision
from backend.app.ingestion.vision.vlm_client import VlmBudget, get_vlm_client

IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


class MarkdownLoader(BaseLoader):
    """Markdown -> Element[] (text/heading/image), images enriched via VLM."""

    def load(self, path: Path, route: RouteDecision, title: str | None = None) -> Document:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        elements: list[Element] = []
        budget = VlmBudget()
        client = get_vlm_client()
        current_section: str | None = None
        buffer: list[str] = []
        vlm_statuses: list[str] = []

        def flush_text() -> None:
            nonlocal buffer
            text = normalize_text("\n".join(buffer))
            buffer = []
            if text:
                elements.append(
                    Element.create(type="text", text=text, section=current_section)
                )

        for line in raw.splitlines():
            heading = HEADING_PATTERN.match(line.strip())
            if heading:
                flush_text()
                current_section = heading.group(2).strip()
                elements.append(
                    Element.create(
                        type="heading",
                        text=f"{heading.group(1)} {current_section}",
                        section=current_section,
                    )
                )
                continue

            image = IMAGE_PATTERN.search(line)
            if image:
                flush_text()
                alt = image.group(1).strip()
                target = image.group(2).strip().strip("<>").split()[0]
                image_path = (path.parent / target).resolve() if not Path(target).is_absolute() else Path(target)
                context = f"section={current_section or ''}; alt={alt}"
                if image_path.exists() and image_path.is_file():
                    enrichment = client.enrich_image(
                        image_path,
                        context=context,
                        budget=budget,
                        fallback_label=image_path.name,
                    )
                    vlm_statuses.append(enrichment.status)
                    combined = enrichment.combined_text or alt or image_path.name
                    elements.append(
                        Element.create(
                            type="image",
                            text=combined,
                            ocr_text=enrichment.ocr_text or None,
                            vlm_caption=enrichment.vlm_caption or None,
                            section=current_section,
                            image_path=str(image_path),
                            metadata={"alt": alt, "vlm_status": enrichment.status},
                        )
                    )
                else:
                    elements.append(
                        Element.create(
                            type="image",
                            text=alt or f"[missing image] {target}",
                            section=current_section,
                            image_path=str(image_path),
                            metadata={"alt": alt, "missing": True},
                        )
                    )
                # Keep any non-image remainder on the line as text.
                remainder = IMAGE_PATTERN.sub("", line).strip()
                if remainder:
                    buffer.append(remainder)
                continue

            buffer.append(line)

        flush_text()
        if not elements:
            elements.append(Element.create(type="markdown", text=normalize_text(raw)))

        heading_title = None
        for element in elements:
            if element.type == "heading":
                heading_title = element.section
                break

        overall_status = "ready"
        if any(status == "fallback" for status in vlm_statuses):
            overall_status = "fallback"
        elif any(status == "disabled" for status in vlm_statuses):
            overall_status = "disabled" if vlm_statuses else "ready"

        return document_from_elements(
            path=path,
            route=route,
            elements=elements,
            title=title or heading_title,
            extra={
                "section": heading_title,
                "element_type": "markdown",
                "actual_parser": "markdown",
                "loader_status": overall_status if vlm_statuses else "ready",
                "vlm_calls": budget.used,
            },
        )
