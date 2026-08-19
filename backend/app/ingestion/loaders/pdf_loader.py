from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from backend.app.core.models import Document, Element
from backend.app.ingestion.loaders.base import BaseLoader
from backend.app.ingestion.loaders.common import document_from_elements, infer_title, normalize_text
from backend.app.ingestion.types import RouteDecision
from backend.app.ingestion.vision.vlm_client import VlmBudget, get_vlm_client


class PdfLoader(BaseLoader):
    """PDF digital (text + embedded images) or OCR mode (page images via VLM)."""

    def load(self, path: Path, route: RouteDecision, title: str | None = None) -> Document:
        mode = route.pdf_mode or "digital"
        budget = VlmBudget()
        if mode == "ocr":
            elements, parser, status = self._load_ocr(path, budget)
        else:
            elements, parser, status = self._load_digital(path, budget)

        text = normalize_text("\n\n".join(el.text for el in elements if el.text))
        return document_from_elements(
            path=path,
            route=route,
            elements=elements,
            title=title or _pdf_meta_title(path) or infer_title(text, path.stem),
            extra={
                "page_count": _page_count(path),
                "element_type": "mixed",
                "actual_parser": parser,
                "loader_status": status,
                "pdf_mode": mode,
                "vlm_calls": budget.used,
            },
        )

    def _load_digital(
        self, path: Path, budget: VlmBudget
    ) -> tuple[list[Element], str, str]:
        reader = PdfReader(str(path))
        client = get_vlm_client()
        elements: list[Element] = []
        statuses: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = normalize_text(page.extract_text() or "")
            if page_text:
                elements.append(
                    Element.create(
                        type="text",
                        text=page_text,
                        page=index,
                        metadata={"source": "pypdf"},
                    )
                )
            for image_index, image in enumerate(getattr(page, "images", []) or [], start=1):
                raw = getattr(image, "data", None)
                if not raw:
                    continue
                name = getattr(image, "name", f"page{index}_img{image_index}")
                enrichment = client.enrich_image(
                    raw,
                    mime_type="image/png",
                    context=f"pdf digital page={index}; nearby text={page_text[:400]}",
                    budget=budget,
                    fallback_label=str(name),
                )
                statuses.append(enrichment.status)
                elements.append(
                    Element.create(
                        type="image",
                        text=enrichment.combined_text,
                        ocr_text=enrichment.ocr_text or None,
                        vlm_caption=enrichment.vlm_caption or None,
                        page=index,
                        image_path=str(name),
                        metadata={"vlm_status": enrichment.status, "embedded": True},
                    )
                )
        if not elements:
            elements.append(
                Element.create(
                    type="text",
                    text="[pdf_digital_empty] no extractable text or images",
                    page=1,
                )
            )
        status = _aggregate_status(statuses) if statuses else "ready"
        return elements, "pypdf_digital", status

    def _load_ocr(
        self, path: Path, budget: VlmBudget
    ) -> tuple[list[Element], str, str]:
        client = get_vlm_client()
        elements: list[Element] = []
        statuses: list[str] = []
        try:
            import pypdfium2 as pdfium
        except ImportError:
            return (
                [
                    Element.create(
                        type="text",
                        text="[pdf_ocr_fallback] pypdfium2 unavailable",
                        page=1,
                    )
                ],
                "pdf_ocr_unavailable",
                "fallback",
            )

        pdf = pdfium.PdfDocument(str(path))
        page_count = len(pdf)
        for index in range(page_count):
            page_no = index + 1
            if budget.remaining <= 0:
                elements.append(
                    Element.create(
                        type="text",
                        text=f"[pdf_ocr_skipped] page={page_no} (vlm_max_calls reached)",
                        page=page_no,
                        metadata={"skipped": True},
                    )
                )
                continue
            page = pdf[index]
            bitmap = page.render(scale=2).to_pil()
            from io import BytesIO

            buffer = BytesIO()
            bitmap.save(buffer, format="PNG")
            enrichment = client.enrich_image(
                buffer.getvalue(),
                mime_type="image/png",
                context=f"scanned pdf page={page_no}",
                budget=budget,
                fallback_label=f"{path.name}#page{page_no}",
            )
            statuses.append(enrichment.status)
            elements.append(
                Element.create(
                    type="image",
                    text=enrichment.combined_text,
                    ocr_text=enrichment.ocr_text or None,
                    vlm_caption=enrichment.vlm_caption or None,
                    page=page_no,
                    metadata={"vlm_status": enrichment.status, "pdf_mode": "ocr"},
                )
            )
        status = _aggregate_status(statuses) if statuses else "fallback"
        return elements, "pdf_ocr_vlm", status


def _aggregate_status(statuses: list[str]) -> str:
    if any(status == "fallback" for status in statuses):
        return "fallback"
    if statuses and all(status == "disabled" for status in statuses):
        return "disabled"
    if any(status == "ready" for status in statuses):
        return "ready"
    return statuses[0] if statuses else "ready"


def _page_count(path: Path) -> int:
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001
        return 0


def _pdf_meta_title(path: Path) -> str | None:
    try:
        metadata = PdfReader(str(path)).metadata
    except Exception:  # noqa: BLE001
        return None
    if not metadata:
        return None
    raw = getattr(metadata, "title", None)
    if raw is None and hasattr(metadata, "get"):
        raw = metadata.get("/Title")
    if not raw:
        return None
    title = str(raw).strip()
    if not title or title.lower() in {"untitled", "null"}:
        return None
    return title[:200]
