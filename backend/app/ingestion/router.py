from __future__ import annotations

from pathlib import Path

from backend.app.ingestion.detectors.modality import (
    DOCUMENT_TYPES,
    UNSUPPORTED_TYPES,
    detect_file_type,
    is_image_file,
    is_table_file,
)
from backend.app.ingestion.detectors.pdf_mode import detect_pdf_mode
from backend.app.ingestion.exceptions import UnsupportedFileTypeError
from backend.app.ingestion.types import RouteDecision


class DocumentRouter:
    """Route uploads: image | table | document (no MinerU)."""

    def route(self, path: Path, force_loader: str | None = None) -> RouteDecision:
        if force_loader:
            return self._forced_route(path, force_loader)

        if is_image_file(path):
            file_type = detect_file_type(path)
            return RouteDecision(
                layer1="image",
                loader="ocr_vlm",
                reason="standalone image -> single VLM call (OCR + caption)",
                file_type=file_type,
                modality="image",
            )

        if is_table_file(path):
            file_type = detect_file_type(path)
            return RouteDecision(
                layer1="table",
                loader="table",
                reason=f"table file ({file_type}) -> table parser",
                layer2=file_type,
                file_type=file_type,
                modality="table",
            )

        file_type = detect_file_type(path)
        if file_type in UNSUPPORTED_TYPES:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{file_type}'. "
                "json/yaml/xml are not ingested in the current routing design."
            )

        if file_type == "pdf":
            pdf_mode, reason = detect_pdf_mode(path)
            return RouteDecision(
                layer1="document",
                loader="pdf",
                reason=reason,
                layer2="pdf",
                file_type="pdf",
                modality="mixed",
                pdf_mode=pdf_mode,
            )

        loader = self._document_loader_for(file_type)
        if loader is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{file_type or path.suffix}'. "
                f"Supported document types: {sorted(DOCUMENT_TYPES)}"
            )

        return RouteDecision(
            layer1="document",
            loader=loader,
            reason=f"document type {file_type} -> {loader}",
            layer2=file_type,
            file_type=file_type,
            modality=self._modality_for(file_type),
        )

    def _document_loader_for(self, file_type: str) -> str | None:
        mapping = {
            "txt": "text",
            "log": "text",
            "md": "markdown",
            "docx": "unstructured",
            "ppt": "unstructured",
            "pptx": "unstructured",
            "html": "unstructured",
        }
        return mapping.get(file_type)

    def _modality_for(self, file_type: str) -> str:
        if file_type in {"docx", "ppt", "pptx", "html", "md"}:
            return "mixed"
        return "text"

    def _forced_route(self, path: Path, force_loader: str) -> RouteDecision:
        from backend.app.ingestion.types import Layer1

        file_type = detect_file_type(path)
        layer1: Layer1
        if is_image_file(path):
            layer1 = "image"
        elif is_table_file(path):
            layer1 = "table"
        else:
            layer1 = "document"
        pdf_mode = None
        if force_loader == "pdf" or file_type == "pdf":
            pdf_mode, _ = detect_pdf_mode(path)
        return RouteDecision(
            layer1=layer1,
            layer2=None if layer1 == "image" else file_type,
            loader=force_loader,
            reason=f"forced loader={force_loader}",
            file_type=file_type,
            modality="mixed" if force_loader in {"ocr_vlm", "pdf", "unstructured"} else "text",
            pdf_mode=pdf_mode,
        )
