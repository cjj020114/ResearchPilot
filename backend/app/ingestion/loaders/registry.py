from __future__ import annotations

from backend.app.ingestion.loaders.base import BaseLoader
from backend.app.ingestion.loaders.markdown_loader import MarkdownLoader
from backend.app.ingestion.loaders.ocr_vlm_loader import OcrVlmLoader
from backend.app.ingestion.loaders.pdf_loader import PdfLoader
from backend.app.ingestion.loaders.table_loader import TableLoader
from backend.app.ingestion.loaders.text_loader import TextLoader
from backend.app.ingestion.loaders.unstructured_loader import UnstructuredLoader


def build_loader_registry() -> dict[str, BaseLoader]:
    unstructured = UnstructuredLoader()
    return {
        "text": TextLoader(),
        "markdown": MarkdownLoader(),
        "pdf": PdfLoader(),
        "ocr_vlm": OcrVlmLoader(),
        "table": TableLoader(),
        "unstructured": unstructured,
        # Aliases kept for force_loader / older clients.
        "docx": unstructured,
        "pptx": unstructured,
        "html": unstructured,
    }
