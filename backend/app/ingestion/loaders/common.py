from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.app.core.models import Document, Element
from backend.app.ingestion.assemble import assemble_elements_text
from backend.app.ingestion.types import RouteDecision


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_title(text: str, fallback: str) -> str:
    """Infer a human-readable title, skipping page markers and placeholders."""
    for line in text.splitlines():
        cleaned = line.strip(" #\t")
        if not cleaned:
            continue
        if re.fullmatch(r"\[page:\d+\]", cleaned, flags=re.IGNORECASE):
            continue
        if cleaned.startswith("[") and cleaned.endswith("]") and (
            "placeholder" in cleaned.lower() or "fallback" in cleaned.lower()
        ):
            continue
        if cleaned.startswith("[page:") and cleaned.endswith("]"):
            continue
        if 5 <= len(cleaned) <= 120:
            return cleaned
    stem = Path(fallback).stem if fallback else fallback
    return stem or fallback


def attach_route_metadata(
    document: Document,
    route: RouteDecision,
    extra: dict[str, Any] | None = None,
) -> Document:
    metadata = {
        **route.to_metadata(),
        "filename": Path(document.source).name,
        **(extra or {}),
    }
    if "actual_parser" not in metadata:
        metadata["actual_parser"] = route.loader
    document.metadata = {**document.metadata, **metadata}
    return document


def document_from_elements(
    *,
    path: Path,
    route: RouteDecision,
    elements: list[Element],
    title: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Document:
    text = assemble_elements_text(elements)
    document = Document.create(
        title=title or infer_title(text, path.stem),
        source=str(path),
        text=text,
        metadata={"file_type": route.file_type or path.suffix.lstrip(".")},
        elements=elements,
    )
    return attach_route_metadata(document, route, extra)
