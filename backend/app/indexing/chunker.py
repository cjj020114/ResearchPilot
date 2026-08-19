from __future__ import annotations

import re
from enum import StrEnum
from uuid import uuid4

from backend.app.core.models import Chunk, Document, Element
from backend.app.ingestion.assemble import element_display_text


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    HEADING = "heading"


class Chunker:
    def __init__(self, chunk_size: int = 900, overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document, strategy: ChunkStrategy = ChunkStrategy.HEADING) -> list[Chunk]:
        if document.elements:
            return self._chunk_elements(document, strategy)
        return self._chunk_plain_text(document, strategy)

    def _chunk_plain_text(
        self, document: Document, strategy: ChunkStrategy
    ) -> list[Chunk]:
        if strategy == ChunkStrategy.FIXED:
            texts = self._fixed(document.text)
        elif strategy == ChunkStrategy.RECURSIVE:
            texts = self._recursive(document.text)
        else:
            texts = self._heading_aware(document.text)
        return self._build_chunks(document, texts, strategy=strategy, element=None)

    def _chunk_elements(
        self, document: Document, strategy: ChunkStrategy
    ) -> list[Chunk]:
        assert document.elements is not None
        chunks: list[Chunk] = []
        for element in document.elements:
            body = element_display_text(element)
            if not body:
                continue
            if element.type == "image":
                pieces = [body]
            elif element.type == "heading" and len(body) <= self.chunk_size:
                pieces = [body]
            else:
                pieces = self._split_element_text(body, strategy)
            for piece in pieces:
                built = self._build_chunks(
                    document, [piece], strategy=strategy, element=element
                )
                chunks.extend(built)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks

    def _split_element_text(self, text: str, strategy: ChunkStrategy) -> list[str]:
        if strategy == ChunkStrategy.FIXED:
            return self._fixed(text)
        if strategy == ChunkStrategy.RECURSIVE:
            return self._recursive(text)
        return self._heading_aware(text)

    def _build_chunks(
        self,
        document: Document,
        texts: list[str],
        strategy: ChunkStrategy,
        element: Element | None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, text in enumerate(texts):
            page = element.page if element and element.page is not None else self._page_for_text(text)
            heading = (
                element.section
                if element and element.section
                else self._heading_for_text(text)
            )
            metadata = {
                **document.metadata,
                "document_title": document.title,
                "source": document.source,
                "chunk_index": index,
                "chunk_strategy": strategy.value,
                "chunk_by_element": element is not None,
                "page": page,
                "heading": heading,
            }
            if element is not None:
                metadata.update(
                    {
                        "element_id": element.id,
                        "element_type": element.type,
                        "section": element.section,
                        "image_path": element.image_path,
                        "table_id": element.table_id,
                        "modality": "image" if element.type == "image" else "text",
                        "asset_id": element.id if element.type == "image" else None,
                        "ocr_text": element.ocr_text,
                        "vlm_caption": element.vlm_caption,
                    }
                )
            else:
                metadata["modality"] = "text"
            chunks.append(
                Chunk(
                    id=str(uuid4()),
                    document_id=document.id,
                    text=text.strip(),
                    metadata=metadata,
                )
            )
        return [chunk for chunk in chunks if chunk.text]

    def _fixed(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks: list[str] = []
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, len(text), step):
            chunks.append(text[start : start + self.chunk_size])
        return chunks

    def _recursive(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= self.chunk_size:
                current = paragraph
            else:
                chunks.extend(self._fixed(paragraph))
                current = ""
        if current:
            chunks.append(current)
        return chunks or [text]

    def _heading_aware(self, text: str) -> list[str]:
        sections = re.split(r"(?m)(?=^#{1,4}\s+|^\d+(\.\d+)*\s+[A-Z][^\n]{3,})", text)
        merged = "\n".join(part for part in sections if part and not re.fullmatch(r"\d+", part))
        return self._recursive(merged)

    def _page_for_text(self, text: str) -> int | None:
        matches = re.findall(r"\[page:(\d+)\]", text)
        return int(matches[-1]) if matches else None

    def _heading_for_text(self, text: str) -> str | None:
        for line in text.splitlines():
            clean = line.strip()
            if clean.startswith("#"):
                return clean.lstrip("# ").strip()
            if re.match(r"^\d+(\.\d+)*\s+\S+", clean):
                return clean
        return None
