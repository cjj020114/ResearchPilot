from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

ElementType = Literal["text", "heading", "image", "table", "markdown"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Element:
    id: str
    type: ElementType
    text: str
    ocr_text: str | None = None
    vlm_caption: str | None = None
    page: int | None = None
    section: str | None = None
    image_path: str | None = None
    table_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        type: ElementType,
        text: str,
        *,
        ocr_text: str | None = None,
        vlm_caption: str | None = None,
        page: int | None = None,
        section: str | None = None,
        image_path: str | None = None,
        table_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Element":
        return cls(
            id=str(uuid4()),
            type=type,
            text=text,
            ocr_text=ocr_text,
            vlm_caption=vlm_caption,
            page=page,
            section=section,
            image_path=image_path,
            table_id=table_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Element":
        return cls(
            id=str(payload["id"]),
            type=payload["type"],
            text=str(payload.get("text", "")),
            ocr_text=payload.get("ocr_text"),
            vlm_caption=payload.get("vlm_caption"),
            page=payload.get("page"),
            section=payload.get("section"),
            image_path=payload.get("image_path"),
            table_id=payload.get("table_id"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class Document:
    id: str
    title: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    elements: list[Element] | None = None

    @classmethod
    def create(
        cls,
        title: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        elements: list[Element] | None = None,
    ) -> "Document":
        return cls(
            id=str(uuid4()),
            title=title,
            source=source,
            text=text,
            metadata=metadata or {},
            elements=elements,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "elements": [element.to_dict() for element in self.elements]
            if self.elements is not None
            else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Document":
        raw_elements = payload.get("elements")
        elements: list[Element] | None
        if raw_elements is None:
            elements = None
        else:
            elements = [
                Element.from_dict(item) if isinstance(item, dict) else item
                for item in raw_elements
            ]
        return cls(
            id=str(payload["id"]),
            title=str(payload["title"]),
            source=str(payload["source"]),
            text=str(payload.get("text", "")),
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now()),
            elements=elements,
        )


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
