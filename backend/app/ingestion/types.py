from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Layer1 = Literal["image", "table", "document"]
PdfMode = Literal["digital", "ocr"]


@dataclass(frozen=True)
class RouteDecision:
    layer1: Layer1
    loader: str
    reason: str
    layer2: str | None = None
    file_type: str | None = None
    modality: str = "text"
    pdf_mode: PdfMode | None = None

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "route_layer1": payload["layer1"],
            "route_layer2": payload["layer2"],
            "loader": payload["loader"],
            "route_reason": payload["reason"],
            "modality": payload["modality"],
            "file_type": payload["file_type"],
            "pdf_mode": payload["pdf_mode"],
            "element_type": "text",
            "page": None,
            "section": None,
            "image_path": None,
            "table_id": None,
        }
