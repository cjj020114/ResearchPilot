from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.core.models import Document
from backend.app.ingestion.loaders.registry import build_loader_registry
from backend.app.ingestion.router import DocumentRouter
from backend.app.ingestion.types import RouteDecision


class IngestionPipeline:
    def __init__(self) -> None:
        self.router = DocumentRouter()
        self.loaders = build_loader_registry()

    def ingest_file(
        self,
        path: Path,
        title: str | None = None,
        force_loader: str | None = None,
    ) -> tuple[Document, RouteDecision]:
        route = self.router.route(path, force_loader=force_loader)
        loader = self.loaders.get(route.loader)
        if loader is None:
            raise ValueError(f"No loader registered for '{route.loader}'")
        document = loader.load(path, route, title=title)
        return document, route

    def route_only(self, path: Path, force_loader: str | None = None) -> RouteDecision:
        return self.router.route(path, force_loader=force_loader)

    def describe_route(self, route: RouteDecision) -> dict[str, Any]:
        return {
            "layer1": route.layer1,
            "layer2": route.layer2,
            "loader": route.loader,
            "reason": route.reason,
            "modality": route.modality,
            "file_type": route.file_type,
            "pdf_mode": route.pdf_mode,
        }
