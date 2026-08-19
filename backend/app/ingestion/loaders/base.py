from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.app.core.models import Document
from backend.app.ingestion.types import RouteDecision


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: Path, route: RouteDecision, title: str | None = None) -> Document:
        raise NotImplementedError
