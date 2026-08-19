from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.core.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_path_for_store(vector_store: str | None = None) -> Path:
    """Return the KB registry file for the active vector store (local vs qdrant)."""
    store = (vector_store or settings.vector_store).strip().lower()
    if store == "qdrant":
        return settings.storage_dir / "knowledge_bases_qdrant.json"
    return settings.storage_dir / "knowledge_bases_local.json"


def _migrate_legacy_local_registry(target: Path) -> None:
    """One-time copy of legacy knowledge_bases.json into the local registry file."""
    if target.exists():
        return
    legacy = settings.storage_dir / "knowledge_bases.json"
    if not legacy.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")


@dataclass
class KnowledgeBase:
    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeBase":
        return cls(
            id=str(payload["id"]),
            name=str(payload.get("name") or payload["id"]),
            description=str(payload.get("description") or ""),
            created_at=str(payload.get("created_at") or _utc_now()),
        )


class KnowledgeBaseRegistry:
    """Persistent catalog of knowledge bases for LLM routing/assignment."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or registry_path_for_store()
        if path is None and self.path.name == "knowledge_bases_local.json":
            _migrate_legacy_local_registry(self.path)
        self._items: dict[str, KnowledgeBase] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._items = {}
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        items = payload.get("knowledge_bases", [])
        self._items = {
            str(item["id"]): KnowledgeBase.from_dict(item)
            for item in items
            if isinstance(item, dict) and item.get("id")
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "knowledge_bases": [item.to_dict() for item in self.list_all()],
            "updated_at": _utc_now(),
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_all(self) -> list[KnowledgeBase]:
        return sorted(self._items.values(), key=lambda item: item.created_at)

    def get(self, knowledge_base_id: str) -> KnowledgeBase | None:
        return self._items.get(knowledge_base_id)

    def ensure(
        self,
        *,
        name: str,
        description: str = "",
        knowledge_base_id: str | None = None,
    ) -> KnowledgeBase:
        if knowledge_base_id and knowledge_base_id in self._items:
            existing = self._items[knowledge_base_id]
            changed = False
            if name and existing.name != name:
                existing.name = name
                changed = True
            if description and existing.description != description:
                existing.description = description
                changed = True
            if changed:
                self.save()
            return existing
        new_id = knowledge_base_id or f"kb_{uuid4().hex[:10]}"
        while new_id in self._items:
            new_id = f"kb_{uuid4().hex[:10]}"
        item = KnowledgeBase(id=new_id, name=name.strip() or new_id, description=description.strip())
        self._items[new_id] = item
        self.save()
        return item

    def upsert_from_document_metadata(self, metadata: dict[str, Any]) -> KnowledgeBase | None:
        kb_id = str(metadata.get("knowledge_base_id") or "").strip()
        if not kb_id:
            return None
        name = str(metadata.get("knowledge_base_name") or metadata.get("domain") or kb_id)
        description = str(metadata.get("knowledge_base_description") or "")
        return self.ensure(name=name, description=description, knowledge_base_id=kb_id)

    def catalog_for_prompt(self) -> list[dict[str, str]]:
        return [
            {"id": item.id, "name": item.name, "description": item.description}
            for item in self.list_all()
        ]

    def delete(self, knowledge_base_id: str) -> bool:
        if knowledge_base_id not in self._items:
            return False
        del self._items[knowledge_base_id]
        self.save()
        return True
