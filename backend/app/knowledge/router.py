from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.knowledge.llm_client import TextLLMClient
from backend.app.knowledge.registry import KnowledgeBase, KnowledgeBaseRegistry


@dataclass
class AssignmentResult:
    knowledge_base: KnowledgeBase
    created_new: bool
    reason: str
    confidence: float
    used_llm: bool


@dataclass
class RouteResult:
    knowledge_base_ids: list[str]
    need_selection: bool
    reason: str
    confidence: float
    candidates: list[dict[str, str]]
    used_llm: bool
    message: str | None = None


class KnowledgeLLMRouter:
    """LLM-based upload assignment and query routing over the live KB registry."""

    def __init__(self, registry: KnowledgeBaseRegistry, llm: TextLLMClient | None = None) -> None:
        self.registry = registry
        self.llm = llm or TextLLMClient()

    def assign_document(self, *, title: str, snippet: str, filename: str = "") -> AssignmentResult:
        catalog = self.registry.catalog_for_prompt()
        if not self.llm.enabled:
            kb = self._fallback_assign(title=title, filename=filename, catalog_empty=not catalog)
            return AssignmentResult(
                knowledge_base=kb,
                created_new=kb.id not in {item["id"] for item in catalog},
                reason="LLM not configured; used heuristic assignment",
                confidence=0.2,
                used_llm=False,
            )

        system = (
            "You assign research documents to knowledge bases.\n"
            "Return ONLY JSON with keys:\n"
            '  action: "use_existing" | "create_new",\n'
            "  knowledge_base_id: string|null (required when use_existing),\n"
            "  name: string (required when create_new),\n"
            "  description: string,\n"
            "  reason: string,\n"
            "  confidence: number between 0 and 1.\n"
            "Prefer use_existing when a listed knowledge base clearly matches.\n"
            "Create a new knowledge base only when none fit well."
        )
        user = (
            f"Available knowledge bases:\n{catalog}\n\n"
            f"Document filename: {filename}\n"
            f"Document title: {title}\n"
            f"Document snippet:\n{snippet[:2500]}\n"
        )
        data = self.llm.chat_json(system, user)
        action = str(data.get("action") or "").strip().lower()
        confidence = _as_float(data.get("confidence"), 0.5)
        reason = str(data.get("reason") or "")

        if action == "use_existing":
            kb_id = str(data.get("knowledge_base_id") or "").strip()
            existing = self.registry.get(kb_id) if kb_id else None
            if existing is not None:
                return AssignmentResult(
                    knowledge_base=existing,
                    created_new=False,
                    reason=reason or f"assigned to existing {existing.id}",
                    confidence=confidence,
                    used_llm=True,
                )

        name = str(data.get("name") or title or filename or "未命名知识库").strip()
        description = str(data.get("description") or snippet[:240]).strip()
        created = self.registry.ensure(name=name, description=description)
        return AssignmentResult(
            knowledge_base=created,
            created_new=True,
            reason=reason or f"created knowledge base {created.id}",
            confidence=confidence,
            used_llm=True,
        )

    def route_query(self, question: str, top_n: int = 3) -> RouteResult:
        catalog = self.registry.catalog_for_prompt()
        if not catalog:
            return RouteResult(
                knowledge_base_ids=[],
                need_selection=True,
                reason="no knowledge bases registered",
                confidence=0.0,
                candidates=[],
                used_llm=False,
                message="当前还没有知识库，请先上传文档。",
            )

        if not self.llm.enabled:
            return RouteResult(
                knowledge_base_ids=[],
                need_selection=True,
                reason="LLM not configured",
                confidence=0.0,
                candidates=catalog,
                used_llm=False,
                message="请选择知识库（文本 LLM 未配置，无法自动路由）。",
            )

        system = (
            "You route a user question to research knowledge bases.\n"
            "Return ONLY JSON with keys:\n"
            "  knowledge_base_ids: string[] (1 to 3 ids from the catalog),\n"
            "  need_selection: boolean (true if uncertain),\n"
            "  reason: string,\n"
            "  confidence: number between 0 and 1.\n"
            "If unsure which bases apply, set need_selection=true and "
            "knowledge_base_ids to the best candidates (may be empty)."
        )
        user = f"Available knowledge bases:\n{catalog}\n\nUser question:\n{question}\n"
        data = self.llm.chat_json(system, user)
        raw_ids = data.get("knowledge_base_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        valid_ids = [str(item) for item in raw_ids if self.registry.get(str(item))]
        valid_ids = valid_ids[: max(1, top_n)]
        confidence = _as_float(data.get("confidence"), 0.0)
        need_selection = bool(data.get("need_selection")) or confidence < 0.45 or not valid_ids
        reason = str(data.get("reason") or "")
        if need_selection:
            candidates = [item for item in catalog if item["id"] in valid_ids] or catalog
            return RouteResult(
                knowledge_base_ids=valid_ids,
                need_selection=True,
                reason=reason or "uncertain routing",
                confidence=confidence,
                candidates=candidates,
                used_llm=True,
                message="请选择知识库",
            )
        return RouteResult(
            knowledge_base_ids=valid_ids,
            need_selection=False,
            reason=reason,
            confidence=confidence,
            candidates=[item for item in catalog if item["id"] in valid_ids],
            used_llm=True,
            message=None,
        )

    def _fallback_assign(
        self, *, title: str, filename: str, catalog_empty: bool
    ) -> KnowledgeBase:
        if not catalog_empty and self.registry.list_all():
            return self.registry.list_all()[0]
        name = (title or filename or "默认知识库").strip()[:80]
        return self.registry.ensure(
            name=name,
            description=f"Auto-created for {filename or title}",
        )


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
