from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.knowledge.llm_client import TextLLMClient


@dataclass
class QueryPlan:
    original: str
    rewritten: str = ""
    sub_queries: list[str] = field(default_factory=list)
    step_back: str = ""
    used_llm: bool = False
    reason: str = ""

    def retrieval_queries(self) -> list[str]:
        queries: list[str] = []
        for item in [self.original, self.rewritten, self.step_back, *self.sub_queries]:
            text = (item or "").strip()
            if text and text not in queries:
                queries.append(text)
        return queries or [self.original]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "sub_queries": self.sub_queries,
            "step_back": self.step_back,
            "used_llm": self.used_llm,
            "reason": self.reason,
            "retrieval_queries": self.retrieval_queries(),
        }


class QueryExpander:
    """LLM query rewrite + decomposition + step-back expansion."""

    def __init__(self, llm: TextLLMClient | None = None) -> None:
        self.llm = llm or TextLLMClient()

    def expand(self, question: str, max_sub_queries: int = 4) -> QueryPlan:
        question = question.strip()
        if not question:
            return QueryPlan(original=question, reason="empty question")
        if not self.llm.enabled:
            return QueryPlan(
                original=question,
                rewritten=question,
                used_llm=False,
                reason="LLM not configured; using original question only",
            )

        system = (
            "You expand a research question for retrieval.\n"
            "Return ONLY JSON with keys:\n"
            "  rewritten: string (retrieval-friendly rewrite, keep user intent),\n"
            "  sub_queries: string[] (2 to 4 atomic sub-questions if complex; else 1-2),\n"
            "  step_back: string (more abstract step-back question),\n"
            "  reason: string.\n"
            "Do not invent facts; only rephrase/decompose for search."
        )
        user = f"User question:\n{question}\n"
        try:
            data = self.llm.chat_json(system, user)
        except Exception as exc:  # noqa: BLE001
            return QueryPlan(
                original=question,
                rewritten=question,
                used_llm=False,
                reason=f"query expansion failed: {exc}",
            )

        raw_subs = data.get("sub_queries") or []
        if isinstance(raw_subs, str):
            raw_subs = [raw_subs]
        sub_queries = [str(item).strip() for item in raw_subs if str(item).strip()]
        sub_queries = sub_queries[:max_sub_queries]
        rewritten = str(data.get("rewritten") or question).strip() or question
        step_back = str(data.get("step_back") or "").strip()
        return QueryPlan(
            original=question,
            rewritten=rewritten,
            sub_queries=sub_queries,
            step_back=step_back,
            used_llm=True,
            reason=str(data.get("reason") or ""),
        )
