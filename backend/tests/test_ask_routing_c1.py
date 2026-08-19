from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import backend.app.core.service as service_mod
from backend.app.core.models import Chunk, SearchResult
from backend.app.knowledge.router import RouteResult


class _DummyGenerator:
    def best_score(self, results: list[SearchResult]) -> float:
        return results[0].score if results else 0.0

    def generate(
        self,
        question: str,
        results: list[SearchResult],
        *,
        query_image_path: str | None = None,
    ) -> dict[str, Any]:
        del question, results, query_image_path
        return {
            "answer": "ok",
            "citations": [],
            "generator": "dummy",
            "trace": [],
            "confidence": "high",
        }

    def confidence_label(self, results: list[SearchResult], threshold: float = 0.0) -> str:
        del results, threshold
        return "high"


class _DummyExpander:
    def expand(self, question: str, max_sub_queries: int = 4) -> Any:
        del max_sub_queries
        return SimpleNamespace(
            retrieval_queries=lambda: [question or "image"],
            to_dict=lambda: {},
        )


class _DummyRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search_multi(self, *args: Any, **kwargs: Any) -> list[SearchResult]:
        self.calls.append({"args": args, "kwargs": kwargs})
        chunk = Chunk(id="c1", document_id="d1", text="t", metadata={})
        return [SearchResult(chunk=chunk, score=0.9, vector_score=0.9)]


def _make_service(monkeypatch: Any, tmp_path: Path) -> Any:
    svc = object.__new__(service_mod.ResearchPilotService)
    img = tmp_path / "q.png"
    img.write_bytes(b"x")
    svc.kb_registry = SimpleNamespace(
        list_all=lambda: [SimpleNamespace(id="kb_a"), SimpleNamespace(id="kb_b")],
        get=lambda kb_id: SimpleNamespace(id=kb_id),
        catalog_for_prompt=lambda: [{"id": "kb_a"}, {"id": "kb_b"}],
    )
    svc.kb_router = SimpleNamespace()
    svc.query_expander = _DummyExpander()
    svc.retriever = _DummyRetriever()
    svc.generator = _DummyGenerator()
    monkeypatch.setattr(service_mod, "settings", SimpleNamespace(
        kb_route_top_n=3,
        retrieval_candidate_k=5,
        retrieval_retry_candidate_k=10,
        retrieval_low_confidence_threshold=0.01,
        query_max_sub_queries=4,
        enable_clip_multimodal=True,
    ))
    return svc, img


def test_image_only_skips_routing(monkeypatch: Any, tmp_path: Path) -> None:
    svc, img = _make_service(monkeypatch, tmp_path)
    called = {"route": False}

    def _route(*_a: Any, **_k: Any) -> RouteResult:
        called["route"] = True
        raise AssertionError("route_query should not be called for image-only")

    svc.kb_router.route_query = _route
    out = svc.ask(question="", query_image_path=str(img), auto_route=True, use_rerank=True)
    assert called["route"] is False
    assert out["routing"]["routing_mode"] == "image_only_all_kbs"
    assert out["need_selection"] is False
    assert set(out["routing"]["knowledge_base_ids"]) == {"kb_a", "kb_b"}
    assert out["retrieval_retry"]["image_only_dense"] is True
    assert out["query_plan"]["reason"].startswith("image-only")
    assert svc.retriever.calls
    assert svc.retriever.calls[0]["kwargs"]["use_rerank"] is False
    assert svc.retriever.calls[0]["kwargs"]["query_image_path"] == str(img)


def test_text_image_fallback_when_router_uncertain(monkeypatch: Any, tmp_path: Path) -> None:
    svc, img = _make_service(monkeypatch, tmp_path)

    def _route(question: str, top_n: int = 3) -> RouteResult:
        del top_n
        assert "销量" in question
        return RouteResult(
            knowledge_base_ids=[],
            reason="用户未提供图片，无法确定",
            confidence=0.2,
            used_llm=True,
            need_selection=True,
            candidates=[{"id": "kb_a"}],
            message="请选择知识库",
        )

    svc.kb_router.route_query = _route
    out = svc.ask(
        question="这张销量图是什么",
        query_image_path=str(img),
        auto_route=True,
        use_rerank=False,
    )
    assert out["need_selection"] is False
    assert out["routing"]["routing_mode"] == "text_image_fallback_all_kbs"
    assert set(out["routing"]["knowledge_base_ids"]) == {"kb_a", "kb_b"}
    assert out.get("answer") == "ok"
