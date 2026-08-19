from __future__ import annotations

from backend.app.core.models import Chunk, SearchResult
from backend.app.retrieval.reranker import CloudReranker, LexicalReranker


def _result(chunk_id: str, text: str, score: float = 0.5) -> SearchResult:
    chunk = Chunk(
        id=chunk_id,
        document_id="doc",
        text=text,
        metadata={},
    )
    return SearchResult(chunk=chunk, score=score, vector_score=score, bm25_score=0.0)


def test_lexical_reranker_prefers_overlap() -> None:
    reranker = LexicalReranker()
    results = [
        _result("a", "unrelated astronomy content", score=0.55),
        _result("b", "hybrid bm25 dense retrieval ranking", score=0.50),
    ]
    ranked = reranker.rerank("hybrid bm25 dense retrieval", results, top_k=2)
    assert ranked[0].chunk.id == "b"


def test_cloud_reranker_falls_back_without_key() -> None:
    reranker = CloudReranker(api_key="", fallback=LexicalReranker())
    results = [
        _result("a", "apple fruit juice", score=0.50),
        _result("b", "banana yellow", score=0.51),
    ]
    ranked = reranker.rerank("apple fruit", results, top_k=1)
    assert len(ranked) == 1
    assert ranked[0].chunk.id == "a"


def test_cloud_reranker_parses_mock_response(monkeypatch: object) -> None:
    reranker = CloudReranker(
        api_key="test-key",
        api_base="https://example.com/v1",
        model="BAAI/bge-reranker-v2-m3",
        fallback=LexicalReranker(),
    )
    results = [
        _result("a", "doc a", score=0.8),
        _result("b", "doc b", score=0.7),
        _result("c", "doc c", score=0.6),
    ]

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 2, "relevance_score": 0.99},
                    {"index": 0, "relevance_score": 0.5},
                ]
            }

    class _Client:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> _Resp:
            return _Resp()

    import backend.app.retrieval.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod.httpx, "Client", _Client)  # type: ignore[attr-defined]
    ranked = reranker.rerank("q", results, top_k=2)
    assert [item.chunk.id for item in ranked] == ["c", "a"]
    assert ranked[0].rerank_score == 0.99
