from __future__ import annotations

from typing import Any, Literal, cast

from backend.app.core.models import SearchResult
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.reranker import LexicalReranker, Reranker

RetrievalMode = Literal["dense", "bm25", "hybrid"]


class HybridRetriever:
    def __init__(
        self,
        vector_index: Any,
        bm25_index: BM25Index,
        reranker: Reranker | None = None,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
    ) -> None:
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.reranker = reranker or LexicalReranker()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 20,
        filters: dict[str, Any] | None = None,
        use_rerank: bool = True,
        mode: RetrievalMode = "hybrid",
        query_image_path: str | None = None,
    ) -> list[SearchResult]:
        if mode == "dense":
            merged = cast(
                list[SearchResult],
                self.vector_index.vector_search(
                    query,
                    top_k=candidate_k,
                    filters=filters,
                    query_image_path=query_image_path,
                ),
            )
            for result in merged:
                result.score = result.vector_score
        elif mode == "bm25":
            # BM25 is text-only; if only image query, fall back to dense multimodal.
            if (not query or not query.strip()) and query_image_path:
                merged = cast(
                    list[SearchResult],
                    self.vector_index.vector_search(
                        "",
                        top_k=candidate_k,
                        filters=filters,
                        query_image_path=query_image_path,
                    ),
                )
                for result in merged:
                    result.score = result.vector_score
            else:
                merged = self.bm25_index.search(query, top_k=candidate_k, filters=filters)
                for result in merged:
                    result.score = result.bm25_score
        else:
            vector_results = cast(
                list[SearchResult],
                self.vector_index.vector_search(
                    query,
                    top_k=candidate_k,
                    filters=filters,
                    query_image_path=query_image_path,
                ),
            )
            if query and query.strip():
                bm25_results = self.bm25_index.search(query, top_k=candidate_k, filters=filters)
                merged = self._merge(vector_results, bm25_results)
            else:
                merged = vector_results
                for result in merged:
                    result.score = result.vector_score

        if use_rerank and query and query.strip():
            return self.reranker.rerank(query, merged, top_k=top_k)
        return merged[:top_k]

    def search_multi(
        self,
        queries: list[str],
        top_k: int = 6,
        candidate_k: int = 20,
        filters: dict[str, Any] | None = None,
        use_rerank: bool = True,
        mode: RetrievalMode = "hybrid",
        query_image_path: str | None = None,
    ) -> list[SearchResult]:
        """Run search for each query and merge by chunk id (keep best score)."""
        by_chunk: dict[str, SearchResult] = {}
        unique_queries = [q.strip() for q in queries if q and q.strip()]
        if not unique_queries and query_image_path:
            unique_queries = [""]
        if not unique_queries:
            return []
        per_query_k = max(top_k, candidate_k)
        for query in unique_queries:
            results = self.search(
                query,
                top_k=per_query_k,
                candidate_k=candidate_k,
                filters=filters,
                use_rerank=use_rerank,
                mode=mode,
                query_image_path=query_image_path,
            )
            for result in results:
                existing = by_chunk.get(result.chunk.id)
                if existing is None or result.score > existing.score:
                    by_chunk[result.chunk.id] = result
        merged = sorted(by_chunk.values(), key=lambda item: item.score, reverse=True)
        return merged[:top_k]

    def _merge(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> list[SearchResult]:
        by_chunk: dict[str, SearchResult] = {}
        max_vector = max((result.vector_score for result in vector_results), default=1.0) or 1.0
        max_bm25 = max((result.bm25_score for result in bm25_results), default=1.0) or 1.0

        for result in vector_results:
            normalized = result.vector_score / max_vector
            result.score = self.vector_weight * normalized
            by_chunk[result.chunk.id] = result

        for result in bm25_results:
            normalized = result.bm25_score / max_bm25
            existing = by_chunk.get(result.chunk.id)
            if existing:
                existing.bm25_score = result.bm25_score
                existing.score += self.bm25_weight * normalized
            else:
                result.score = self.bm25_weight * normalized
                by_chunk[result.chunk.id] = result

        return sorted(by_chunk.values(), key=lambda item: item.score, reverse=True)
