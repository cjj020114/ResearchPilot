from __future__ import annotations

import logging
from typing import Protocol

import httpx

from backend.app.core.config import settings
from backend.app.core.models import SearchResult
from backend.app.retrieval.bm25 import tokenize

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...


class LexicalReranker:
    """Lightweight lexical overlap reranker (fallback / offline)."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return results[:top_k]
        ranked: list[SearchResult] = []
        for result in results:
            chunk_terms = set(tokenize(result.chunk.text))
            lexical_overlap = len(query_terms & chunk_terms) / len(query_terms)
            citation_bonus = 0.05 if result.chunk.metadata.get("page") else 0.0
            rerank_score = 0.65 * result.score + 0.30 * lexical_overlap + citation_bonus
            result.rerank_score = rerank_score
            result.score = rerank_score
            ranked.append(result)
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]


class CloudReranker:
    """OpenAI-compatible / SiliconFlow POST /rerank client."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        fallback: LexicalReranker | None = None,
    ) -> None:
        self.api_base = (api_base or settings.reranker_api_base).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.reranker_api_key
        self.model = model or settings.reranker_model
        self.timeout = timeout if timeout is not None else settings.reranker_timeout_seconds
        self.fallback = fallback or LexicalReranker()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model and self.api_base)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        if not self.enabled:
            logger.warning("Cloud reranker not configured; falling back to lexical reranker.")
            return self.fallback.rerank(query, results, top_k=top_k)
        try:
            return self._rerank_cloud(query, results, top_k=top_k)
        except Exception as exc:
            logger.warning("Cloud reranker failed (%s); falling back to lexical reranker.", exc)
            return self.fallback.rerank(query, results, top_k=top_k)

    def _rerank_cloud(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        documents = [result.chunk.text for result in results]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.api_base}/rerank"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        ranked_items = data.get("results") or data.get("data") or []
        if not isinstance(ranked_items, list) or not ranked_items:
            raise ValueError(f"Unexpected rerank response: {data!r}")

        reranked: list[SearchResult] = []
        seen: set[int] = set()
        for item in ranked_items:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index")
            if raw_index is None:
                continue
            index = int(raw_index)
            if index < 0 or index >= len(results) or index in seen:
                continue
            seen.add(index)
            raw_score = item.get("relevance_score", item.get("score", 0.0))
            score = float(raw_score if raw_score is not None else 0.0)
            result = results[index]
            result.rerank_score = score
            result.score = score
            reranked.append(result)

        # Keep any candidates the API omitted, preserving original relative order.
        for index, result in enumerate(results):
            if index not in seen:
                reranked.append(result)
        return reranked[:top_k]


def build_reranker() -> LexicalReranker | CloudReranker:
    provider = (settings.reranker_provider or "cloud").strip().lower()
    if provider == "lexical":
        return LexicalReranker()
    if provider == "cloud":
        return CloudReranker()
    logger.warning("Unknown RERANKER_PROVIDER=%s; using lexical.", provider)
    return LexicalReranker()
