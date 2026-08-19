from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from backend.app.core.models import Chunk, SearchResult
from backend.app.retrieval.filters import matches_filters


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: dict[str, Chunk] = {}
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_freqs: defaultdict[str, int] = defaultdict(int)
        self.avg_doc_len = 0.0

    def rebuild(self, chunks: list[Chunk]) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.term_freqs.clear()
        self.doc_freqs.clear()
        lengths: list[int] = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            lengths.append(len(tokens))
            counts = Counter(tokens)
            self.term_freqs[chunk.id] = counts
            for token in counts:
                self.doc_freqs[token] += 1
        self.avg_doc_len = sum(lengths) / len(lengths) if lengths else 0.0

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_terms = tokenize(query)
        scores: list[SearchResult] = []
        total_docs = max(1, len(self.chunks))
        for chunk_id, chunk in self.chunks.items():
            if not matches_filters(chunk, filters):
                continue
            score = 0.0
            doc_len = sum(self.term_freqs[chunk_id].values()) or 1
            for term in query_terms:
                tf = self.term_freqs[chunk_id].get(term, 0)
                if tf == 0:
                    continue
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1))
                score += idf * numerator / denominator
            if score > 0:
                scores.append(SearchResult(chunk=chunk, score=score, bm25_score=score))
        return sorted(scores, key=lambda result: result.score, reverse=True)[:top_k]
