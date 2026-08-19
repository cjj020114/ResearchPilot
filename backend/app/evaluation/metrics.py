from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalExample:
    question: str
    relevant_chunk_ids: list[str]
    answer: str | None = None


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved & relevant) / len(relevant)


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Binary success: 1.0 if any relevant id appears in Top-k, else 0.0."""
    if not relevant_ids or k <= 0:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return 1.0 if retrieved & set(relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def context_precision(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = retrieved_ids[:k]
    if not retrieved:
        return 0.0
    relevant = set(relevant_ids)
    return sum(1 for chunk_id in retrieved if chunk_id in relevant) / len(retrieved)


def aggregate(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and v == v]  # drop None/NaN
    return sum(clean) / len(clean) if clean else 0.0
