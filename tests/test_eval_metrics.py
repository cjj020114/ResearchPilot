from __future__ import annotations

from backend.app.evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank


def test_hit_at_k() -> None:
    retrieved = ["a", "b", "c"]
    assert hit_at_k(retrieved, ["c"], k=3) == 1.0
    assert hit_at_k(retrieved, ["c"], k=2) == 0.0
    assert hit_at_k(retrieved, [], k=3) == 0.0


def test_recall_and_mrr() -> None:
    retrieved = ["x", "y", "z"]
    assert recall_at_k(retrieved, ["y", "z"], k=3) == 1.0
    assert reciprocal_rank(retrieved, ["y"]) == 0.5
