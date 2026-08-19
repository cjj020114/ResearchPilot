from backend.app.evaluation.metrics import context_precision, recall_at_k, reciprocal_rank


def test_recall_at_k_counts_relevant_hits() -> None:
    assert recall_at_k(["a", "b", "c"], ["b", "d"], 2) == 0.5


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    assert reciprocal_rank(["x", "b", "c"], ["b"]) == 0.5


def test_context_precision_limits_to_top_k() -> None:
    assert context_precision(["a", "b", "c"], ["a", "c"], 2) == 0.5
