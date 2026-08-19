from backend.app.core.models import Chunk
from backend.app.retrieval.filters import build_scope_filter, matches_filters


def test_build_scope_filter_defaults_to_general_scope() -> None:
    filters = build_scope_filter()

    assert filters["domain"] == "general"
    assert filters["knowledge_base_id"] == "kb_general"


def test_matches_filters_rejects_other_domain() -> None:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        text="content",
        metadata={"domain": "ai", "knowledge_base_id": "kb_ai"},
    )

    assert matches_filters(chunk, {"domain": "ai"})
    assert not matches_filters(chunk, {"domain": "law"})


def test_matches_filters_supports_multi_domain_query() -> None:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        text="content",
        metadata={"domain": "ai", "knowledge_base_id": "kb_ai"},
    )

    assert matches_filters(chunk, {"domain": ["ai", "law"]})
