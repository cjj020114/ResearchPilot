from __future__ import annotations

from pathlib import Path

from backend.app.core.service import ResearchPilotService
from backend.app.generation.answer import AnswerGenerator
from backend.app.indexing.chunker import ChunkStrategy, Chunker
from backend.app.indexing.embeddings import HashEmbeddingProvider
from backend.app.indexing.store import ResearchIndex
from backend.app.ingestion.parser import DocumentParser
from backend.app.knowledge.llm_client import TextLLMClient
from backend.app.knowledge.registry import KnowledgeBaseRegistry
from backend.app.knowledge.router import AssignmentResult, KnowledgeLLMRouter, RouteResult
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.query_expansion import QueryExpander, QueryPlan


class _ScriptedRouter(KnowledgeLLMRouter):
    def assign_document(self, *, title: str, snippet: str, filename: str = "") -> AssignmentResult:
        if "Contract" in snippet or "Law" in title:
            kb = self.registry.ensure(
                name="law",
                description="law notes",
                knowledge_base_id="kb_law",
            )
            return AssignmentResult(kb, False, "scripted law", 1.0, False)
        kb = self.registry.ensure(
            name="ai_research",
            description="ai notes",
            knowledge_base_id="kb_ai_research",
        )
        return AssignmentResult(kb, False, "scripted ai", 1.0, False)

    def route_query(self, question: str, top_n: int = 3) -> RouteResult:
        if "Transformer" in question or "self-attention" in question.lower():
            return RouteResult(
                knowledge_base_ids=["kb_ai_research"],
                need_selection=False,
                reason="scripted ai route",
                confidence=0.9,
                candidates=self.registry.catalog_for_prompt(),
                used_llm=False,
            )
        return RouteResult(
            knowledge_base_ids=["kb_law"],
            need_selection=False,
            reason="scripted law route",
            confidence=0.9,
            candidates=self.registry.catalog_for_prompt(),
            used_llm=False,
        )


def _build_local_service(tmp_path: Path) -> ResearchPilotService:
    service = ResearchPilotService.__new__(ResearchPilotService)
    embedding_provider = HashEmbeddingProvider(dimensions=32)
    service.parser = DocumentParser()
    service.pipeline = __import__(
        "backend.app.ingestion.pipeline", fromlist=["IngestionPipeline"]
    ).IngestionPipeline()
    service.chunker = Chunker(chunk_size=200, overlap=20)
    service.index = ResearchIndex(
        embedding_provider=embedding_provider,
        storage_path=tmp_path / "index.json",
    )
    service.bm25 = BM25Index()
    service.bm25.rebuild([])
    service.retriever = HybridRetriever(service.index, service.bm25)
    service.llm = TextLLMClient()
    service.generator = AnswerGenerator(service.llm)
    service.configured_store = "local"
    service.kb_registry = KnowledgeBaseRegistry(path=tmp_path / "knowledge_bases_local.json")
    service.kb_router = _ScriptedRouter(service.kb_registry, service.llm)
    service.query_expander = QueryExpander(service.llm)
    return service


def test_domain_isolation_and_duplicate_detection(tmp_path: Path) -> None:
    service = _build_local_service(tmp_path)

    ai_result = service.ingest_text(
        text="Transformer uses self-attention for sequence modeling.",
        title="AI Notes",
        strategy=ChunkStrategy.RECURSIVE,
    )
    law_result = service.ingest_text(
        text="Contract law requires offer, acceptance, and consideration.",
        title="Law Notes",
        strategy=ChunkStrategy.RECURSIVE,
    )
    duplicate = service.ingest_text(
        text="Transformer uses self-attention for sequence modeling.",
        title="AI Notes Copy",
        strategy=ChunkStrategy.RECURSIVE,
    )

    assert ai_result["duplicate"] is False
    assert law_result["duplicate"] is False
    assert duplicate["duplicate"] is True
    assert ai_result["knowledge_base_id"] == "kb_ai_research"
    assert law_result["knowledge_base_id"] == "kb_law"

    ai_docs = service.list_documents(knowledge_base_id="kb_ai_research")
    law_docs = service.list_documents(knowledge_base_id="kb_law")
    assert len(ai_docs) == 1
    assert len(law_docs) == 1

    ai_answer = service.ask("What does Transformer use?")
    law_answer = service.ask("What are the elements of a contract?")

    assert ai_answer["need_selection"] is False
    assert ai_answer["filters"]["knowledge_base_id"] == ["kb_ai_research"]
    assert all(
        item["metadata"].get("knowledge_base_id") == "kb_ai_research"
        for item in ai_answer["trace"]
    )
    assert all(item["metadata"].get("knowledge_base_id") == "kb_law" for item in law_answer["trace"])

    stats = service.knowledge_base_stats(knowledge_base_id="kb_ai_research")
    assert stats["documents"] == 1
    assert stats["chunks"] >= 1


def test_registry_creates_unique_ids(tmp_path: Path) -> None:
    registry = KnowledgeBaseRegistry(path=tmp_path / "kbs.json")
    first = registry.ensure(name="EMG", description="prosthesis")
    second = registry.ensure(name="Law", description="contracts")
    assert first.id != second.id
    assert len(registry.list_all()) == 2


def test_registry_paths_are_store_isolated() -> None:
    from backend.app.knowledge.registry import registry_path_for_store

    assert registry_path_for_store("local").name == "knowledge_bases_local.json"
    assert registry_path_for_store("qdrant").name == "knowledge_bases_qdrant.json"


def test_delete_knowledge_base_cascades_documents(tmp_path: Path) -> None:
    service = _build_local_service(tmp_path)
    service.ingest_text(
        text="Transformer uses self-attention for sequence modeling.",
        title="AI Notes",
        strategy=ChunkStrategy.RECURSIVE,
    )
    assert len(service.list_documents(knowledge_base_id="kb_ai_research")) == 1
    result = service.delete_knowledge_base("kb_ai_research")
    assert result["deleted"] is True
    assert result["deleted_documents"] == 1
    assert service.list_documents(knowledge_base_id="kb_ai_research") == []
    assert service.kb_registry.get("kb_ai_research") is None


def test_query_plan_retrieval_queries_dedupes() -> None:
    plan = QueryPlan(
        original="What is self-attention?",
        rewritten="Explain self-attention in Transformers",
        sub_queries=["What is self-attention?", "How does multi-head attention work?"],
        step_back="What are core attention mechanisms in neural nets?",
    )
    queries = plan.retrieval_queries()
    assert queries[0] == "What is self-attention?"
    assert "Explain self-attention in Transformers" in queries
    assert queries.count("What is self-attention?") == 1
