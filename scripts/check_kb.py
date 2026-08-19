from __future__ import annotations

from pathlib import Path
import tempfile

from backend.app.core.service import ResearchPilotService
from backend.app.indexing.chunker import ChunkStrategy
from backend.app.indexing.embeddings import HashEmbeddingProvider
from backend.app.indexing.store import ResearchIndex
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.ingestion.parser import DocumentParser
from backend.app.generation.answer import AnswerGenerator


def build_service(tmp: Path) -> ResearchPilotService:
    service = ResearchPilotService.__new__(ResearchPilotService)
    service.parser = DocumentParser()
    service.chunker = __import__("backend.app.indexing.chunker", fromlist=["Chunker"]).Chunker(
        chunk_size=200, overlap=20
    )
    service.index = ResearchIndex(HashEmbeddingProvider(32), tmp / "index.json")
    service.bm25 = BM25Index()
    service.bm25.rebuild([])
    service.retriever = HybridRetriever(service.index, service.bm25)
    service.generator = AnswerGenerator()
    return service


with tempfile.TemporaryDirectory() as tmpdir:
    service = build_service(Path(tmpdir))
    result = service.ingest_text(
        "Transformer uses self-attention.",
        title="AI",
        domain="ai_research",
        knowledge_base_id="kb_ai_research",
        strategy=ChunkStrategy.RECURSIVE,
    )
    print("ingest", result)
    print("stats", service.knowledge_base_stats(domain="ai_research", knowledge_base_id="kb_ai_research"))
    print("ask", service.ask("self-attention", domain="ai_research", knowledge_base_id="kb_ai_research")["filters"])
