from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.app.indexing.chunker import ChunkStrategy


class TextIngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    source: str = "manual"
    chunk_strategy: ChunkStrategy = ChunkStrategy.HEADING
    force_reindex: bool = False
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    use_rerank: bool = True
    auto_route: bool = True
    selected_knowledge_base_ids: list[str] | None = None
    filters: dict[str, Any] | None = None


class EvaluationRequest(BaseModel):
    dataset_path: str = "datasets/golden_set.example.jsonl"
    top_k: int = Field(default=5, ge=1, le=20)
    use_rerank: bool = True
    include_generation: bool = False
    knowledge_base_ids: list[str] | None = None
    retrieval_mode: str = "hybrid"
