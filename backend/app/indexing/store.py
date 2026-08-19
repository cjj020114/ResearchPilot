from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from backend.app.core.models import Chunk, Document, SearchResult
from backend.app.indexing.clip_embeddings import ClipEmbeddingProvider
from backend.app.indexing.embeddings import EmbeddingProvider
from backend.app.retrieval.filters import matches_filters


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_norm * right_norm)


class ResearchIndex:
    def __init__(self, embedding_provider: EmbeddingProvider, storage_path: Path | None = None) -> None:
        self.embedding_provider = embedding_provider
        self.storage_path = storage_path
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, list[float]] = {}  # text_vec (CLIP text tower when multimodal)
        self.image_vectors: dict[str, list[float]] = {}  # image_vec (CLIP image tower)
        if storage_path and storage_path.exists():
            self.load(storage_path)

    def add_document(self, document: Document, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        text_vectors = self.embedding_provider.embed(texts)
        self.documents[document.id] = document
        image_paths: list[str] = []
        image_chunk_ids: list[str] = []
        for chunk, vector in zip(chunks, text_vectors, strict=True):
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = vector
            image_path = str(chunk.metadata.get("image_path") or "").strip()
            is_image = (
                chunk.metadata.get("modality") == "image"
                or chunk.metadata.get("element_type") == "image"
            )
            if is_image and image_path and Path(image_path).exists():
                image_paths.append(image_path)
                image_chunk_ids.append(chunk.id)

        if image_paths and isinstance(self.embedding_provider, ClipEmbeddingProvider):
            image_vectors = self.embedding_provider.embed_images(image_paths)
            for chunk_id, image_vector in zip(image_chunk_ids, image_vectors, strict=True):
                self.image_vectors[chunk_id] = image_vector
        self.persist()

    def delete_document(self, document_id: str) -> int:
        self.documents.pop(document_id, None)
        chunk_ids = [chunk_id for chunk_id, chunk in self.chunks.items() if chunk.document_id == document_id]
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)
            self.vectors.pop(chunk_id, None)
            self.image_vectors.pop(chunk_id, None)
        self.persist()
        return len(chunk_ids)

    def find_duplicate(self, file_hash: str, knowledge_base_id: str) -> Document | None:
        for document in self.documents.values():
            if (
                document.metadata.get("file_hash") == file_hash
                and document.metadata.get("knowledge_base_id") == knowledge_base_id
            ):
                return document
        return None

    def stats(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        documents = [
            document
            for document in self.documents.values()
            if self._metadata_matches(document.metadata, filters)
        ]
        chunks = [
            chunk
            for chunk in self.chunks.values()
            if self._matches_filters(chunk, filters)
        ]
        domains = sorted(
            {
                str(chunk.metadata.get("domain"))
                for chunk in chunks
                if chunk.metadata.get("domain") is not None
            }
        )
        knowledge_base_ids = sorted(
            {
                str(chunk.metadata.get("knowledge_base_id"))
                for chunk in chunks
                if chunk.metadata.get("knowledge_base_id") is not None
            }
        )
        return {
            "store": "local",
            "documents": len(documents),
            "chunks": len(chunks),
            "image_chunks": len(self.image_vectors),
            "domains": domains,
            "knowledge_base_ids": knowledge_base_ids,
        }

    def vector_search(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
        query_image_path: str | Path | None = None,
        target: str = "auto",
    ) -> list[SearchResult]:
        """Search text_vec and/or image_vec; merge by chunk_id (keep best score).

        target: auto | text | image | both
        - auto: image query -> both; text query -> text (and image if CLIP enabled)
        """
        query_text_vec: list[float] | None = None
        query_image_vec: list[float] | None = None

        if query and query.strip():
            query_text_vec = self.embedding_provider.embed([query.strip()])[0]
        if query_image_path and Path(query_image_path).exists():
            if isinstance(self.embedding_provider, ClipEmbeddingProvider):
                query_image_vec = self.embedding_provider.embed_images([query_image_path])[0]

        search_text = target in {"auto", "text", "both"}
        search_image = target in {"image", "both"} or (
            target == "auto" and isinstance(self.embedding_provider, ClipEmbeddingProvider)
        )
        if target == "auto" and query_image_vec is not None:
            search_text = True
            search_image = True
        if target == "auto" and query_image_vec is None:
            # Text query in CLIP space can also hit image_vec (文搜图).
            search_image = isinstance(self.embedding_provider, ClipEmbeddingProvider)

        by_chunk: dict[str, SearchResult] = {}

        def _consider(chunk_id: str, score: float) -> None:
            chunk = self.chunks.get(chunk_id)
            if chunk is None or not matches_filters(chunk, filters):
                return
            existing = by_chunk.get(chunk_id)
            if existing is None or score > existing.score:
                by_chunk[chunk_id] = SearchResult(
                    chunk=chunk,
                    score=score,
                    vector_score=score,
                )

        # Query text tower against text_vec / image_vec
        if query_text_vec is not None:
            if search_text:
                for chunk_id, vector in self.vectors.items():
                    _consider(chunk_id, cosine_similarity(query_text_vec, vector))
            if search_image:
                for chunk_id, vector in self.image_vectors.items():
                    _consider(chunk_id, cosine_similarity(query_text_vec, vector))

        # Query image tower against text_vec / image_vec (图搜文 / 图搜图)
        if query_image_vec is not None:
            if search_text:
                for chunk_id, vector in self.vectors.items():
                    _consider(chunk_id, cosine_similarity(query_image_vec, vector))
            if search_image:
                for chunk_id, vector in self.image_vectors.items():
                    _consider(chunk_id, cosine_similarity(query_image_vec, vector))

        return sorted(by_chunk.values(), key=lambda result: result.score, reverse=True)[:top_k]

    def persist(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [document.to_dict() for document in self.documents.values()],
            "chunks": [chunk.__dict__ for chunk in self.chunks.values()],
            "vectors": self.vectors,
            "image_vectors": self.image_vectors,
        }
        self.storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.documents = {
            item["id"]: Document.from_dict(item) for item in payload.get("documents", [])
        }
        self.chunks = {item["id"]: Chunk(**item) for item in payload.get("chunks", [])}
        self.vectors = payload.get("vectors", {})
        self.image_vectors = payload.get("image_vectors", {})

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any] | None) -> bool:
        return matches_filters(chunk, filters)

    def _metadata_matches(self, metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if expected is None:
                continue
            if isinstance(expected, list):
                if metadata.get(key) not in expected:
                    return False
                continue
            if metadata.get(key) != expected:
                return False
        return True
