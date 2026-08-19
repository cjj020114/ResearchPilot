from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from backend.app.core.models import Chunk, Document
from backend.app.indexing.embeddings import HashEmbeddingProvider
from backend.app.indexing.qdrant_store import (
    IMAGE_VECTOR_NAME,
    TEXT_VECTOR_NAME,
    QdrantResearchIndex,
)


class FakeClipProvider(HashEmbeddingProvider):
    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        return [self._embed_one(f"img::{path}") for path in image_paths]


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict[str, Any]
    vector: Any = None


class FakeQdrantClient:
    """In-memory stand-in for qdrant-client 1.18 query_points API."""

    def __init__(self) -> None:
        self.collections: dict[str, Any] = {}
        self.points: dict[str, dict[str, FakePoint]] = {}
        self.deleted_collections: list[str] = []
        self.query_calls: list[dict[str, Any]] = []

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def get_collection(self, name: str) -> Any:
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        self.deleted_collections.append(name)
        self.collections.pop(name, None)
        self.points.pop(name, None)

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.collections[collection_name] = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors_config))
        )
        self.points.setdefault(collection_name, {})

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        bucket = self.points.setdefault(collection_name, {})
        for point in points:
            bucket[str(point.id)] = FakePoint(
                id=str(point.id),
                score=0.0,
                payload=dict(point.payload or {}),
                vector=point.vector,
            )

    def scroll(
        self,
        collection_name: str,
        limit: int = 256,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[FakePoint], None]:
        del limit, offset, with_payload, with_vectors
        return list(self.points.get(collection_name, {}).values()), None

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        using: str | None = None,
        query_filter: Any = None,
        search_params: Any = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> Any:
        del query_filter, with_payload
        self.query_calls.append(
            {"using": using, "limit": limit, "search_params": search_params}
        )
        scored: list[FakePoint] = []
        for point in self.points.get(collection_name, {}).values():
            vector = point.vector
            if isinstance(vector, dict):
                if using is None or using not in vector:
                    continue
                candidate = vector[using]
            else:
                if using is not None:
                    continue
                candidate = vector
            score = sum(a * b for a, b in zip(query, candidate, strict=False))
            scored.append(
                FakePoint(id=point.id, score=float(score), payload=dict(point.payload), vector=vector)
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return SimpleNamespace(points=scored[:limit])

    def delete(self, collection_name: str, points_selector: Any) -> None:
        del collection_name, points_selector


def test_qdrant_clip_upsert_and_dual_search(tmp_path: Path, monkeypatch: Any) -> None:
    import backend.app.indexing.qdrant_store as qdrant_mod

    monkeypatch.setattr(qdrant_mod, "ClipEmbeddingProvider", FakeClipProvider)

    client = FakeQdrantClient()
    # Seed an old single-vector collection (incompatible) → should be recreated.
    from qdrant_client.http import models

    client.create_collection(
        "research_chunks",
        models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    index = QdrantResearchIndex(
        embedding_provider=FakeClipProvider(dimensions=32),
        url="http://localhost:6333",
        collection_name="research_chunks",
        vector_size=32,
        catalog_path=tmp_path / "qdrant_documents.json",
        enable_clip_multimodal=True,
        exact_search=True,
        client=client,
    )

    assert "research_chunks" in client.deleted_collections
    vectors_cfg = client.collections["research_chunks"].config.params.vectors
    assert isinstance(vectors_cfg, dict)
    assert TEXT_VECTOR_NAME in vectors_cfg and IMAGE_VECTOR_NAME in vectors_cfg

    img_file = tmp_path / "figure.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    chunk_id = str(uuid4())
    document = Document(id="doc1", title="img", source="t", text="caption", metadata={})
    chunk = Chunk(
        id=chunk_id,
        document_id="doc1",
        text="OCR: neural network diagram",
        metadata={
            "modality": "image",
            "element_type": "image",
            "image_path": str(img_file),
            "asset_id": "asset1",
        },
    )
    index.add_document(document, [chunk])

    stored = client.points["research_chunks"][chunk_id]
    assert isinstance(stored.vector, dict)
    assert TEXT_VECTOR_NAME in stored.vector
    assert IMAGE_VECTOR_NAME in stored.vector

    text_hits = index.vector_search("neural network", top_k=5)
    assert text_hits and text_hits[0].chunk.id == chunk_id
    usings = {call["using"] for call in client.query_calls}
    assert TEXT_VECTOR_NAME in usings and IMAGE_VECTOR_NAME in usings
    assert any(
        getattr(call.get("search_params"), "exact", None) is True
        for call in client.query_calls
    )

    client.query_calls.clear()
    image_hits = index.vector_search("", top_k=5, query_image_path=img_file)
    assert image_hits and image_hits[0].chunk.id == chunk_id
    assert {call["using"] for call in client.query_calls} == {
        TEXT_VECTOR_NAME,
        IMAGE_VECTOR_NAME,
    }

    stats = index.stats()
    assert stats["image_chunks"] == 1
    assert stats["clip_multimodal"] is True
