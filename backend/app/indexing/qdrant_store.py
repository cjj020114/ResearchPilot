from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from backend.app.core.models import Chunk, Document, SearchResult
from backend.app.indexing.clip_embeddings import ClipEmbeddingProvider
from backend.app.indexing.embeddings import EmbeddingProvider
from backend.app.retrieval.filters import matches_filters

TEXT_VECTOR_NAME = "text"
IMAGE_VECTOR_NAME = "image"


class QdrantResearchIndex:
    """Qdrant-backed vector index with a local document catalog.

    When ``enable_clip_multimodal`` is True, uses named vectors ``text`` + ``image``
    (方案 1A). Incompatible existing collections are dropped and recreated (option A).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        url: str,
        collection_name: str,
        vector_size: int,
        catalog_path: Path,
        enable_clip_multimodal: bool = False,
        exact_search: bool = True,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover - depends on optional runtime env
                raise RuntimeError(
                    "qdrant-client is required when VECTOR_STORE=qdrant. "
                    "Install dependencies or set VECTOR_STORE=local."
                ) from exc
            # trust_env=False: do not route localhost through system/VPN HTTP proxies
            # (httpx otherwise may return 502 for 127.0.0.1:6333).
            client = QdrantClient(url=url, trust_env=False, check_compatibility=False)

        self.embedding_provider = embedding_provider
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.catalog_path = catalog_path
        self.enable_clip_multimodal = enable_clip_multimodal
        self.exact_search = exact_search
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self._load_catalog()
        self._ensure_collection()
        self.refresh_chunks()

    def add_document(self, document: Document, chunks: list[Chunk]) -> None:
        from qdrant_client.http import models

        text_vectors = self.embedding_provider.embed([chunk.text for chunk in chunks])
        image_by_chunk: dict[str, list[float]] = {}
        if self.enable_clip_multimodal and isinstance(
            self.embedding_provider, ClipEmbeddingProvider
        ):
            image_paths: list[str] = []
            image_chunk_ids: list[str] = []
            for chunk in chunks:
                image_path = str(chunk.metadata.get("image_path") or "").strip()
                is_image = (
                    chunk.metadata.get("modality") == "image"
                    or chunk.metadata.get("element_type") == "image"
                )
                if is_image and image_path and Path(image_path).exists():
                    image_paths.append(image_path)
                    image_chunk_ids.append(chunk.id)
            if image_paths:
                for chunk_id, image_vector in zip(
                    image_chunk_ids,
                    self.embedding_provider.embed_images(image_paths),
                    strict=True,
                ):
                    image_by_chunk[chunk_id] = image_vector

        points = []
        for chunk, text_vector in zip(chunks, text_vectors, strict=True):
            if self.enable_clip_multimodal:
                named: dict[str, list[float]] = {TEXT_VECTOR_NAME: text_vector}
                maybe_image = image_by_chunk.get(chunk.id)
                if maybe_image is not None:
                    named[IMAGE_VECTOR_NAME] = maybe_image
                vector: Any = named
            else:
                vector = text_vector
            points.append(
                models.PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "document_id": chunk.document_id,
                        **chunk.metadata,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
        self.documents[document.id] = document
        self._persist_catalog()
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def delete_document(self, document_id: str) -> int:
        from qdrant_client.http import models

        deleted_count = sum(1 for chunk in self.chunks.values() if chunk.document_id == document_id)
        if deleted_count:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        self.documents.pop(document_id, None)
        self._persist_catalog()
        self.refresh_chunks()
        return deleted_count

    def find_duplicate(self, file_hash: str, knowledge_base_id: str) -> Document | None:
        for document in self.documents.values():
            if (
                document.metadata.get("file_hash") == file_hash
                and document.metadata.get("knowledge_base_id") == knowledge_base_id
            ):
                return document
        return None

    def vector_search(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
        query_image_path: str | Path | None = None,
        target: str = "auto",
    ) -> list[SearchResult]:
        """Search text/image named vectors; merge by chunk_id (keep best score)."""
        query_text_vec: list[float] | None = None
        query_image_vec: list[float] | None = None

        if query and query.strip():
            query_text_vec = self.embedding_provider.embed([query.strip()])[0]
        if (
            query_image_path
            and Path(query_image_path).exists()
            and self.enable_clip_multimodal
            and isinstance(self.embedding_provider, ClipEmbeddingProvider)
        ):
            query_image_vec = self.embedding_provider.embed_images([query_image_path])[0]

        search_text = target in {"auto", "text", "both"}
        search_image = target in {"image", "both"} or (
            target == "auto" and self.enable_clip_multimodal
        )
        if target == "auto" and query_image_vec is not None:
            search_text = True
            search_image = True
        if target == "auto" and query_image_vec is None:
            search_image = self.enable_clip_multimodal

        by_chunk: dict[str, SearchResult] = {}
        query_filter = self._to_qdrant_filter(filters)

        def _consider_points(points: list[Any]) -> None:
            for point in points:
                chunk_id = str(point.id)
                chunk = self.chunks.get(chunk_id)
                if chunk is None:
                    payload = dict(point.payload or {})
                    chunk = self._chunk_from_payload(chunk_id, payload)
                if not matches_filters(chunk, filters):
                    continue
                score = float(point.score)
                existing = by_chunk.get(chunk_id)
                if existing is None or score > existing.score:
                    by_chunk[chunk_id] = SearchResult(
                        chunk=chunk,
                        score=score,
                        vector_score=score,
                    )

        def _query(vector: list[float], using: str | None) -> None:
            from qdrant_client.http import models

            # exact=True ≈ FLAT full scan (aligns with local ResearchIndex brute-force).
            search_params = (
                models.SearchParams(exact=True) if self.exact_search else None
            )
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                using=using,
                query_filter=query_filter,
                search_params=search_params,
                limit=top_k,
                with_payload=True,
            )
            _consider_points(list(getattr(response, "points", response) or []))

        if not self.enable_clip_multimodal:
            if query_text_vec is not None:
                _query(query_text_vec, None)
            return sorted(by_chunk.values(), key=lambda result: result.score, reverse=True)[:top_k]

        if query_text_vec is not None:
            if search_text:
                _query(query_text_vec, TEXT_VECTOR_NAME)
            if search_image:
                _query(query_text_vec, IMAGE_VECTOR_NAME)
        if query_image_vec is not None:
            if search_text:
                _query(query_image_vec, TEXT_VECTOR_NAME)
            if search_image:
                _query(query_image_vec, IMAGE_VECTOR_NAME)

        return sorted(by_chunk.values(), key=lambda result: result.score, reverse=True)[:top_k]

    def refresh_chunks(self) -> None:
        chunks: dict[str, Chunk] = {}
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                chunk = self._chunk_from_payload(str(point.id), payload)
                chunks[chunk.id] = chunk
            if offset is None:
                break
        self.chunks = chunks

    def stats(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        chunks = [chunk for chunk in self.chunks.values() if matches_filters(chunk, filters)]
        document_ids = {chunk.document_id for chunk in chunks}
        domains = sorted(
            {str(chunk.metadata.get("domain")) for chunk in chunks if chunk.metadata.get("domain")}
        )
        knowledge_base_ids = sorted(
            {
                str(chunk.metadata.get("knowledge_base_id"))
                for chunk in chunks
                if chunk.metadata.get("knowledge_base_id")
            }
        )
        image_chunks = sum(
            1
            for chunk in chunks
            if chunk.metadata.get("modality") == "image"
            or chunk.metadata.get("element_type") == "image"
        )
        return {
            "store": "qdrant",
            "collection": self.collection_name,
            "documents": len(document_ids),
            "chunks": len(chunks),
            "image_chunks": image_chunks,
            "clip_multimodal": self.enable_clip_multimodal,
            "domains": domains,
            "knowledge_base_ids": knowledge_base_ids,
        }

    def _desired_vectors_config(self) -> Any:
        from qdrant_client.http import models

        params = models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
        if self.enable_clip_multimodal:
            return {
                TEXT_VECTOR_NAME: params,
                IMAGE_VECTOR_NAME: models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
            }
        return params

    def _collection_matches_desired(self) -> bool:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if self.enable_clip_multimodal:
            if not isinstance(vectors, dict):
                return False
            text_params = vectors.get(TEXT_VECTOR_NAME)
            image_params = vectors.get(IMAGE_VECTOR_NAME)
            if text_params is None or image_params is None:
                return False
            return (
                int(getattr(text_params, "size", -1)) == self.vector_size
                and int(getattr(image_params, "size", -1)) == self.vector_size
            )
        if isinstance(vectors, dict):
            return False
        return int(getattr(vectors, "size", -1)) == self.vector_size

    def _ensure_collection(self) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and self._collection_matches_desired():
            return
        if exists:
            # Option A: drop incompatible schema (single-vector / wrong dims) and recreate.
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._desired_vectors_config(),
        )

    def _to_qdrant_filter(self, filters: dict[str, Any] | None) -> Any:
        from qdrant_client.http import models

        if not filters:
            return None
        conditions: list[Any] = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, list):
                conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
            else:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=conditions) if conditions else None

    def _chunk_from_payload(self, chunk_id: str, payload: dict[str, Any]) -> Chunk:
        text = str(payload.pop("text", ""))
        document_id = str(payload.get("document_id", ""))
        payload.pop("document_id", None)
        return Chunk(id=chunk_id, document_id=document_id, text=text, metadata=payload)

    def _load_catalog(self) -> None:
        if not self.catalog_path.exists():
            return
        payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.documents = {
            str(item["id"]): Document.from_dict(cast(dict[str, Any], item))
            for item in payload.get("documents", [])
        }

    def _persist_catalog(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": [document.to_dict() for document in self.documents.values()]}
        self.catalog_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
