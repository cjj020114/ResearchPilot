from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

from backend.app.core.config import settings
from backend.app.core.models import Document, SearchResult
from backend.app.generation.answer import AnswerGenerator
from backend.app.indexing.chunker import ChunkStrategy, Chunker
from backend.app.indexing.embeddings import build_embedding_provider
from backend.app.indexing.qdrant_store import QdrantResearchIndex
from backend.app.indexing.store import ResearchIndex
from backend.app.ingestion.parser import DocumentParser
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.knowledge.llm_client import TextLLMClient
from backend.app.knowledge.registry import KnowledgeBaseRegistry
from backend.app.knowledge.router import KnowledgeLLMRouter
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.hybrid import HybridRetriever, RetrievalMode
from backend.app.retrieval.query_expansion import QueryExpander, QueryPlan
from backend.app.retrieval.reranker import build_reranker


class ResearchPilotService:
    def __init__(self) -> None:
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        if settings.enable_clip_multimodal:
            from backend.app.indexing.clip_embeddings import build_clip_embedding_provider

            embedding_provider = build_clip_embedding_provider(
                settings.clip_text_model,
                settings.clip_image_model,
                offline=False,
            )
        else:
            embedding_provider = build_embedding_provider(settings.embedding_model, offline=False)
        self.parser = DocumentParser()
        self.pipeline = IngestionPipeline()
        self.chunker = Chunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        self.configured_store = settings.vector_store.strip().lower()
        self.index: Any
        if self.configured_store == "qdrant":
            # Fail loudly — silent fallback to local made VECTOR_STORE=qdrant look "working"
            # while still serving index.json.
            self.index = QdrantResearchIndex(
                embedding_provider=embedding_provider,
                url=settings.qdrant_url,
                collection_name=settings.qdrant_collection,
                vector_size=settings.embedding_dimensions,
                catalog_path=settings.storage_dir / "qdrant_documents.json",
                enable_clip_multimodal=settings.enable_clip_multimodal,
                exact_search=settings.qdrant_exact_search,
            )
        else:
            self.index = ResearchIndex(
                embedding_provider=embedding_provider,
                storage_path=settings.storage_dir / "index.json",
            )
        self.bm25 = BM25Index()
        self.bm25.rebuild(list(self.index.chunks.values()))
        self.retriever = HybridRetriever(self.index, self.bm25, reranker=build_reranker())
        # LLM_*: answer generation. ROUTER_LLM_*: assign / route / rewrite (falls back to LLM_*).
        self.llm = TextLLMClient.for_answer()
        self.router_llm = TextLLMClient.for_router()
        self.generator = AnswerGenerator(self.llm)
        # Separate registry files so local KB names never appear under qdrant and vice versa.
        self.kb_registry = KnowledgeBaseRegistry()
        self.kb_router = KnowledgeLLMRouter(self.kb_registry, self.router_llm)
        self.query_expander = QueryExpander(self.router_llm)
        self._sync_registry_from_index()

    def ingest_file(
        self,
        path: Path,
        title: str | None = None,
        strategy: ChunkStrategy = ChunkStrategy.HEADING,
        domain: str | None = None,
        knowledge_base_id: str | None = None,
        force_reindex: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        force_loader: str | None = None,
        force_knowledge_base_id: str | None = None,
        force_knowledge_base_name: str | None = None,
        force_knowledge_base_description: str | None = None,
    ) -> dict[str, Any]:
        del domain, knowledge_base_id  # upload assignment is LLM-driven unless force_* is set
        if settings.doc_router_enabled:
            document, route = self.pipeline.ingest_file(path, title=title, force_loader=force_loader)
            route_info = self.pipeline.describe_route(route)
            route_info["actual_parser"] = document.metadata.get("actual_parser", route.loader)
            route_info["loader_status"] = document.metadata.get("loader_status", "ready")
            route_info["vlm_calls"] = document.metadata.get("vlm_calls")
            route_info["element_count"] = (
                len(document.elements) if document.elements is not None else None
            )
        else:
            document = self.parser.parse_file(path, title=title)
            route_info = {
                "loader": "legacy_parser",
                "reason": "DOC_ROUTER_ENABLED=false",
                "actual_parser": "legacy_parser",
                "loader_status": "ready",
                "pdf_mode": None,
                "vlm_calls": None,
                "element_count": len(document.elements) if document.elements else None,
            }
        file_hash = self._hash_bytes(path.read_bytes())
        result = self._index_document(
            document,
            strategy=strategy,
            file_hash=file_hash,
            force_reindex=force_reindex,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            filename=path.name,
            force_knowledge_base_id=force_knowledge_base_id,
            force_knowledge_base_name=force_knowledge_base_name,
            force_knowledge_base_description=force_knowledge_base_description,
        )
        result["route"] = route_info
        return result

    def ingest_text(
        self,
        text: str,
        title: str,
        source: str = "manual",
        strategy: ChunkStrategy = ChunkStrategy.HEADING,
        domain: str | None = None,
        knowledge_base_id: str | None = None,
        force_reindex: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        force_knowledge_base_id: str | None = None,
        force_knowledge_base_name: str | None = None,
        force_knowledge_base_description: str | None = None,
    ) -> dict[str, Any]:
        del domain, knowledge_base_id
        document = self.parser.parse_text(text, title=title, source=source)
        file_hash = self._hash_text(text)
        return self._index_document(
            document,
            strategy=strategy,
            file_hash=file_hash,
            force_reindex=force_reindex,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            filename=source,
            force_knowledge_base_id=force_knowledge_base_id,
            force_knowledge_base_name=force_knowledge_base_name,
            force_knowledge_base_description=force_knowledge_base_description,
        )

    def ask(
        self,
        question: str,
        top_k: int = 6,
        filters: dict[str, Any] | None = None,
        use_rerank: bool = True,
        domain: str | None = None,
        knowledge_base_id: str | None = None,
        selected_knowledge_base_ids: list[str] | None = None,
        auto_route: bool = True,
        query_image_path: str | None = None,
    ) -> dict[str, Any]:
        del domain, knowledge_base_id
        user_question = (question or "").strip()
        has_image = bool(query_image_path and Path(query_image_path).exists())
        image_only = has_image and not user_question
        # Keep user text when provided. Pure-image queries stay empty (no placeholder question).
        question = user_question

        route_info: dict[str, Any]
        kb_ids: list[str]
        all_kb_ids = [item.id for item in self.kb_registry.list_all()]

        if selected_knowledge_base_ids:
            kb_ids = [
                kb_id
                for kb_id in selected_knowledge_base_ids
                if self.kb_registry.get(kb_id) is not None
            ]
            route_info = {
                "need_selection": False,
                "knowledge_base_ids": kb_ids,
                "reason": "user selected knowledge bases",
                "confidence": 1.0,
                "used_llm": False,
                "candidates": self.kb_registry.catalog_for_prompt(),
                "message": None,
                "has_query_image": has_image,
            }
            if not kb_ids:
                return {
                    "answer": "",
                    "citations": [],
                    "confidence": "low",
                    "question": question,
                    "need_selection": True,
                    "message": "请选择知识库",
                    "routing": {
                        **route_info,
                        "need_selection": True,
                        "message": "请选择知识库（所选 ID 无效）",
                        "candidates": self.kb_registry.catalog_for_prompt(),
                    },
                    "retrieved": 0,
                    "trace": [],
                    "filters": filters or {},
                }
        elif auto_route:
            # C1: image-only → skip routing (all KBs).
            # text+image → route by text; on uncertainty fall back to all KBs (do not block).
            # text-only → existing routing (may ask user to select).
            if has_image and not user_question:
                kb_ids = all_kb_ids
                route_info = {
                    "need_selection": False,
                    "knowledge_base_ids": kb_ids,
                    "reason": "image-only query; skipped KB routing (search all)",
                    "confidence": 1.0,
                    "used_llm": False,
                    "candidates": self.kb_registry.catalog_for_prompt(),
                    "message": None,
                    "has_query_image": True,
                    "routing_mode": "image_only_all_kbs",
                }
            elif has_image and user_question:
                routed = self.kb_router.route_query(
                    user_question, top_n=settings.kb_route_top_n
                )
                if routed.need_selection or not routed.knowledge_base_ids:
                    kb_ids = all_kb_ids
                    route_info = {
                        "need_selection": False,
                        "knowledge_base_ids": kb_ids,
                        "reason": (
                            "text+image routing uncertain; fallback to all knowledge bases. "
                            f"router_said: {routed.reason}"
                        ),
                        "confidence": routed.confidence,
                        "used_llm": routed.used_llm,
                        "candidates": routed.candidates,
                        "message": None,
                        "has_query_image": True,
                        "routing_mode": "text_image_fallback_all_kbs",
                        "router_need_selection": True,
                        "router_message": routed.message,
                    }
                else:
                    kb_ids = routed.knowledge_base_ids
                    route_info = {
                        "need_selection": False,
                        "knowledge_base_ids": kb_ids,
                        "reason": routed.reason,
                        "confidence": routed.confidence,
                        "used_llm": routed.used_llm,
                        "candidates": routed.candidates,
                        "message": None,
                        "has_query_image": True,
                        "routing_mode": "text_image_routed",
                    }
            else:
                routed = self.kb_router.route_query(question, top_n=settings.kb_route_top_n)
                route_info = {
                    "need_selection": routed.need_selection,
                    "knowledge_base_ids": routed.knowledge_base_ids,
                    "reason": routed.reason,
                    "confidence": routed.confidence,
                    "used_llm": routed.used_llm,
                    "candidates": routed.candidates,
                    "message": routed.message,
                    "has_query_image": False,
                    "routing_mode": "text_only",
                }
                if routed.need_selection:
                    return {
                        "answer": "",
                        "citations": [],
                        "confidence": "low",
                        "question": question,
                        "need_selection": True,
                        "message": routed.message or "请选择知识库",
                        "routing": route_info,
                        "retrieved": 0,
                        "trace": [],
                        "filters": filters or {},
                    }
                kb_ids = routed.knowledge_base_ids
        else:
            kb_ids = all_kb_ids
            route_info = {
                "need_selection": False,
                "knowledge_base_ids": kb_ids,
                "reason": "auto_route disabled; searching all knowledge bases",
                "confidence": 1.0,
                "used_llm": False,
                "candidates": self.kb_registry.catalog_for_prompt(),
                "message": None,
                "has_query_image": has_image,
                "routing_mode": "auto_route_disabled",
            }

        scoped_filters = {**(filters or {}), "knowledge_base_id": kb_ids}
        candidate_k = settings.retrieval_candidate_k
        low_threshold = settings.retrieval_low_confidence_threshold
        retried = False
        retry_candidate_k = settings.retrieval_retry_candidate_k

        if image_only:
            # Pure image: CLIP image tower only — no placeholder text, expand, or text rerank.
            query_plan = QueryPlan(
                original="",
                used_llm=False,
                reason="image-only query; skipped text expansion (dense image retrieval only)",
            )
            retrieval_queries = [""]
            effective_rerank = False
        else:
            query_plan = self.query_expander.expand(
                question, max_sub_queries=settings.query_max_sub_queries
            )
            retrieval_queries = query_plan.retrieval_queries()
            effective_rerank = use_rerank

        results = self.retriever.search_multi(
            retrieval_queries,
            top_k=top_k,
            candidate_k=candidate_k,
            filters=scoped_filters,
            use_rerank=effective_rerank,
            query_image_path=query_image_path,
        )
        best_score = self.generator.best_score(results)
        if best_score < low_threshold:
            retried = True
            retry_results = self.retriever.search_multi(
                retrieval_queries,
                top_k=top_k,
                candidate_k=retry_candidate_k,
                filters=scoped_filters,
                use_rerank=effective_rerank,
                query_image_path=query_image_path,
            )
            # Keep best score per chunk across first + retry.
            by_id = {item.chunk.id: item for item in results}
            for item in retry_results:
                existing = by_id.get(item.chunk.id)
                if existing is None or item.score > existing.score:
                    by_id[item.chunk.id] = item
            results = sorted(by_id.values(), key=lambda item: item.score, reverse=True)[:top_k]
            best_score = self.generator.best_score(results)

        response = self.generator.generate(
            question,
            results,
            query_image_path=query_image_path if has_image else None,
        )
        response["confidence"] = self.generator.confidence_label(results, low_threshold)
        return {
            **response,
            "question": question or ("（图片查询）" if image_only else question),
            "query_image_path": query_image_path,
            "clip_multimodal": settings.enable_clip_multimodal,
            "filters": scoped_filters,
            "need_selection": False,
            "routing": route_info,
            "query_plan": query_plan.to_dict(),
            "retrieval_retry": {
                "triggered": retried,
                "threshold": low_threshold,
                "best_score": round(best_score, 4),
                "candidate_k_first": candidate_k,
                "candidate_k_retry": retry_candidate_k if retried else None,
                "image_only_dense": image_only,
            },
            "retrieved": len(results),
            "trace": [
                {
                    "chunk_id": result.chunk.id,
                    "score": result.score,
                    "vector_score": result.vector_score,
                    "bm25_score": result.bm25_score,
                    "rerank_score": result.rerank_score,
                    "modality": result.chunk.metadata.get("modality"),
                    "image_path": result.chunk.metadata.get("image_path"),
                    "metadata": result.chunk.metadata,
                }
                for result in results
            ],
        }

    def retrieve_for_evaluation(
        self,
        question: str,
        *,
        top_k: int = 5,
        use_rerank: bool = True,
        retrieval_mode: RetrievalMode = "hybrid",
        knowledge_base_ids: list[str] | None = None,
        auto_route: bool = False,
        use_query_expansion: bool = False,
        use_retrieval_retry: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieval-only path with optional routing / expansion / retry for ablations."""
        route_info: dict[str, Any]
        kb_ids: list[str]

        if auto_route:
            routed = self.kb_router.route_query(question, top_n=settings.kb_route_top_n)
            route_info = {
                "need_selection": routed.need_selection,
                "knowledge_base_ids": routed.knowledge_base_ids,
                "reason": routed.reason,
                "confidence": routed.confidence,
                "used_llm": routed.used_llm,
                "candidates": routed.candidates,
                "message": routed.message,
            }
            if routed.need_selection or not routed.knowledge_base_ids:
                # Ablation fallback: avoid empty retrieval; search all KBs.
                kb_ids = [item.id for item in self.kb_registry.list_all()]
                route_info["fallback_to_all"] = True
            else:
                kb_ids = routed.knowledge_base_ids
        elif knowledge_base_ids:
            kb_ids = [
                kb_id for kb_id in knowledge_base_ids if self.kb_registry.get(kb_id) is not None
            ]
            route_info = {
                "need_selection": False,
                "knowledge_base_ids": kb_ids,
                "reason": "fixed knowledge_base_ids for evaluation",
                "confidence": 1.0,
                "used_llm": False,
                "candidates": self.kb_registry.catalog_for_prompt(),
                "message": None,
            }
        else:
            kb_ids = [item.id for item in self.kb_registry.list_all()]
            route_info = {
                "need_selection": False,
                "knowledge_base_ids": kb_ids,
                "reason": "no routing; searching all knowledge bases",
                "confidence": 1.0,
                "used_llm": False,
                "candidates": self.kb_registry.catalog_for_prompt(),
                "message": None,
            }

        scoped_filters = {**(filters or {}), "knowledge_base_id": kb_ids}
        if use_query_expansion:
            query_plan = self.query_expander.expand(
                question, max_sub_queries=settings.query_max_sub_queries
            )
            retrieval_queries = query_plan.retrieval_queries()
        else:
            from backend.app.retrieval.query_expansion import QueryPlan

            query_plan = QueryPlan(
                original=question,
                rewritten=question,
                used_llm=False,
                reason="query expansion disabled",
            )
            retrieval_queries = [question]

        candidate_k = settings.retrieval_candidate_k
        results: list[SearchResult] = self.retriever.search_multi(
            retrieval_queries,
            top_k=top_k,
            candidate_k=candidate_k,
            filters=scoped_filters,
            use_rerank=use_rerank,
            mode=retrieval_mode,
        )
        best_score = self.generator.best_score(results)
        retried = False
        retry_candidate_k = settings.retrieval_retry_candidate_k
        low_threshold = settings.retrieval_low_confidence_threshold
        if use_retrieval_retry and best_score < low_threshold:
            retried = True
            retry_results = self.retriever.search_multi(
                retrieval_queries,
                top_k=top_k,
                candidate_k=retry_candidate_k,
                filters=scoped_filters,
                use_rerank=use_rerank,
                mode=retrieval_mode,
            )
            by_id = {item.chunk.id: item for item in results}
            for item in retry_results:
                existing = by_id.get(item.chunk.id)
                if existing is None or item.score > existing.score:
                    by_id[item.chunk.id] = item
            results = sorted(by_id.values(), key=lambda item: item.score, reverse=True)[:top_k]
            best_score = self.generator.best_score(results)

        return {
            "question": question,
            "results": results,
            "filters": scoped_filters,
            "routing": route_info,
            "query_plan": query_plan.to_dict(),
            "retrieval_retry": {
                "triggered": retried,
                "threshold": low_threshold,
                "best_score": round(best_score, 4),
                "candidate_k_first": candidate_k,
                "candidate_k_retry": retry_candidate_k if retried else None,
            },
            "retrieved": len(results),
        }

    def list_documents(
        self,
        domain: str | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": document.id,
                "title": document.title,
                "source": document.source,
                "metadata": document.metadata,
                "created_at": document.created_at,
                "chunk_count": sum(
                    1 for chunk in self.index.chunks.values() if chunk.document_id == document.id
                ),
            }
            for document in self.index.documents.values()
            if self._document_matches_scope(document.metadata, domain, knowledge_base_id)
        ]

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.kb_registry.list_all()]

    def delete_document(self, document_id: str) -> dict[str, Any]:
        deleted_chunks = self.index.delete_document(document_id)
        self.bm25.rebuild(list(self.index.chunks.values()))
        return {"document_id": document_id, "deleted_chunks": deleted_chunks}

    def delete_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any]:
        existing = self.kb_registry.get(knowledge_base_id)
        if existing is None:
            return {
                "knowledge_base_id": knowledge_base_id,
                "deleted": False,
                "message": "knowledge base not found",
                "deleted_documents": 0,
                "deleted_chunks": 0,
            }
        doc_ids = [
            document.id
            for document in self.index.documents.values()
            if document.metadata.get("knowledge_base_id") == knowledge_base_id
        ]
        deleted_chunks = 0
        for doc_id in doc_ids:
            deleted_chunks += self.index.delete_document(doc_id)
        removed = self.kb_registry.delete(knowledge_base_id)
        self.bm25.rebuild(list(self.index.chunks.values()))
        return {
            "knowledge_base_id": knowledge_base_id,
            "name": existing.name,
            "deleted": removed,
            "deleted_documents": len(doc_ids),
            "deleted_chunks": deleted_chunks,
        }

    def knowledge_base_stats(
        self,
        domain: str | None = None,
        knowledge_base_id: str | None = None,
    ) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if domain:
            filters["domain"] = domain
        if knowledge_base_id:
            filters["knowledge_base_id"] = knowledge_base_id
        stats = cast(dict[str, Any], self.index.stats(filters=filters or None))
        stats["knowledge_bases"] = self.list_knowledge_bases()
        stats["configured_store"] = self.configured_store
        stats["registry_path"] = str(self.kb_registry.path)
        return stats

    def reindex_existing_documents(
        self,
        strategy: ChunkStrategy = ChunkStrategy.HEADING,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """B3: rebuild vectors from source files that still exist; skip + report the rest."""
        documents = list(self.index.documents.values())
        reindexed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        for document in documents:
            source = str(document.source or "").strip()
            path = Path(source) if source else None
            if path is None or not path.is_file():
                alt = settings.upload_dir / Path(source).name if source else None
                if alt is not None and alt.is_file():
                    path = alt
                else:
                    skipped.append(
                        {
                            "document_id": document.id,
                            "title": document.title,
                            "source": source,
                            "reason": "source file not found",
                        }
                    )
                    continue

            kb_id = document.metadata.get("knowledge_base_id")
            try:
                result = self.ingest_file(
                    path,
                    title=document.title,
                    strategy=strategy,
                    force_reindex=True,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_knowledge_base_id=str(kb_id) if kb_id else None,
                    force_knowledge_base_name=str(
                        document.metadata.get("knowledge_base_name") or ""
                    )
                    or None,
                    force_knowledge_base_description=str(
                        document.metadata.get("knowledge_base_description") or ""
                    )
                    or None,
                )
                reindexed.append(
                    {
                        "document_id": result.get("document_id") or document.id,
                        "title": document.title,
                        "source": str(path),
                        "knowledge_base_id": result.get("knowledge_base_id") or kb_id,
                        "chunk_count": result.get("chunk_count"),
                        "duplicate": result.get("duplicate"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - collect per-doc failures for report
                skipped.append(
                    {
                        "document_id": document.id,
                        "title": document.title,
                        "source": str(path),
                        "reason": str(exc),
                    }
                )

        stats = self.knowledge_base_stats()
        return {
            "reindexed_count": len(reindexed),
            "skipped_count": len(skipped),
            "reindexed": reindexed,
            "skipped": skipped,
            "stats": stats,
            "clip_multimodal": settings.enable_clip_multimodal,
        }

    def _index_document(
        self,
        document: Document,
        strategy: ChunkStrategy,
        file_hash: str,
        force_reindex: bool,
        chunk_size: int | None,
        chunk_overlap: int | None,
        filename: str = "",
        force_knowledge_base_id: str | None = None,
        force_knowledge_base_name: str | None = None,
        force_knowledge_base_description: str | None = None,
    ) -> dict[str, Any]:
        if force_knowledge_base_id:
            kb = self.kb_registry.ensure(
                name=force_knowledge_base_name or force_knowledge_base_id,
                description=force_knowledge_base_description or "",
                knowledge_base_id=force_knowledge_base_id,
            )
            from backend.app.knowledge.router import AssignmentResult

            assignment = AssignmentResult(
                knowledge_base=kb,
                created_new=False,
                reason=f"forced knowledge base {kb.id}",
                confidence=1.0,
                used_llm=False,
            )
        else:
            assignment = self.kb_router.assign_document(
                title=document.title,
                snippet=document.text[:3000],
                filename=filename or Path(document.source).name,
            )
            kb = assignment.knowledge_base
        scope = {
            "domain": kb.name,
            "knowledge_base_id": kb.id,
            "knowledge_base_name": kb.name,
            "knowledge_base_description": kb.description,
        }
        duplicate = self.index.find_duplicate(file_hash, scope["knowledge_base_id"])
        if duplicate and not force_reindex:
            return {
                "document_id": duplicate.id,
                "title": duplicate.title,
                "duplicate": True,
                "message": "同一知识库中已存在相同内容的文档，未重复索引。",
                "domain": scope["domain"],
                "knowledge_base_id": scope["knowledge_base_id"],
                "knowledge_base_assignment": {
                    "id": kb.id,
                    "name": kb.name,
                    "description": kb.description,
                    "created_new": assignment.created_new,
                    "reason": assignment.reason,
                    "confidence": assignment.confidence,
                    "used_llm": assignment.used_llm,
                },
                "stats": self.knowledge_base_stats(knowledge_base_id=kb.id),
            }
        if duplicate and force_reindex:
            self.delete_document(duplicate.id)

        document.metadata.update({**scope, "file_hash": file_hash})
        active_chunker = Chunker(
            chunk_size=chunk_size or settings.chunk_size,
            overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        )
        chunks = active_chunker.chunk(document, strategy=strategy)
        self.index.add_document(document, chunks)
        self.bm25.rebuild(list(self.index.chunks.values()))
        return {
            "document_id": document.id,
            "title": document.title,
            "chunk_count": len(chunks),
            "strategy": strategy.value,
            "duplicate": False,
            "domain": scope["domain"],
            "knowledge_base_id": scope["knowledge_base_id"],
            "knowledge_base_assignment": {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "created_new": assignment.created_new,
                "reason": assignment.reason,
                "confidence": assignment.confidence,
                "used_llm": assignment.used_llm,
            },
            "stats": self.knowledge_base_stats(knowledge_base_id=kb.id),
        }

    def _sync_registry_from_index(self) -> None:
        for document in self.index.documents.values():
            self.kb_registry.upsert_from_document_metadata(document.metadata)

    def _document_matches_scope(
        self,
        metadata: dict[str, Any],
        domain: str | None,
        knowledge_base_id: str | None,
    ) -> bool:
        if domain and metadata.get("domain") != domain:
            return False
        if knowledge_base_id and metadata.get("knowledge_base_id") != knowledge_base_id:
            return False
        return True

    def _hash_bytes(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _hash_text(self, text: str) -> str:
        return self._hash_bytes(text.encode("utf-8"))


service = ResearchPilotService()
