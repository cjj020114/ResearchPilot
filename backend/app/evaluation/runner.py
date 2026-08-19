from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from backend.app.evaluation.metrics import (
    aggregate,
    context_precision,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from backend.app.retrieval.hybrid import HybridRetriever, RetrievalMode


class _Askable(Protocol):
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
    ) -> dict[str, Any]: ...


class _EvalRetrievable(Protocol):
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
    ) -> dict[str, Any]: ...


class EvaluationRunner:
    def __init__(
        self,
        retriever: HybridRetriever,
        ask_service: _Askable | None = None,
        eval_service: _EvalRetrievable | None = None,
    ) -> None:
        self.retriever = retriever
        self.ask_service = ask_service
        self.eval_service = eval_service

    def run(
        self,
        dataset_path: Path,
        top_k: int = 5,
        use_rerank: bool = True,
        include_generation: bool = False,
        knowledge_base_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_mode: RetrievalMode = "hybrid",
        auto_route: bool = False,
        use_query_expansion: bool = False,
        use_retrieval_retry: bool = False,
        search_all_knowledge_bases: bool = False,
    ) -> dict[str, Any]:
        examples = self._load_jsonl(dataset_path)
        recalls: list[float] = []
        hits: list[float] = []
        reciprocal_ranks: list[float] = []
        precisions: list[float] = []
        rows: list[dict[str, Any]] = []
        use_pipeline = self.eval_service is not None and (
            auto_route
            or use_query_expansion
            or use_retrieval_retry
            or search_all_knowledge_bases
        )

        scoped_filters = dict(filters or {})
        if knowledge_base_ids and not use_pipeline:
            scoped_filters["knowledge_base_id"] = knowledge_base_ids

        for example in examples:
            question = str(example["question"])
            relevant_ids = [str(item) for item in example.get("relevant_chunk_ids", [])]
            route_info: dict[str, Any] | None = None
            query_plan: dict[str, Any] | None = None
            retry_info: dict[str, Any] | None = None

            if use_pipeline and self.eval_service is not None:
                payload = self.eval_service.retrieve_for_evaluation(
                    question,
                    top_k=top_k,
                    use_rerank=use_rerank,
                    retrieval_mode=retrieval_mode,
                    knowledge_base_ids=None if search_all_knowledge_bases else knowledge_base_ids,
                    auto_route=auto_route,
                    use_query_expansion=use_query_expansion,
                    use_retrieval_retry=use_retrieval_retry,
                    filters=filters,
                )
                results = payload["results"]
                route_info = payload.get("routing")
                query_plan = payload.get("query_plan")
                retry_info = payload.get("retrieval_retry")
            else:
                results = self.retriever.search(
                    question,
                    top_k=top_k,
                    use_rerank=use_rerank,
                    filters=scoped_filters or None,
                    mode=retrieval_mode,
                )

            retrieved_ids = [result.chunk.id for result in results]
            recall = recall_at_k(retrieved_ids, relevant_ids, top_k)
            hit = hit_at_k(retrieved_ids, relevant_ids, top_k)
            rr = reciprocal_rank(retrieved_ids, relevant_ids)
            precision = context_precision(retrieved_ids, relevant_ids, top_k)
            recalls.append(recall)
            hits.append(hit)
            reciprocal_ranks.append(rr)
            precisions.append(precision)
            row: dict[str, Any] = {
                "question": question,
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": relevant_ids,
                "recall": recall,
                "hit": hit,
                "reciprocal_rank": rr,
                "context_precision": precision,
            }
            if route_info is not None:
                row["routing"] = route_info
            if query_plan is not None:
                row["query_plan"] = query_plan
            if retry_info is not None:
                row["retrieval_retry"] = retry_info
            if include_generation and self.ask_service is not None:
                answer_payload = self.ask_service.ask(
                    question,
                    top_k=top_k,
                    use_rerank=use_rerank,
                    selected_knowledge_base_ids=knowledge_base_ids,
                    auto_route=not bool(knowledge_base_ids),
                    filters=filters,
                )
                contexts = self._contexts_from_answer(answer_payload, results)
                row["answer"] = str(answer_payload.get("answer") or "")
                row["contexts"] = contexts
                row["reference_answer"] = example.get("answer")
            rows.append(row)

        metrics: dict[str, Any] = {
            f"recall@{top_k}": round(aggregate(recalls), 4),
            f"hit@{top_k}": round(aggregate(hits), 4),
            "mrr": round(aggregate(reciprocal_ranks), 4),
            "context_precision": round(aggregate(precisions), 4),
        }

        generation_metrics: dict[str, Any] | None = None
        if include_generation:
            from backend.app.evaluation.ragas_metrics import score_generation_rows

            generation_metrics = score_generation_rows(rows)
            metrics.update(generation_metrics.get("aggregate", {}))

        return {
            "dataset_path": str(dataset_path),
            "top_k": top_k,
            "use_rerank": use_rerank,
            "retrieval_mode": retrieval_mode,
            "auto_route": auto_route,
            "use_query_expansion": use_query_expansion,
            "use_retrieval_retry": use_retrieval_retry,
            "search_all_knowledge_bases": search_all_knowledge_bases,
            "include_generation": include_generation,
            "knowledge_base_ids": knowledge_base_ids or [],
            "example_count": len(examples),
            "metrics": metrics,
            "generation": generation_metrics,
            "rows": rows,
        }

    def _contexts_from_answer(
        self,
        answer_payload: dict[str, Any],
        fallback_results: list[Any],
    ) -> list[str]:
        chunks = getattr(self.retriever.vector_index, "chunks", {}) or {}
        contexts: list[str] = []
        for cite in answer_payload.get("citations") or []:
            chunk_id = str(cite.get("chunk_id") or "")
            chunk = chunks.get(chunk_id)
            if chunk is not None and str(chunk.text).strip():
                contexts.append(str(chunk.text))
        if contexts:
            return contexts
        return [result.chunk.text for result in fallback_results if str(result.chunk.text).strip()]

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        examples: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                examples.append(json.loads(line))
        return examples
