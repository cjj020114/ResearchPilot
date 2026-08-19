from __future__ import annotations

import logging
import sys
import types
from typing import Any

from backend.app.core.config import settings
from backend.app.evaluation.metrics import aggregate
from backend.app.knowledge.llm_client import TextLLMClient

logger = logging.getLogger(__name__)

# Extended RAGAS-style metric set (2B).
RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


def score_generation_rows(
    rows: list[dict[str, Any]],
    *,
    prefer: str = "auto",
) -> dict[str, Any]:
    """Score generation quality via RAGAS when possible, else LLM-as-judge.

    prefer: auto | ragas | llm
    """
    usable = [row for row in rows if str(row.get("answer") or "").strip()]
    if not usable:
        zeros = {name: 0.0 for name in RAGAS_METRIC_NAMES}
        return {
            "backend": "none",
            "aggregate": zeros,
            "detail": [],
            "message": "No generated answers to score.",
        }

    if prefer == "llm":
        return _llm_judge_extended(usable)

    ragas_result = _try_ragas(usable) if prefer in {"auto", "ragas"} else None
    if ragas_result is not None and ragas_result.get("backend") == "ragas":
        return ragas_result
    if prefer == "ragas" and ragas_result is not None:
        return ragas_result

    fallback = _llm_judge_extended(usable)
    if ragas_result is not None and ragas_result.get("message"):
        fallback["message"] = (
            f"{ragas_result.get('message')}; fell back to LLM-as-judge. "
            f"{fallback.get('message')}"
        )
    return fallback


def _patch_ragas_imports() -> None:
    """ragas 0.4.x imports removed langchain VertexAI path on newer community."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    module = types.ModuleType(name)

    class ChatVertexAI:  # noqa: N801 - match upstream symbol
        pass

    module.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[name] = module


def _try_ragas(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        _patch_ragas_imports()
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:
        logger.warning("RAGAS import failed: %s", exc)
        return None

    if not (settings.llm_api_key and settings.llm_model):
        return {
            "backend": "ragas_failed",
            "aggregate": {},
            "detail": [],
            "message": "RAGAS needs LLM_API_KEY / LLM_MODEL.",
        }

    samples: list[Any] = []
    for row in rows:
        samples.append(
            SingleTurnSample(
                user_input=str(row["question"]),
                response=str(row.get("answer") or ""),
                retrieved_contexts=[
                    str(item) for item in (row.get("contexts") or []) if str(item).strip()
                ],
                reference=str(row.get("reference_answer") or row.get("ground_truth") or ""),
            )
        )
    dataset = EvaluationDataset(samples=samples)

    try:
        chat = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=(settings.llm_api_base or "https://api.openai.com/v1").rstrip("/"),
            temperature=0,
            timeout=float(getattr(settings, "llm_timeout_seconds", 60) or 60),
            max_retries=2,
        )
        ragas_llm = LangchainLLMWrapper(chat)
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        ragas_embeddings = LangchainEmbeddingsWrapper(embedder)
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
            show_progress=True,
            batch_size=4,
        )
    except Exception as exc:
        return {
            "backend": "ragas_failed",
            "aggregate": {},
            "detail": [],
            "message": f"RAGAS evaluate failed: {exc}",
        }

    payload = result.to_pandas().to_dict(orient="records") if hasattr(result, "to_pandas") else []
    buckets: dict[str, list[float]] = {name: [] for name in RAGAS_METRIC_NAMES}
    detail: list[dict[str, Any]] = []
    key_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevance",
        "answer_relevance": "answer_relevance",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
        "answer_correctness": "answer_correctness",
    }

    for index, row in enumerate(rows):
        item = payload[index] if index < len(payload) else {}
        scored: dict[str, float | None] = {name: None for name in RAGAS_METRIC_NAMES}
        for src, dst in key_map.items():
            value = _as_float(item.get(src))
            if value is None:
                continue
            scored[dst] = value
            buckets[dst].append(value)
            row[dst] = value
        detail.append({"question": row["question"], **scored})

    # If nearly all scores missing, treat as failure so caller can fall back.
    total_scores = sum(len(values) for values in buckets.values())
    if total_scores == 0:
        return {
            "backend": "ragas_failed",
            "aggregate": {},
            "detail": detail,
            "message": "RAGAS returned no numeric scores.",
        }

    return {
        "backend": "ragas",
        "aggregate": {
            name: round(aggregate(values), 4) for name, values in buckets.items()
        },
        "detail": detail,
        "message": (
            "Scored with RAGAS: faithfulness, answer_relevancy, "
            "context_precision, context_recall, answer_correctness."
        ),
    }


def _llm_judge_extended(rows: list[dict[str, Any]]) -> dict[str, Any]:
    llm = TextLLMClient()
    if not llm.enabled:
        return {
            "backend": "unavailable",
            "aggregate": {name: 0.0 for name in RAGAS_METRIC_NAMES},
            "detail": [],
            "message": "RAGAS unavailable and Text LLM not configured.",
        }

    buckets: dict[str, list[float]] = {name: [] for name in RAGAS_METRIC_NAMES}
    detail: list[dict[str, Any]] = []
    for row in rows:
        question = str(row["question"])
        answer = str(row.get("answer") or "")
        reference = str(row.get("reference_answer") or row.get("ground_truth") or "")
        contexts = "\n\n".join(str(item) for item in (row.get("contexts") or [])[:6])
        scores = {
            "faithfulness": _score_one(
                llm,
                system=(
                    "You judge RAG faithfulness. Return ONLY JSON: "
                    '{"score": number between 0 and 1, "reason": string}. '
                    "1 = every claim is supported by contexts; 0 = unsupported invention."
                ),
                user=f"Contexts:\n{contexts}\n\nAnswer:\n{answer}",
            ),
            "answer_relevance": _score_one(
                llm,
                system=(
                    "You judge answer relevance. Return ONLY JSON: "
                    '{"score": number between 0 and 1, "reason": string}. '
                    "1 = directly answers the question; 0 = off-topic."
                ),
                user=f"Question:\n{question}\n\nAnswer:\n{answer}",
            ),
            "context_precision": _score_one(
                llm,
                system=(
                    "You judge context precision for RAG. Return ONLY JSON: "
                    '{"score": number between 0 and 1, "reason": string}. '
                    "1 = retrieved contexts are mostly useful for answering; "
                    "0 = contexts are mostly irrelevant noise."
                ),
                user=f"Question:\n{question}\n\nContexts:\n{contexts}",
            ),
            "context_recall": _score_one(
                llm,
                system=(
                    "You judge context recall against a reference answer. Return ONLY JSON: "
                    '{"score": number between 0 and 1, "reason": string}. '
                    "1 = contexts cover the key facts in the reference; "
                    "0 = important reference facts are missing from contexts."
                ),
                user=(
                    f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
                    f"Contexts:\n{contexts}"
                ),
            ),
            "answer_correctness": _score_one(
                llm,
                system=(
                    "You judge answer correctness vs reference. Return ONLY JSON: "
                    '{"score": number between 0 and 1, "reason": string}. '
                    "1 = answer matches reference key facts; 0 = contradicts or misses them."
                ),
                user=(
                    f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
                    f"Model answer:\n{answer}"
                ),
            ),
        }
        for name, value in scores.items():
            row[name] = value
            buckets[name].append(value)
        detail.append({"question": question, **scores})

    return {
        "backend": "llm_judge_ragas_style",
        "aggregate": {
            name: round(aggregate(values), 4) for name, values in buckets.items()
        },
        "detail": detail,
        "message": (
            "Used LLM-as-judge for faithfulness, answer_relevance, "
            "context_precision, context_recall, answer_correctness."
        ),
    }


def _score_one(llm: TextLLMClient, *, system: str, user: str) -> float:
    data = llm.chat_json(system, user)
    value = _as_float(data.get("score"))
    if value is None:
        return 0.0
    return max(0.0, min(1.0, value))


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
