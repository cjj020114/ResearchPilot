from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

from backend.app.core.config import settings
from backend.app.core.models import SearchResult
from backend.app.knowledge.llm_client import TextLLMClient


class AnswerGenerator:
    """Grounded answer generator: LLM/VLM with retrieved context, extractive fallback."""

    def __init__(self, llm: TextLLMClient | None = None) -> None:
        self.llm = llm or TextLLMClient()

    def generate(
        self,
        question: str,
        contexts: list[SearchResult],
        *,
        query_image_path: str | None = None,
    ) -> dict[str, object]:
        if not contexts:
            return {
                "answer": "我没有在当前知识库中找到足够可靠的依据来回答这个问题。",
                "citations": [],
                "confidence": "low",
                "generator": "none",
            }

        citations = [self._citation(result) for result in contexts[: max(3, len(contexts))]]
        if self.llm.enabled and settings.llm_provider.lower() != "extractive":
            try:
                answer, generator = self._generate_with_llm(
                    question, contexts, query_image_path=query_image_path
                )
                return {
                    "answer": answer,
                    "citations": citations[:5],
                    "confidence": self.confidence_label(contexts),
                    "generator": generator,
                }
            except Exception as exc:  # noqa: BLE001 - fall back for demo resilience
                extractive = self._extractive(question, contexts)
                extractive["generator"] = "extractive_fallback"
                extractive["generator_error"] = str(exc)
                return extractive

        result = self._extractive(question, contexts)
        result["generator"] = "extractive"
        return result

    def _generate_with_llm(
        self,
        question: str,
        contexts: list[SearchResult],
        *,
        query_image_path: str | None = None,
    ) -> tuple[str, str]:
        blocks: list[str] = []
        recalled_images: list[Path] = []
        for index, result in enumerate(contexts[:8], start=1):
            meta = result.chunk.metadata
            title = meta.get("document_title") or meta.get("source") or "unknown"
            page = meta.get("page")
            modality = meta.get("modality") or meta.get("element_type") or "text"
            header = f"[{index}] title={title} modality={modality}"
            if page is not None:
                header += f" page={page}"
            header += f" score={result.score:.3f}"
            blocks.append(f"{header}\n{result.chunk.text.strip()}")
            image_path = meta.get("image_path")
            if image_path and Path(str(image_path)).is_file() and len(recalled_images) < 3:
                recalled_images.append(Path(str(image_path)))

        context_text = "\n\n".join(blocks)
        has_query_image = bool(query_image_path and Path(query_image_path).is_file())
        image_only = has_query_image and not (question or "").strip()

        system = (
            "You are ResearchPilot, a careful multimodal research assistant.\n"
            "Use ONLY the provided evidence snippets and any images actually attached in this message.\n"
            "If evidence is insufficient, say so clearly.\n"
            "Cite evidence by [index] markers in the answer.\n"
            "Respond in Chinese unless the question is clearly in another language.\n"
            "About images:\n"
            "- If this user message includes image attachments, analyze them. "
            "The first attached image (if present) is the user query image; "
            "later ones are retrieved evidence figures.\n"
            "- If this user message includes NO image attachments, say clearly that "
            "no image was attached in this request, and answer from text evidence only. "
            "Do not invent visual details.\n"
            "- When an evidence item is modality=image, its text is OCR/caption from ingest; "
            "if the corresponding figure is also attached, prefer the figure for visual claims.\n"
        )

        if image_only:
            user_question = (
                "用户只上传了查询图片（见附件）。请结合检索到的资料与图片，"
                "说明查询图与资料的关系，并概括相关要点。"
            )
        else:
            user_question = f"Original question:\n{question}\n"

        user = (
            f"{user_question}\n"
            f"Evidence retrieved (text may include OCR/VLM captions for figures):\n"
            f"{context_text}\n"
        )

        image_paths: list[Path] = []
        if has_query_image:
            image_paths.append(Path(str(query_image_path)))
        for path in recalled_images:
            if path.resolve() not in {p.resolve() for p in image_paths}:
                image_paths.append(path)

        if image_paths:
            answer = self.llm.chat_with_images(system, user, image_paths)
            return answer, "vlm"
        answer = self.llm.chat_text(system, user)
        return answer, "llm"

    def confidence_label(self, results: list[SearchResult], threshold_low: float = 0.35) -> str:
        best = max((result.score for result in results), default=0.0)
        if best >= 0.75:
            return "high"
        if best >= threshold_low:
            return "medium"
        return "low"

    def best_score(self, results: list[SearchResult]) -> float:
        return max((result.score for result in results), default=0.0)

    def _extractive(self, question: str, contexts: list[SearchResult]) -> dict[str, object]:
        evidence = contexts[:3]
        bullets = []
        for index, result in enumerate(evidence, start=1):
            snippet = self._snippet(result.chunk.text)
            bullets.append(f"{index}. {snippet}")
        label = question or "（图片查询）"
        answer = (
            f"基于当前检索到的科研资料，问题“{label}”可以从以下证据回答：\n"
            + "\n".join(bullets)
            + "\n\n请优先核对引用片段；如果需要正式写作，建议回到原文页码确认。"
        )
        return {
            "answer": answer,
            "citations": [self._citation(result) for result in evidence],
            "confidence": self.confidence_label(evidence),
        }

    def _snippet(self, text: str, length: int = 360) -> str:
        compact = " ".join(text.split())
        return compact if len(compact) <= length else compact[: length - 3] + "..."

    def _citation(self, result: SearchResult) -> dict[str, object]:
        metadata = result.chunk.metadata
        image_path = metadata.get("image_path")
        image_url = None
        if image_path:
            image_url = f"/api/media?path={quote(str(image_path), safe='')}"
        return {
            "chunk_id": result.chunk.id,
            "document_id": result.chunk.document_id,
            "document_title": metadata.get("document_title"),
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "heading": metadata.get("heading"),
            "score": round(result.score, 4),
            "vector_score": round(result.vector_score, 4),
            "bm25_score": round(result.bm25_score, 4),
            "rerank_score": round(result.rerank_score, 4),
            "knowledge_base_id": metadata.get("knowledge_base_id"),
            "modality": metadata.get("modality"),
            "image_path": image_path,
            "image_url": image_url,
            "metadata": asdict(result.chunk)["metadata"],
        }
