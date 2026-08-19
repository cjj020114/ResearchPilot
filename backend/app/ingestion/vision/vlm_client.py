from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import settings


@dataclass(frozen=True)
class VlmEnrichment:
    ocr_text: str
    vlm_caption: str
    raw_text: str
    status: str  # ready | fallback | disabled
    error: str | None = None

    @property
    def combined_text(self) -> str:
        parts = [part.strip() for part in (self.ocr_text, self.vlm_caption) if part and part.strip()]
        if parts:
            return "\n\n".join(parts)
        return self.raw_text.strip()


class VlmBudget:
    """Shared per-document call budget for scanned pages and embedded images.

    ``max_calls <= 0`` means unlimited.
    """

    def __init__(self, max_calls: int | None = None) -> None:
        self.max_calls = settings.vlm_max_calls if max_calls is None else max_calls
        self.used = 0

    @property
    def unlimited(self) -> bool:
        return self.max_calls <= 0

    @property
    def remaining(self) -> int:
        if self.unlimited:
            return 10**9
        return max(0, self.max_calls - self.used)

    def try_consume(self) -> bool:
        if not self.unlimited and self.used >= self.max_calls:
            return False
        self.used += 1
        return True


class VlmClient:
    """OpenAI-compatible vision client (SiliconFlow / Bailian / etc.)."""

    def __init__(self) -> None:
        self.enabled = bool(
            settings.enable_cloud_vlm and settings.vlm_api_key and settings.vlm_model
        )
        self.api_base = (settings.vlm_api_base or "https://api.openai.com/v1").rstrip("/")
        self.api_key = settings.vlm_api_key
        self.model = settings.vlm_model
        self.timeout = settings.vlm_timeout_seconds

    def enrich_image(
        self,
        image: Path | bytes,
        *,
        mime_type: str | None = None,
        context: str | None = None,
        budget: VlmBudget | None = None,
        fallback_label: str = "image",
    ) -> VlmEnrichment:
        if budget is not None and not budget.try_consume():
            return self._fallback(
                fallback_label,
                status="fallback",
                error="vlm_max_calls_exceeded",
            )
        if not self.enabled:
            return self._fallback(fallback_label, status="disabled", error="vlm_disabled")

        try:
            data_url = self._to_data_url(image, mime_type=mime_type)
            prompt = self._build_prompt(context)
            content = self._chat_vision(data_url, prompt)
            parsed = self._parse_response(content)
            return VlmEnrichment(
                ocr_text=parsed.get("ocr_text", "") or "",
                vlm_caption=parsed.get("vlm_caption", "") or "",
                raw_text=content,
                status="ready",
            )
        except Exception as exc:  # noqa: BLE001 - soft-fail per product decision
            return self._fallback(
                fallback_label,
                status="fallback",
                error=str(exc),
            )

    def _fallback(self, label: str, *, status: str, error: str | None) -> VlmEnrichment:
        stub = (
            f"[vlm_fallback] {label}\n"
            "OCR text and VLM caption unavailable; document ingestion continues."
        )
        return VlmEnrichment(
            ocr_text="",
            vlm_caption="",
            raw_text=stub,
            status=status,
            error=error,
        )

    def _build_prompt(self, context: str | None) -> str:
        context_block = f"\nSurrounding document context:\n{context[:1200]}\n" if context else ""
        return (
            "You are extracting content from a research document image.\n"
            "Return ONLY a JSON object with keys:\n"
            '  "ocr_text": verbatim readable text in the image (empty string if none),\n'
            '  "vlm_caption": a concise semantic description of the figure/page.\n'
            "Do not wrap the JSON in markdown fences."
            f"{context_block}"
        )

    def _chat_vision(self, data_url: str, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0,
        )
        message = response.choices[0].message.content
        if not message:
            raise RuntimeError("empty VLM response")
        return message if isinstance(message, str) else str(message)

    def _to_data_url(self, image: Path | bytes, mime_type: str | None = None) -> str:
        if isinstance(image, Path):
            raw = image.read_bytes()
            guessed, _ = mimetypes.guess_type(str(image))
            mime = mime_type or guessed or "image/png"
        else:
            raw = image
            mime = mime_type or "image/png"
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _parse_response(self, content: str) -> dict[str, Any]:
        cleaned = content.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # Soft parse: treat whole reply as caption if JSON missing.
        return {"ocr_text": "", "vlm_caption": cleaned}


_client: VlmClient | None = None


def get_vlm_client() -> VlmClient:
    global _client
    if _client is None:
        _client = VlmClient()
    return _client
