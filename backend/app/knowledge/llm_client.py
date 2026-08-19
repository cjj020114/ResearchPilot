from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Sequence

from backend.app.core.config import settings


class TextLLMClient:
    """OpenAI-compatible text/vision LLM client (SiliconFlow / Bailian / etc.)."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        config_label: str = "LLM",
    ) -> None:
        self.config_label = config_label
        self.api_base = (api_base or settings.llm_api_base or "https://api.openai.com/v1").rstrip(
            "/"
        )
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model if model is not None else settings.llm_model
        self.timeout = float(
            timeout
            if timeout is not None
            else getattr(settings, "llm_timeout_seconds", 60) or 60
        )
        self.enabled = bool(self.api_key and self.model)

    @classmethod
    def for_answer(cls) -> "TextLLMClient":
        """Final answer generation (LLM_*)."""
        return cls(
            api_base=settings.llm_api_base,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            config_label="LLM",
        )

    @classmethod
    def for_router(cls) -> "TextLLMClient":
        """KB assignment, query routing, and query rewrite (ROUTER_LLM_*, fallback LLM_*)."""
        return cls(
            api_base=settings.router_llm_api_base or settings.llm_api_base,
            api_key=settings.router_llm_api_key or settings.llm_api_key,
            model=settings.router_llm_model or settings.llm_model,
            timeout=settings.router_llm_timeout_seconds or settings.llm_timeout_seconds,
            config_label="ROUTER_LLM",
        )

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                f"Text LLM is not configured ({self.config_label}_API_KEY / {self.config_label}_MODEL)."
            )

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        self._require_enabled()
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return self._parse_json(content)

    def chat_text(self, system: str, user: str) -> str:
        self._require_enabled()
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    def chat_with_images(
        self,
        system: str,
        user_text: str,
        image_paths: Sequence[str | Path] | None = None,
    ) -> str:
        """Multimodal chat: text + optional local image files (for VLM answer models)."""
        self._require_enabled()
        paths = [Path(p) for p in (image_paths or []) if p and Path(p).is_file()]
        if not paths:
            return self.chat_text(system, user_text)

        from openai import OpenAI

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for path in paths:
            content.append(
                {"type": "image_url", "image_url": {"url": self._to_data_url(path)}}
            )
        client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    def _to_data_url(self, path: Path) -> str:
        raw = path.read_bytes()
        guessed, _ = mimetypes.guess_type(str(path))
        mime = guessed or "image/png"
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _parse_json(self, content: str) -> dict[str, Any]:
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
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        raise ValueError(f"LLM did not return valid JSON: {content[:300]}")
