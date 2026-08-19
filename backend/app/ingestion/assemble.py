from __future__ import annotations

import re

from backend.app.core.models import Element


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def element_display_text(element: Element) -> str:
    parts: list[str] = []
    if element.type == "image":
        if element.ocr_text:
            parts.append(element.ocr_text.strip())
        if element.vlm_caption:
            parts.append(element.vlm_caption.strip())
        if not parts and element.text:
            parts.append(element.text.strip())
    elif element.text:
        parts.append(element.text.strip())
    return _normalize("\n\n".join(part for part in parts if part))


def assemble_elements_text(elements: list[Element]) -> str:
    blocks: list[str] = []
    for element in elements:
        body = element_display_text(element)
        if not body:
            continue
        if element.page is not None and not body.startswith("[page:"):
            body = f"[page:{element.page}]\n{body}"
        blocks.append(body)
    return _normalize("\n\n".join(blocks))
