from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TABLE_TYPES = {"csv", "xlsx", "xls"}
UNSUPPORTED_TYPES = {"json", "yaml", "yml", "xml"}
DOCUMENT_TYPES = {"txt", "log", "md", "pdf", "docx", "ppt", "pptx", "html", "htm"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_table_file(path: Path) -> bool:
    return detect_file_type(path) in TABLE_TYPES


def detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "markdown":
        return "md"
    if suffix == "htm":
        return "html"
    return suffix or "unknown"
