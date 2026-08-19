from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: backend/app/core/config.py -> ../../..
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Prefer repo-root .env so keys live in one file instead of system env vars.
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()  # also allow cwd .env when running from elsewhere


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "ResearchPilot"
    storage_dir: Path = Path(_env("RESEARCHPILOT_STORAGE_DIR", "storage"))
    upload_dir: Path = Path(_env("RESEARCHPILOT_UPLOAD_DIR", "storage/uploads"))
    vector_store: str = _env("VECTOR_STORE", "local")
    qdrant_url: str = _env("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = _env("QDRANT_COLLECTION", "research_chunks")
    # Qdrant has no Faiss-style "FLAT" enum; exact=True forces full scan (local-parity).
    qdrant_exact_search: bool = _env_bool("QDRANT_EXACT_SEARCH", True)
    default_domain: str = _env("DEFAULT_DOMAIN", "general")
    default_knowledge_base_id: str = _env("DEFAULT_KNOWLEDGE_BASE_ID", "kb_general")
    chunk_size: int = int(_env("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "120"))
    # 1A: unified CLIP space for text + image (图搜文 / 图搜图). Default on.
    enable_clip_multimodal: bool = _env_bool("ENABLE_CLIP_MULTIMODAL", True)
    clip_text_model: str = _env(
        "CLIP_TEXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )
    clip_image_model: str = _env("CLIP_IMAGE_MODEL", "clip-ViT-B-32")
    # CLIP ViT-B/32 is typically 512-d; MiniLM fallback is 384-d.
    embedding_dimensions: int = int(
        _env(
            "EMBEDDING_DIMENSIONS",
            "512" if _env_bool("ENABLE_CLIP_MULTIMODAL", True) else "384",
        )
    )
    embedding_model: str = _env(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Reranker: cloud (SiliconFlow /v1/rerank) with lexical fallback.
    # RERANKER_PROVIDER: cloud | lexical
    reranker_provider: str = _env("RERANKER_PROVIDER", "cloud")
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_api_base: str = _env("RERANKER_API_BASE", "https://api.siliconflow.cn/v1")
    reranker_api_key: str = _env("RERANKER_API_KEY", "")
    reranker_timeout_seconds: float = float(_env("RERANKER_TIMEOUT_SECONDS", "30"))

    # Answer generation (and multimodal Q&A). Prefer a capable / VL model.
    llm_provider: str = _env("LLM_PROVIDER", "openai_compatible")
    llm_api_base: str = _env("LLM_API_BASE", "")
    llm_api_key: str = _env("LLM_API_KEY", "")
    llm_model: str = _env("LLM_MODEL", "")
    llm_timeout_seconds: float = float(_env("LLM_TIMEOUT_SECONDS", "60"))
    # KB assign + query routing + query rewrite. Prefer a cheap text model.
    # Empty fields fall back to the corresponding LLM_* value.
    router_llm_api_base: str = _env("ROUTER_LLM_API_BASE", "")
    router_llm_api_key: str = _env("ROUTER_LLM_API_KEY", "")
    router_llm_model: str = _env("ROUTER_LLM_MODEL", "")
    router_llm_timeout_seconds: float = float(_env("ROUTER_LLM_TIMEOUT_SECONDS", "60"))
    # Query routing picks up to this many knowledge bases.
    kb_route_top_n: int = int(_env("KB_ROUTE_TOP_N", "3"))
    # Low-confidence threshold for secondary retrieval (best fused score).
    retrieval_low_confidence_threshold: float = float(
        _env("RETRIEVAL_LOW_CONFIDENCE_THRESHOLD", "0.35")
    )
    retrieval_candidate_k: int = int(_env("RETRIEVAL_CANDIDATE_K", "20"))
    retrieval_retry_candidate_k: int = int(_env("RETRIEVAL_RETRY_CANDIDATE_K", "40"))
    query_max_sub_queries: int = int(_env("QUERY_MAX_SUB_QUERIES", "4"))
    # Document routing / multimodal parsing
    doc_router_enabled: bool = _env_bool("DOC_ROUTER_ENABLED", True)
    pdf_scan_text_density_threshold: float = float(
        _env("PDF_SCAN_TEXT_DENSITY_THRESHOLD", "40")
    )
    # Cloud VLM (OpenAI-compatible: SiliconFlow / Bailian / etc.)
    enable_cloud_vlm: bool = _env_bool("ENABLE_CLOUD_VLM", False)
    vlm_api_base: str = _env("VLM_API_BASE", "")
    vlm_api_key: str = _env("VLM_API_KEY", "")
    vlm_model: str = _env("VLM_MODEL", "")
    vlm_timeout_seconds: float = float(_env("VLM_TIMEOUT_SECONDS", "60"))
    # Max VLM calls per document (scanned PDF pages + embedded images).
    # Set to 0 (or negative) for unlimited calls.
    vlm_max_calls: int = int(_env("VLM_MAX_CALLS", "0"))


settings = Settings()
