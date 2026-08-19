from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.api.schemas import AskRequest, EvaluationRequest, TextIngestRequest
from backend.app.core.config import settings
from backend.app.core.service import service
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.indexing.chunker import ChunkStrategy
from backend.app.ingestion.exceptions import UnsupportedFileTypeError

router = APIRouter()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@router.get("/health")
def health() -> dict[str, object]:
    stats = service.knowledge_base_stats()
    return {
        "status": "ok",
        "configured_store": stats.get("configured_store", settings.vector_store),
        "store": stats.get("store"),
        "registry_path": stats.get("registry_path"),
        "documents": stats.get("documents", 0),
        "chunks": stats.get("chunks", 0),
        "knowledge_bases": stats.get("knowledge_bases", []),
        "clip_multimodal": settings.enable_clip_multimodal,
        "image_chunks": stats.get("image_chunks", 0),
    }


@router.get("/documents")
def list_documents(
    domain: str | None = None,
    knowledge_base_id: str | None = None,
) -> list[dict[str, object]]:
    return service.list_documents(domain=domain, knowledge_base_id=knowledge_base_id)


@router.post("/documents/text")
def ingest_text(payload: TextIngestRequest) -> dict[str, object]:
    return service.ingest_text(
        text=payload.text,
        title=payload.title,
        source=payload.source,
        strategy=payload.chunk_strategy,
        force_reindex=payload.force_reindex,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    chunk_strategy: ChunkStrategy = Form(default=ChunkStrategy.HEADING),
    force_reindex: bool = Form(default=False),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
) -> dict[str, object]:
    safe_name = Path(file.filename or "document.txt").name
    target = settings.upload_dir / safe_name
    content = await file.read()
    target.write_bytes(content)
    try:
        return service.ingest_file(
            target,
            title=title,
            strategy=chunk_strategy,
            force_reindex=force_reindex,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, object]:
    return service.delete_document(document_id)


@router.post("/documents/reindex-existing")
def reindex_existing_documents(
    chunk_strategy: ChunkStrategy = Form(default=ChunkStrategy.HEADING),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
) -> dict[str, object]:
    """B3: re-embed existing docs from source files that still exist; skip missing files."""
    return service.reindex_existing_documents(
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@router.get("/media")
def get_media(path: str) -> FileResponse:
    """Serve indexed images under storage/ for frontend visualization."""
    raw = unquote(path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="path is required")
    target = Path(raw).expanduser()
    allowed = [settings.storage_dir, settings.upload_dir]
    if not any(_is_under(target, root) for root in allowed):
        raise HTTPException(status_code=403, detail="path outside storage")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


@router.delete("/knowledge-bases/{knowledge_base_id}")
def delete_knowledge_base(knowledge_base_id: str) -> dict[str, object]:
    result = service.delete_knowledge_base(knowledge_base_id)
    if not result.get("deleted"):
        raise HTTPException(status_code=404, detail=result.get("message") or "not found")
    return result


@router.post("/ask")
def ask(payload: AskRequest) -> dict[str, object]:
    return service.ask(
        question=payload.question,
        top_k=payload.top_k,
        filters=payload.filters,
        use_rerank=payload.use_rerank,
        selected_knowledge_base_ids=payload.selected_knowledge_base_ids,
        auto_route=payload.auto_route,
    )


@router.post("/ask/image")
async def ask_with_image(
    file: UploadFile = File(...),
    question: str = Form(default=""),
    top_k: int = Form(default=6),
    use_rerank: bool = Form(default=True),
    auto_route: bool = Form(default=True),
    selected_knowledge_base_ids: str = Form(default=""),
) -> dict[str, object]:
    """Multimodal ask: 图搜图 / 图搜文 (CLIP). Optional text question for generation focus."""
    if not settings.enable_clip_multimodal:
        raise HTTPException(
            status_code=400,
            detail="CLIP multimodal is disabled. Set ENABLE_CLIP_MULTIMODAL=true and reindex.",
        )
    suffix = Path(file.filename or "query.png").suffix.lower() or ".png"
    target = settings.upload_dir / f"query_{Path(file.filename or 'image').stem}{suffix}"
    target.write_bytes(await file.read())
    selected_ids = [
        item.strip() for item in selected_knowledge_base_ids.split(",") if item.strip()
    ]
    return service.ask(
        question=question,
        top_k=top_k,
        use_rerank=use_rerank,
        auto_route=auto_route if not selected_ids else False,
        selected_knowledge_base_ids=selected_ids or None,
        query_image_path=str(target),
    )


@router.get("/knowledge-bases")
def list_knowledge_bases() -> list[dict[str, object]]:
    return service.list_knowledge_bases()


@router.get("/knowledge-bases/stats")
def knowledge_base_stats(
    domain: str | None = None,
    knowledge_base_id: str | None = None,
) -> dict[str, object]:
    return service.knowledge_base_stats(domain=domain, knowledge_base_id=knowledge_base_id)


@router.post("/evaluate")
def evaluate(payload: EvaluationRequest) -> dict[str, object]:
    mode = payload.retrieval_mode if payload.retrieval_mode in {"dense", "bm25", "hybrid"} else "hybrid"
    runner = EvaluationRunner(service.retriever, ask_service=service, eval_service=service)
    return runner.run(
        Path(payload.dataset_path),
        top_k=payload.top_k,
        use_rerank=payload.use_rerank,
        include_generation=payload.include_generation,
        knowledge_base_ids=payload.knowledge_base_ids,
        retrieval_mode=mode,  # type: ignore[arg-type]
    )
