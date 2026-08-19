"""Wipe the active vector store's KB/index and seed the general research knowledge base.

Uses VECTOR_STORE from .env:
  - local  -> clears index.json + knowledge_bases_local.json
  - qdrant -> clears Qdrant collection + qdrant_documents.json + knowledge_bases_qdrant.json

Does NOT touch the other store's data.

Usage (from repo root, conda env researchpilot):

  python scripts/reset_and_seed_general_kb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED_DIR = ROOT / "datasets" / "seed_general_research"
KB_ID = "kb_general_research"
KB_NAME = "通用科研知识"
KB_DESC = "通用科研方法与实践要点：文献、阅读、实验、写作、笔记、可复现与协作"


def _reset_active_store() -> str:
    from backend.app.core.config import settings
    from backend.app.knowledge.registry import registry_path_for_store

    store = settings.vector_store.strip().lower()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_path_for_store(store)

    if store == "qdrant":
        catalog = settings.storage_dir / "qdrant_documents.json"
        if catalog.exists():
            catalog.unlink()
            print(f"[reset] deleted {catalog}")
        if registry_path.exists():
            registry_path.unlink()
            print(f"[reset] deleted {registry_path}")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required when VECTOR_STORE=qdrant") from exc
        client = QdrantClient(url=settings.qdrant_url, trust_env=False)
        if client.collection_exists(settings.qdrant_collection):
            client.delete_collection(settings.qdrant_collection)
            print(f"[reset] deleted qdrant collection {settings.qdrant_collection}")
        else:
            print(f"[reset] qdrant collection {settings.qdrant_collection} already absent")
    else:
        index_path = settings.storage_dir / "index.json"
        if index_path.exists():
            index_path.unlink()
            print(f"[reset] deleted {index_path}")
        if registry_path.exists():
            registry_path.unlink()
            print(f"[reset] deleted {registry_path}")

    registry_path.write_text(
        json.dumps({"knowledge_bases": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[reset] wrote empty {registry_path.name} (store={store})")
    return store


def main() -> int:
    if not SEED_DIR.exists():
        print(f"[error] seed dir missing: {SEED_DIR}", file=sys.stderr)
        return 1

    store = _reset_active_store()

    # Import after wipe so the singleton loads an empty index/registry.
    from backend.app.core.service import service
    from backend.app.indexing.chunker import ChunkStrategy

    actual = getattr(service, "configured_store", None) or store
    index_store = service.knowledge_base_stats().get("store")
    print(f"[boot] configured_store={actual} index_store={index_store} registry={service.kb_registry.path}")
    if str(index_store) != store:
        print(
            f"[error] expected index store={store}, got {index_store}. "
            "Refusing to seed into the wrong backend.",
            file=sys.stderr,
        )
        return 1

    for kb in list(service.kb_registry.list_all()):
        service.delete_knowledge_base(kb.id)
    for doc_id in list(service.index.documents.keys()):
        service.delete_document(doc_id)

    service.kb_registry.ensure(
        name=KB_NAME,
        description=KB_DESC,
        knowledge_base_id=KB_ID,
    )

    results = []
    for path in sorted(SEED_DIR.glob("*.md")):
        result = service.ingest_file(
            path,
            title=path.stem,
            strategy=ChunkStrategy.HEADING,
            force_reindex=True,
            force_knowledge_base_id=KB_ID,
            force_knowledge_base_name=KB_NAME,
            force_knowledge_base_description=KB_DESC,
        )
        row = {
            "file": path.name,
            "document_id": result.get("document_id"),
            "chunk_count": result.get("chunk_count"),
            "duplicate": result.get("duplicate"),
            "knowledge_base_id": result.get("knowledge_base_id"),
        }
        results.append(row)
        print(
            f"[ingest] {path.name} -> kb={row['knowledge_base_id']} "
            f"chunks={row['chunk_count']} duplicate={row['duplicate']}"
        )

    stats = service.knowledge_base_stats()
    print(
        "[done] "
        f"store={stats.get('store')} "
        f"documents={stats.get('documents')} chunks={stats.get('chunks')} "
        f"image_chunks={stats.get('image_chunks')} "
        f"kbs={[kb.get('id') for kb in stats.get('knowledge_bases', [])]}"
    )
    report_path = settings_storage_report_path()
    report_path.write_text(
        json.dumps(
            {
                "store": stats.get("store"),
                "configured_store": stats.get("configured_store"),
                "registry_path": stats.get("registry_path"),
                "knowledge_base": {
                    "id": KB_ID,
                    "name": KB_NAME,
                    "description": KB_DESC,
                },
                "ingest": results,
                "stats": {
                    "documents": stats.get("documents"),
                    "chunks": stats.get("chunks"),
                    "image_chunks": stats.get("image_chunks"),
                    "knowledge_bases": stats.get("knowledge_bases"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] report -> {report_path}")
    return 0


def settings_storage_report_path() -> Path:
    from backend.app.core.config import settings

    return settings.storage_dir / "seed_general_research_report.json"


if __name__ == "__main__":
    raise SystemExit(main())
