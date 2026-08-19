from __future__ import annotations

from pathlib import Path

from backend.app.core.models import Chunk, Document
from backend.app.indexing.embeddings import HashEmbeddingProvider
from backend.app.indexing.store import ResearchIndex


class FakeClipProvider(HashEmbeddingProvider):
    """Minimal stand-in that looks like ClipEmbeddingProvider for isinstance checks."""

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        # Distinct from text hash so dual-path merge is observable.
        return [self._embed_one(f"img::{path}") for path in image_paths]


def test_vector_search_merges_text_and_image_hits_by_chunk_id(
    tmp_path: Path, monkeypatch
) -> None:
    # Make ResearchIndex treat FakeClip as ClipEmbeddingProvider.
    import backend.app.indexing.store as store_mod

    monkeypatch.setattr(store_mod, "ClipEmbeddingProvider", FakeClipProvider)

    index = ResearchIndex(FakeClipProvider(dimensions=32), storage_path=tmp_path / "index.json")
    document = Document(
        id="doc1",
        title="img doc",
        source="test",
        text="caption about neural nets",
        metadata={},
    )
    img_file = tmp_path / "figure.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # path exists; embed uses path string

    chunk = Chunk(
        id="chunk_img",
        document_id="doc1",
        text="OCR: neural network diagram",
        metadata={
            "modality": "image",
            "element_type": "image",
            "image_path": str(img_file),
            "asset_id": "asset1",
        },
    )
    index.add_document(document, [chunk])
    assert "chunk_img" in index.vectors
    assert "chunk_img" in index.image_vectors

    text_hits = index.vector_search("neural network", top_k=5)
    assert text_hits and text_hits[0].chunk.id == "chunk_img"

    image_hits = index.vector_search("", top_k=5, query_image_path=img_file)
    assert image_hits and image_hits[0].chunk.id == "chunk_img"

    both = index.vector_search("neural", top_k=5, query_image_path=img_file)
    assert len(both) == 1
    assert both[0].chunk.id == "chunk_img"
