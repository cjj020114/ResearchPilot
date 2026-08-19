from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.core.models import Document, Element
from backend.app.indexing.chunker import ChunkStrategy, Chunker
from backend.app.ingestion.detectors.modality import detect_file_type, is_image_file, is_table_file
from backend.app.ingestion.exceptions import UnsupportedFileTypeError
from backend.app.ingestion.router import DocumentRouter


def test_is_image_file(tmp_path: Path) -> None:
    assert is_image_file(tmp_path / "a.png")
    assert not is_image_file(tmp_path / "a.pdf")


def test_is_table_file(tmp_path: Path) -> None:
    assert is_table_file(tmp_path / "a.csv")
    assert is_table_file(tmp_path / "a.xlsx")
    assert is_table_file(tmp_path / "a.xls")
    assert not is_table_file(tmp_path / "a.txt")


def test_route_txt_is_document_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello research pilot", encoding="utf-8")
    route = DocumentRouter().route(path)
    assert route.layer1 == "document"
    assert route.layer2 == "txt"
    assert route.loader == "text"


def test_route_image_uses_ocr_vlm(tmp_path: Path) -> None:
    path = tmp_path / "scan_page_01.png"
    path.write_bytes(b"not-a-real-image")
    route = DocumentRouter().route(path)
    assert route.layer1 == "image"
    assert route.loader == "ocr_vlm"


def test_route_markdown_stays_markdown(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    content = "\n".join(f"![fig{i}](./img{i}.png)" for i in range(5))
    path.write_text("# Title\n" + content, encoding="utf-8")
    route = DocumentRouter().route(path)
    assert route.layer1 == "document"
    assert route.loader == "markdown"


def test_route_rejects_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        DocumentRouter().route(path)


def test_route_table_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    route = DocumentRouter().route(path)
    assert route.layer1 == "table"
    assert route.loader == "table"
    assert detect_file_type(path) == "csv"


def test_infer_title_skips_page_marker() -> None:
    from backend.app.ingestion.loaders.common import infer_title

    title = infer_title("[page:1]\n\nReal Paper Title About NIR\nbody", "scan.pdf")
    assert title == "Real Paper Title About NIR"


def test_infer_title_falls_back_to_filename() -> None:
    from backend.app.ingestion.loaders.common import infer_title

    title = infer_title("[page:1]\n[page:2]", "my_research_paper.pdf")
    assert title == "my_research_paper"


def test_chunker_keeps_image_element_as_one_chunk() -> None:
    image = Element.create(
        type="image",
        text="ocr line\n\ncaption about figure",
        ocr_text="ocr line",
        vlm_caption="caption about figure",
    )
    long_text = Element.create(type="text", text=("paragraph " * 80).strip())
    document = Document.create(
        title="demo",
        source="demo.md",
        text="unused",
        elements=[image, long_text],
    )
    chunks = Chunker(chunk_size=100, overlap=10).chunk(document, strategy=ChunkStrategy.FIXED)
    image_chunks = [chunk for chunk in chunks if chunk.metadata.get("element_id") == image.id]
    assert len(image_chunks) == 1
    assert image_chunks[0].metadata["element_type"] == "image"
    assert len(chunks) >= 2


def test_chunker_plain_text_without_elements() -> None:
    document = Document.create(
        title="manual",
        source="manual",
        text="hello world\n\nsecond paragraph",
        elements=None,
    )
    chunks = Chunker(chunk_size=50, overlap=5).chunk(document, strategy=ChunkStrategy.RECURSIVE)
    assert chunks
    assert all(chunk.metadata.get("chunk_by_element") is False for chunk in chunks)
