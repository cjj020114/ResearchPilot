"""Seed the offline research KB and build a runnable golden set.

Usage (from repo root, conda env researchpilot):

  python scripts/seed_research_kb.py
  python scripts/seed_research_kb.py --evaluate
  python scripts/seed_research_kb.py --evaluate --include-generation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.service import service
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.indexing.chunker import ChunkStrategy

SEED_DIR = ROOT / "datasets" / "seed_research"
QUESTIONS_PATH = SEED_DIR / "eval_questions.json"
GOLDEN_PATH = ROOT / "datasets" / "golden_set.seed.jsonl"
KB_ID = "kb_research_seed"
KB_NAME = "科研种子库"
KB_DESC = "公开资料整理的 RAG / EMG-NIRS 离线评测语料"


def _ingest_seed(*, force_reindex: bool) -> list[dict]:
    results = []
    for path in sorted(SEED_DIR.glob("*.md")):
        title = path.stem
        result = service.ingest_file(
            path,
            title=title,
            strategy=ChunkStrategy.HEADING,
            force_reindex=force_reindex,
            force_knowledge_base_id=KB_ID,
            force_knowledge_base_name=KB_NAME,
            force_knowledge_base_description=KB_DESC,
        )
        results.append({"file": path.name, **{k: result.get(k) for k in (
            "document_id", "chunk_count", "duplicate", "knowledge_base_id", "message"
        )}})
        print(f"[ingest] {path.name} -> kb={result.get('knowledge_base_id')} "
              f"chunks={result.get('chunk_count')} duplicate={result.get('duplicate')}")
    return results


def _find_chunk_ids(anchor: str, knowledge_base_id: str) -> list[str]:
    hits: list[str] = []
    for chunk in service.index.chunks.values():
        if chunk.metadata.get("knowledge_base_id") != knowledge_base_id:
            continue
        if anchor in chunk.text:
            hits.append(chunk.id)
    return hits


def _build_golden() -> Path:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    lines: list[str] = []
    missing: list[str] = []
    for item in questions:
        anchor = str(item["anchor"])
        chunk_ids = _find_chunk_ids(anchor, KB_ID)
        if not chunk_ids:
            missing.append(anchor)
            continue
        payload = {
            "question": item["question"],
            "relevant_chunk_ids": chunk_ids,
            "answer": item.get("answer"),
            "anchor": anchor,
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    GOLDEN_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[golden] wrote {len(lines)} examples -> {GOLDEN_PATH}")
    if missing:
        print(f"[golden] WARNING missing anchors: {missing}")
    return GOLDEN_PATH


def _evaluate(*, include_generation: bool, top_k: int) -> dict:
    runner = EvaluationRunner(service.retriever, ask_service=service)
    report = runner.run(
        GOLDEN_PATH,
        top_k=top_k,
        use_rerank=True,
        include_generation=include_generation,
        knowledge_base_ids=[KB_ID],
    )
    out = ROOT / "storage" / "eval_seed_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report.get("metrics"), ensure_ascii=False, indent=2))
    if report.get("generation"):
        print("[generation]", report["generation"].get("backend"), report["generation"].get("message"))
    print(f"[evaluate] report -> {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed research KB and build golden set")
    parser.add_argument("--force-reindex", action="store_true", help="Reindex even if duplicate")
    parser.add_argument("--evaluate", action="store_true", help="Run retrieval (+optional RAGAS) eval")
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Also score faithfulness / answer_relevance",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-ingest", action="store_true", help="Only rebuild golden / evaluate")
    args = parser.parse_args()

    if not args.skip_ingest:
        _ingest_seed(force_reindex=args.force_reindex)
    _build_golden()
    if args.evaluate:
        _evaluate(include_generation=args.include_generation, top_k=args.top_k)


if __name__ == "__main__":
    main()
