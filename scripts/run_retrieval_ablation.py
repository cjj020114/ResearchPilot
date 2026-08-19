"""Retrieval + query-optimization ablation.

Suites:
  retrieval  - dense / bm25 / hybrid / hybrid+rerank
  query_opt - no_opt vs full_opt on hybrid+rerank (+ distractor KB for routing)
  all        - both

Usage:
  python scripts/run_retrieval_ablation.py --suite query_opt
  python scripts/run_retrieval_ablation.py --suite all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.service import service
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.indexing.chunker import ChunkStrategy
from backend.app.retrieval.hybrid import RetrievalMode

SEED_KB = "kb_research_seed"
DISTRACTOR_KB = "kb_distractor"
DISTRACTOR_DIR = ROOT / "datasets" / "seed_distractor"

RETRIEVAL_ARMS: list[tuple[str, RetrievalMode, bool]] = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = ["arm", "hit@k", "mrr", "recall@k", "context_precision", "latency_s"]
    lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
    for row in rows:
        metrics = row["metrics"]
        top_k = row["top_k"]
        lines.append(
            " | ".join(
                [
                    row["arm"],
                    f"{metrics.get(f'hit@{top_k}', 0):.4f}",
                    f"{metrics.get('mrr', 0):.4f}",
                    f"{metrics.get(f'recall@{top_k}', 0):.4f}",
                    f"{metrics.get('context_precision', 0):.4f}",
                    f"{row['latency_seconds']:.2f}",
                ]
            )
        )
    return "\n".join(lines)


def _ensure_distractor_kb(*, force_reindex: bool) -> None:
    if not DISTRACTOR_DIR.exists():
        raise SystemExit(f"Missing distractor corpus: {DISTRACTOR_DIR}")
    for path in sorted(DISTRACTOR_DIR.glob("*.md")):
        result = service.ingest_file(
            path,
            title=path.stem,
            strategy=ChunkStrategy.HEADING,
            force_reindex=force_reindex,
            force_knowledge_base_id=DISTRACTOR_KB,
            force_knowledge_base_name="干扰知识库",
            force_knowledge_base_description="旅行/烹饪等无关语料，用于查询路由对比实验",
        )
        print(
            f"[distractor] {path.name} -> kb={result.get('knowledge_base_id')} "
            f"chunks={result.get('chunk_count')} duplicate={result.get('duplicate')}",
            flush=True,
        )


def _run_arm(
    runner: EvaluationRunner,
    *,
    arm_name: str,
    dataset: Path,
    top_k: int,
    **kwargs: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    report = runner.run(dataset, top_k=top_k, include_generation=False, **kwargs)
    latency = time.perf_counter() - started
    arm_report = {
        "arm": arm_name,
        "top_k": top_k,
        "latency_seconds": round(latency, 3),
        "metrics": report["metrics"],
        "example_count": report["example_count"],
        "config": {
            "retrieval_mode": report.get("retrieval_mode"),
            "use_rerank": report.get("use_rerank"),
            "auto_route": report.get("auto_route"),
            "use_query_expansion": report.get("use_query_expansion"),
            "use_retrieval_retry": report.get("use_retrieval_retry"),
            "search_all_knowledge_bases": report.get("search_all_knowledge_bases"),
            "knowledge_base_ids": report.get("knowledge_base_ids"),
        },
    }
    print(f"[{arm_name}] {json.dumps(report['metrics'], ensure_ascii=False)} ({latency:.2f}s)", flush=True)
    return arm_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval / query-opt ablation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "golden_set.seed.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--kb-id", default=SEED_KB)
    parser.add_argument("--suite", choices=["retrieval", "query_opt", "all"], default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "storage" / "ablation_report.json",
    )
    parser.add_argument(
        "--force-reindex-distractor",
        action="store_true",
        help="Re-ingest distractor KB even if duplicate",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"Dataset not found: {args.dataset}\n"
            "Run: python scripts/seed_research_kb.py --force-reindex"
        )

    runner = EvaluationRunner(
        service.retriever,
        ask_service=service,
        eval_service=service,
    )
    reports: list[dict[str, Any]] = []

    if args.suite in {"retrieval", "all"}:
        print("=== suite: retrieval ===")
        for arm_name, mode, use_rerank in RETRIEVAL_ARMS:
            reports.append(
                _run_arm(
                    runner,
                    arm_name=arm_name,
                    dataset=args.dataset,
                    top_k=args.top_k,
                    use_rerank=use_rerank,
                    retrieval_mode=mode,
                    knowledge_base_ids=[args.kb_id],
                )
            )

    if args.suite in {"query_opt", "all"}:
        print("=== suite: query_opt (base=hybrid+rerank) ===")
        _ensure_distractor_kb(force_reindex=args.force_reindex_distractor)
        # no_opt: original query only, no route/expand/retry, search ALL KBs (seed+distractor noise)
        reports.append(
            _run_arm(
                runner,
                arm_name="no_opt",
                dataset=args.dataset,
                top_k=args.top_k,
                use_rerank=True,
                retrieval_mode="hybrid",
                auto_route=False,
                use_query_expansion=False,
                use_retrieval_retry=False,
                search_all_knowledge_bases=True,
            )
        )
        # full_opt: LLM route + rewrite/sub/step-back + low-confidence retry
        reports.append(
            _run_arm(
                runner,
                arm_name="full_opt",
                dataset=args.dataset,
                top_k=args.top_k,
                use_rerank=True,
                retrieval_mode="hybrid",
                auto_route=True,
                use_query_expansion=True,
                use_retrieval_retry=True,
                search_all_knowledge_bases=False,
                knowledge_base_ids=None,
            )
        )

    payload = {
        "dataset_path": str(args.dataset),
        "suite": args.suite,
        "top_k": args.top_k,
        "arms": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(_format_table(reports))
    print(f"\n[ablation] report -> {args.output}")


if __name__ == "__main__":
    main()
