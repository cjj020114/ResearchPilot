"""Run full_opt answer generation + extended RAGAS metrics on the seed golden set.

Usage:
  python scripts/run_ragas_eval.py
  python scripts/run_ragas_eval.py --top-k 5 --limit 16
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
from backend.app.evaluation.metrics import (
    aggregate,
    context_precision as retrieval_context_precision,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from backend.app.evaluation.ragas_metrics import score_generation_rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _contexts_from_ask(payload: dict[str, Any]) -> list[str]:
    chunks = getattr(service.index, "chunks", {}) or {}
    contexts: list[str] = []
    for cite in payload.get("citations") or []:
        chunk_id = str(cite.get("chunk_id") or "")
        chunk = chunks.get(chunk_id)
        if chunk is not None and str(chunk.text).strip():
            contexts.append(str(chunk.text))
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="full_opt + RAGAS evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "golden_set.seed.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "storage" / "ragas_report.json",
    )
    parser.add_argument(
        "--answers-cache",
        type=Path,
        default=ROOT / "storage" / "ragas_answers_cache.json",
        help="Cache full_opt answers so scoring can be retried without re-asking",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip ask(); score from --answers-cache",
    )
    parser.add_argument(
        "--prefer",
        choices=["auto", "ragas", "llm"],
        default="auto",
        help="Scoring backend preference",
    )
    args = parser.parse_args()

    if not args.dataset.exists() and not args.score_only:
        raise SystemExit(f"Dataset not found: {args.dataset}")

    rows: list[dict[str, Any]] = []
    hits: list[float] = []
    recalls: list[float] = []
    rrs: list[float] = []
    precisions: list[float] = []

    if args.score_only:
        if not args.answers_cache.exists():
            raise SystemExit(f"Answers cache not found: {args.answers_cache}")
        cached = json.loads(args.answers_cache.read_text(encoding="utf-8"))
        rows = list(cached.get("rows") or [])
        print(f"[ragas] score-only from cache examples={len(rows)}", flush=True)
        for row in rows:
            hits.append(float(row.get("hit") or 0.0))
            recalls.append(float(row.get("recall") or 0.0))
            rrs.append(float(row.get("reciprocal_rank") or 0.0))
            precisions.append(float(row.get("retrieval_context_precision") or 0.0))
    else:
        examples = _load_jsonl(args.dataset)
        if args.limit and args.limit > 0:
            examples = examples[: args.limit]

        print(
            f"[ragas] examples={len(examples)} top_k={args.top_k} pipeline=full_opt(ask)",
            flush=True,
        )
        for index, example in enumerate(examples, start=1):
            question = str(example["question"])
            relevant_ids = [str(item) for item in example.get("relevant_chunk_ids", [])]
            started = time.perf_counter()
            payload = service.ask(
                question,
                top_k=args.top_k,
                use_rerank=True,
                auto_route=True,
            )
            elapsed = time.perf_counter() - started
            retrieved_ids = [
                str(item.get("chunk_id") or "")
                for item in (payload.get("citations") or [])
                if item.get("chunk_id")
            ]
            if not retrieved_ids:
                retrieved_ids = [
                    str(item.get("chunk_id") or "")
                    for item in (payload.get("trace") or [])
                    if item.get("chunk_id")
                ]

            hit = hit_at_k(retrieved_ids, relevant_ids, args.top_k)
            recall = recall_at_k(retrieved_ids, relevant_ids, args.top_k)
            rr = reciprocal_rank(retrieved_ids, relevant_ids)
            precision = retrieval_context_precision(retrieved_ids, relevant_ids, args.top_k)
            hits.append(hit)
            recalls.append(recall)
            rrs.append(rr)
            precisions.append(precision)

            row = {
                "question": question,
                "answer": str(payload.get("answer") or ""),
                "contexts": _contexts_from_ask(payload),
                "reference_answer": example.get("answer"),
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "hit": hit,
                "recall": recall,
                "reciprocal_rank": rr,
                "retrieval_context_precision": precision,
                "routing": payload.get("routing"),
                "query_plan": payload.get("query_plan"),
                "need_selection": payload.get("need_selection"),
                "latency_seconds": round(elapsed, 3),
            }
            rows.append(row)
            args.answers_cache.parent.mkdir(parents=True, exist_ok=True)
            args.answers_cache.write_text(
                json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{index}/{len(examples)}] hit={hit:.0f} rr={rr:.2f} "
                f"answer_len={len(row['answer'])} contexts={len(row['contexts'])} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    print(f"[ragas] scoring generation metrics prefer={args.prefer}...", flush=True)
    generation = score_generation_rows(rows, prefer=args.prefer)
    report = {
        "dataset_path": str(args.dataset),
        "pipeline": "full_opt_ask",
        "top_k": args.top_k,
        "example_count": len(rows),
        "retrieval_metrics": {
            f"hit@{args.top_k}": round(aggregate(hits), 4),
            "mrr": round(aggregate(rrs), 4),
            f"recall@{args.top_k}": round(aggregate(recalls), 4),
            "context_precision": round(aggregate(precisions), 4),
        },
        "generation": generation,
        "metrics": {
            **{
                f"hit@{args.top_k}": round(aggregate(hits), 4),
                "mrr": round(aggregate(rrs), 4),
                f"recall@{args.top_k}": round(aggregate(recalls), 4),
            },
            **(generation.get("aggregate") or {}),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"[ragas] backend={generation.get('backend')} {generation.get('message')}", flush=True)
    print(f"[ragas] report -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
