from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer


def chat_template(tokenizer, messages: list[dict], **kwargs):
    try:
        result = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **kwargs)
    # transformers 5.x may return a BatchEncoding from apply_chat_template,
    # while older releases return the input-id list directly.
    if kwargs.get("tokenize") and hasattr(result, "get"):
        result = result.get("input_ids")
    if kwargs.get("tokenize") and hasattr(result, "tolist"):
        result = result.tolist()
    if (
        kwargs.get("tokenize")
        and isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], list)
    ):
        result = result[0]
    return result


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = max(0, min(len(values) - 1, math.ceil(fraction * len(values)) - 1))
    return sorted(values)[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 v5 SFT 数据的真实 token 长度")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats: list[dict] = []
    for row in rows:
        prompt_ids = chat_template(
            tokenizer,
            row["messages"][:2],
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = chat_template(
            tokenizer,
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        stats.append(
            {
                "id": row["id"],
                "document_type": row["metadata"]["document_type"],
                "prompt_tokens": len(prompt_ids),
                "target_tokens": max(0, len(full_ids) - len(prompt_ids)),
                "total_tokens": len(full_ids),
            }
        )

    totals = [item["total_tokens"] for item in stats]
    summary = {
        "samples": len(stats),
        "min": min(totals, default=0),
        "p50": percentile(totals, 0.50),
        "p90": percentile(totals, 0.90),
        "p95": percentile(totals, 0.95),
        "p99": percentile(totals, 0.99),
        "max": max(totals, default=0),
        "over_4096": sum(value > 4096 for value in totals),
        "over_8192": sum(value > 8192 for value in totals),
        "over_16384": sum(value > 16384 for value in totals),
        "longest": sorted(
            stats, key=lambda item: item["total_tokens"], reverse=True
        )[: args.top],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
