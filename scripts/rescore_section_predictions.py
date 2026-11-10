#!/usr/bin/env python3
"""Rescore saved extraction predictions against a revised JSONL Gold dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_from_sample(sample: dict) -> str:
    return sample["messages"][1]["content"].split("Q: ", 1)[-1].rsplit("\nA:", 1)[0]


def score(predicted: list[tuple[str, str]], gold: list[tuple[str, str]]) -> tuple[int, int, int]:
    predicted_counter = Counter(predicted)
    gold_counter = Counter(gold)
    tp = sum((predicted_counter & gold_counter).values())
    return tp, sum(predicted_counter.values()) - tp, sum(gold_counter.values()) - tp


def safe_div(left: int, right: int) -> float:
    return left / right if right else 0.0


def finalize(counter: Counter) -> dict:
    precision = safe_div(counter["tp"], counter["tp"] + counter["fp"])
    recall = safe_div(counter["tp"], counter["tp"] + counter["fn"])
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "cases": counter["cases"],
        "tp": counter["tp"],
        "fp": counter["fp"],
        "fn": counter["fn"],
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "case_exact_rate": round(safe_div(counter["case_exact"], counter["cases"]), 6),
        "json_valid_rate": round(safe_div(counter["json_valid"], counter["cases"]), 6),
        "schema_valid_rate": round(safe_div(counter["schema_valid"], counter["cases"]), 6),
        "human_review_rate": round(safe_div(counter["human_review"] , counter["cases"]), 6),
        "long_extraction_recall": round(
            safe_div(counter["long_tp"], counter["long_tp"] + counter["long_fn"]), 6
        ),
        "grounding_rate": round(safe_div(counter["grounded"], counter["predicted"]), 6),
        "hit_max_new_tokens_cases": counter["hit_max_new_tokens"],
        "validation_dropped_fields": counter["validation_dropped_fields"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gold-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    gold_rows = load_jsonl(args.gold_data)
    gold_by_id = {row["id"]: row for row in gold_rows}
    if set(gold_by_id) != {row["id"] for row in predictions}:
        raise ValueError("prediction IDs and revised Gold IDs do not match")

    overall: Counter = Counter()
    by_document: dict[str, Counter] = defaultdict(Counter)
    rescored: list[dict] = []
    for row in predictions:
        gold_row = gold_by_id[row["id"]]
        gold_payload = json.loads(gold_row["messages"][-1]["content"])
        gold = [
            (item["slot_attributes"]["field_name"], item["slot"])
            for item in gold_payload["extractions"]
        ]
        predicted = [tuple(pair) for pair in row["predicted_pairs"]]
        tp, fp, fn = score(predicted, gold)
        long_gold = [pair for pair in gold if len(pair[1]) >= 100]
        long_tp, _, long_fn = score(predicted, long_gold)
        source = source_from_sample(gold_row)
        values = Counter(
            cases=1,
            tp=tp,
            fp=fp,
            fn=fn,
            case_exact=int(Counter(predicted) == Counter(gold)),
            json_valid=int(row.get("strict_json_valid", row.get("parse_error") is None)),
            schema_valid=int(row.get("schema_valid", False)),
            human_review=int(row.get("human_review_required", False)),
            long_tp=long_tp,
            long_fn=long_fn,
            grounded=sum(bool(value) and value in source for _, value in predicted),
            predicted=len(predicted),
            hit_max_new_tokens=int(row.get("hit_max_new_tokens", False)),
            validation_dropped_fields=len(row.get("validation_errors", [])),
        )
        overall.update(values)
        by_document[row["document_type"]].update(values)
        rescored_row = dict(row)
        rescored_row["gold"] = gold_payload
        rescored_row["rescored"] = {"tp": tp, "fp": fp, "fn": fn}
        rescored.append(rescored_row)

    report = {
        "predictions": str(args.predictions),
        "gold_data": str(args.gold_data),
        "overall": finalize(overall),
        "by_document_type": {
            document_type: finalize(counter)
            for document_type, counter in sorted(by_document.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rescored),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
