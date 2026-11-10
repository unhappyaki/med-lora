#!/usr/bin/env python3
"""Build the small, targeted V13.1 patch after the V13 holdout analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from build_v11_section_boundary_data import load_jsonl, source_from_sample, write_jsonl
from build_v12_first_course_patch_data import payload_from_sample


def fingerprint(row: dict) -> str:
    return hashlib.sha1(re.sub(r"\s+", "", source_from_sample(row)).encode("utf-8")).hexdigest()


def document_type(row: dict) -> str:
    return row["metadata"]["document_type"]


def fields(row: dict) -> set[str]:
    return {item["slot_attributes"]["field_name"] for item in payload_from_sample(row)}


def max_slot_length(row: dict) -> int:
    return max((len(item["slot"]) for item in payload_from_sample(row)), default=0)


def output_chars(row: dict) -> int:
    return len(row["messages"][-1]["content"])


def clone_with_bucket(row: dict, bucket: str, split: str) -> dict:
    item = deepcopy(row)
    item["metadata"] = dict(item.get("metadata", {}))
    item["metadata"].update(
        {
            "training_role": split,
            "v131_bucket": bucket,
            "label_revision": "v13.1_targeted_boundary_patch",
        }
    )
    return item


def unique_pool(*groups: list[dict]) -> list[dict]:
    by_fp: dict[str, dict] = {}
    for group in groups:
        for row in group:
            by_fp.setdefault(fingerprint(row), row)
    return list(by_fp.values())


def take_ranked(candidates: list[dict], used: set[str], count: int, key) -> list[dict]:
    selected = []
    for row in sorted(candidates, key=key, reverse=True):
        fp = fingerprint(row)
        if fp in used:
            continue
        selected.append(row)
        used.add(fp)
        if len(selected) == count:
            return selected
    raise ValueError(f"needed {count} rows, found {len(selected)}")


def take_random(candidates: list[dict], used: set[str], count: int, seed: int) -> list[dict]:
    available = [row for row in candidates if fingerprint(row) not in used]
    random.Random(seed).shuffle(available)
    if len(available) < count:
        raise ValueError(f"needed {count} random rows, found {len(available)}")
    selected = available[:count]
    used.update(fingerprint(row) for row in selected)
    return selected


def preop_boundary_score(row: dict) -> tuple[int, int, str]:
    row_fields = fields(row)
    source = source_from_sample(row)
    score = 0
    score += 220 if "讨论记录" in row_fields else 0
    score += 160 if "主持人小结意见" in row_fields else 0
    score += 130 if "记录者" in row_fields else 0
    score += 90 if "主持人" in row_fields else 0
    score += 80 if any(token in source for token in ("男", "女", "岁", "医师", "主任")) else 0
    score += 60 if "讨论记录" in row_fields and len(source) >= 1200 else 0
    score += min(source.count("\n"), 60)
    return score, len(source), row["id"]


def verbatim_score(row: dict) -> tuple[int, int, str]:
    source = source_from_sample(row)
    tags = set(row.get("metadata", {}).get("edge_tags", []))
    score = 0
    score += 500 if row.get("metadata", {}).get("synthetic_family") == "first_course_verbatim_character_traps" else 0
    score += 100 if "verbatim_whitespace" in tags or "grounding_must_not_normalize" in tags else 0
    score += source.count("\n\n") * 25
    score += source.count("  ") * 10
    score += 30 if re.search(r"[（(][^\n]{0,80}[）)]", source) else 0
    score += 25 if re.search(r"\d\n\d", source) else 0
    score += min(max_slot_length(row) // 80, 40)
    return score, max_slot_length(row), row["id"]


def json_stability_score(row: dict) -> tuple[int, int, str]:
    source = source_from_sample(row)
    field_count = len(payload_from_sample(row))
    special = sum(source.count(char) for char in ('"', "（", "）", "(", ")", "\\", "\n"))
    score = field_count * 200 + min(output_chars(row) // 50, 100) + min(special, 100)
    return score, output_chars(row), row["id"]


def validate(rows: list[dict], schema: dict) -> list[dict]:
    errors = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    for row in rows:
        if row["id"] in seen_ids:
            errors.append({"id": row["id"], "error": "duplicate_id"})
        seen_ids.add(row["id"])
        fp = fingerprint(row)
        if fp in seen_sources:
            errors.append({"id": row["id"], "error": "duplicate_source"})
        seen_sources.add(fp)
        source = source_from_sample(row)
        try:
            payload = json.loads(row["messages"][-1]["content"])
        except json.JSONDecodeError as exc:
            errors.append({"id": row["id"], "error": "invalid_gold_json", "detail": str(exc)})
            continue
        if set(payload) != {"extractions"} or not isinstance(payload["extractions"], list):
            errors.append({"id": row["id"], "error": "invalid_gold_schema"})
            continue
        document_schema = schema[document_type(row)]
        allowed = set(document_schema.get("header_fields", [])) | set(document_schema.get("core_fields", [])) | set(document_schema.get("optional_fields", []))
        seen_fields: set[str] = set()
        for item in payload["extractions"]:
            field = item["slot_attributes"]["field_name"]
            value = item["slot"]
            if field not in allowed:
                errors.append({"id": row["id"], "error": "unknown_field", "field": field})
            if field in seen_fields:
                errors.append({"id": row["id"], "error": "duplicate_field", "field": field})
            seen_fields.add(field)
            if not value or value not in source:
                errors.append({"id": row["id"], "error": "ungrounded_gold", "field": field})
    return errors


def build_split(
    *,
    split: str,
    pool: list[dict],
    used: set[str],
    preop_count: int,
    verbatim_count: int,
    json_each: int,
    rehearsal_each: int,
) -> list[dict]:
    preop_pool = [row for row in pool if document_type(row) == "术前讨论"]
    first_pool = [row for row in pool if document_type(row) == "首次病程记录"]
    selected: list[dict] = []

    rows = take_ranked(preop_pool, used, preop_count, preop_boundary_score)
    selected.extend(clone_with_bucket(row, "preop_parent_child_boundary", split) for row in rows)

    rows = take_ranked(first_pool, used, verbatim_count, verbatim_score)
    selected.extend(clone_with_bucket(row, "first_course_verbatim_copy", split) for row in rows)

    for doc, candidates in (("术前讨论", preop_pool), ("首次病程记录", first_pool)):
        rows = take_ranked(candidates, used, json_each, json_stability_score)
        selected.extend(clone_with_bucket(row, "long_json_stability", split) for row in rows)

    for offset, (doc, candidates) in enumerate((("术前讨论", preop_pool), ("首次病程记录", first_pool))):
        rows = take_random(candidates, used, rehearsal_each, 20261110 + offset + (100 if split == "validation" else 0))
        selected.extend(clone_with_bucket(row, "balanced_rehearsal", split) for row in rows)

    random.Random(20261110 if split == "train" else 20261111).shuffle(selected)
    return selected


def summarize(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "document_types": dict(Counter(document_type(row) for row in rows)),
        "buckets": dict(Counter(row["metadata"]["v131_bucket"] for row in rows)),
        "field_frequency": dict(
            Counter(
                item["slot_attributes"]["field_name"]
                for row in rows
                for item in payload_from_sample(row)
            )
        ),
        "max_source_chars": max(len(source_from_sample(row)) for row in rows),
        "max_output_chars": max(output_chars(row) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12-dir", required=True, type=Path)
    parser.add_argument("--v13-dir", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))["documents"]
    dedicated_train = load_jsonl(args.v12_dir / "v12_dedicated_section_train.jsonl")
    dedicated_validation = load_jsonl(args.v12_dir / "v12_dedicated_section_validation.jsonl")
    v13_train = load_jsonl(args.v13_dir / "v13_balanced_patch_train.jsonl")
    v13_validation = load_jsonl(args.v13_dir / "v13_balanced_patch_validation.jsonl")
    holdout = load_jsonl(args.v13_dir / "v13_real_holdout_relabelled.jsonl")

    train_pool = unique_pool(v13_train, dedicated_train)
    validation_pool = unique_pool(v13_validation, dedicated_validation)
    holdout_fps = {fingerprint(row) for row in holdout}
    used_train = set(holdout_fps)
    train = build_split(
        split="train",
        pool=train_pool,
        used=used_train,
        preop_count=60,
        verbatim_count=50,
        json_each=12,
        rehearsal_each=60,
    )
    used_validation = set(holdout_fps) | {fingerprint(row) for row in train}
    validation = build_split(
        split="validation",
        pool=validation_pool,
        used=used_validation,
        preop_count=12,
        verbatim_count=12,
        json_each=4,
        rehearsal_each=8,
    )

    split_rows = {"train": train, "validation": validation, "holdout": holdout}
    errors = [
        {"split": split, **error}
        for split, rows in split_rows.items()
        for error in validate(rows, schema)
    ]
    fps = {split: {fingerprint(row) for row in rows} for split, rows in split_rows.items()}
    overlaps = {
        "train_validation": len(fps["train"] & fps["validation"]),
        "train_holdout": len(fps["train"] & fps["holdout"]),
        "validation_holdout": len(fps["validation"] & fps["holdout"]),
    }
    for pair, count in overlaps.items():
        if count:
            errors.append({"error": "source_overlap", "pair": pair, "count": count})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "v131_targeted_train.jsonl", train)
    write_jsonl(args.output_dir / "v131_targeted_validation.jsonl", validation)
    write_jsonl(args.output_dir / "v131_real_holdout_relabelled.jsonl", holdout)
    summary = {
        "strategy": "V13.1 targeted parent-child boundaries, verbatim copying, JSON stability, and balanced rehearsal",
        "splits": {split: summarize(rows) if split != "holdout" else {"count": len(rows), "document_types": dict(Counter(document_type(row) for row in rows))} for split, rows in split_rows.items()},
        "source_overlap": overlaps,
        "validation_errors": errors,
        "passed": not errors,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
