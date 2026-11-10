#!/usr/bin/env python3
"""Build a balanced v13 continuation patch from the v12 error analysis.

The v12 patch improved first-course cases but contained 191 first-course versus
65 pre-op discussion samples.  This builder keeps the useful v12 examples,
adds pre-op rehearsal to a 1:1 document-type ratio, and adds synthetic
first-course examples that require character-for-character copying of awkward
punctuation, blank lines, and repeated characters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from build_v11_section_boundary_data import load_jsonl, source_from_sample, write_jsonl
from build_v12_first_course_patch_data import SYSTEM_PROMPT, extraction, payload_from_sample
from section_schema_prompt import build_section_prompt_v2


def fingerprint(row: dict) -> str:
    source = source_from_sample(row)
    return hashlib.sha1(re.sub(r"\s+", "", source).encode("utf-8")).hexdigest()


def verbatim_trap_sample(index: int, split: str, schema: dict) -> dict:
    disease = ["急性胃炎", "社区获得性肺炎", "右侧输尿管结石", "腰椎间盘突出症"][index % 4]
    traps = [
        "既往治疗方案已核对（未开具)。目前症状较前缓解。",
        "查体未见皮疹及静脉曲张。\n\n剑突下轻压痛，无反跳痛。",
        "拟进入自\n自然临床路径，继续观察症状变化。",
        "患者诉症状反复，予  对症处理后稍缓解。",
    ]
    feature = (
        f"1、患者因相关不适{index % 7 + 1}天入院，起病后症状反复，休息后可稍缓解。\n"
        f"2、既往史及个人史已核对，否认明确药物过敏史。\n"
        f"3、体格检查：T 36.5℃，P 78次/分，R 18次/分，BP 126/76mmHg。\n"
        f"4、专科情况：{traps[index % len(traps)]}\n"
        "5、辅助检查结果与当前临床表现基本一致。"
    )
    diagnosis = disease
    evidence = (
        f"1、症状和体征符合{disease}；2、辅助检查支持当前诊断；"
        f"3、已排除明显禁忌证（未见异常)。"
    )
    plan_lines = [
        "1、完善血常规、生化及相关影像学检查；",
        "2、给予对症治疗，动态观察疗效及不良反应；",
        "3\n、进入自\n自然临床路径管理，复核每日待办项目；",
        "4、向患者及家属说明病情（已告知)。",
        "5、如症状变化，及时复查并调整方案。",
    ]
    if index % 2:
        plan_lines.insert(3, "\n")
    plan = "\n".join(plan_lines)
    source = (
        f"病例特点：\n{feature}\n初步诊断：\n{diagnosis}\n"
        f"诊断依据：\n{evidence}\n诊疗计划：\n{plan}\n"
        f"医师签名：医师{chr(65 + index % 6)}\n"
        f"记录日期：2027年{index % 12 + 1:02d}月{index % 27 + 1:02d}日"
    )
    digest = hashlib.sha1(f"v13|verbatim|{split}|{index}|{source}".encode("utf-8")).hexdigest()
    return {
        "id": f"v13-verbatim-{digest[:20]}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_section_prompt_v2(source, "首次病程记录", schema)},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "extractions": [
                            extraction("病历特点", feature),
                            extraction("初步诊断", diagnosis),
                            extraction("诊断依据", evidence),
                            extraction("诊疗计划", plan),
                        ]
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
        "metadata": {
            "document_type": "首次病程记录",
            "source_type": "synthetic_section_boundary_v13_verbatim_patch",
            "synthetic_family": "first_course_verbatim_character_traps",
            "synthetic_index": index,
            "source_chars": len(source),
            "field_count": 4,
            "html_removed": True,
            "template_braces_removed": True,
            "deidentified": True,
            "strict_source_traceable": True,
            "canonical_section_schema": "v1",
            "section_schema_prompt": "v2_dedicated",
            "task_mode": "large_section_boundary",
            "training_role": split,
            "label_revision": "v13_balanced_verbatim_rehearsal",
            "edge_tags": [
                "copy_punctuation_verbatim",
                "copy_blank_lines_verbatim",
                "copy_repeated_characters_verbatim",
                "no_helpful_normalization",
            ],
        },
    }


def preop_priority(row: dict) -> tuple[int, int, str]:
    payload = payload_from_sample(row)
    fields = {item["slot_attributes"]["field_name"] for item in payload}
    source = source_from_sample(row)
    score = 0
    score += 120 if "记录者" in fields else 0
    score += 100 if "主持人小结意见" in fields else 0
    score += 70 if "主持人" in fields else 0
    score += 50 if "性别" not in fields and ("男" in source or "女" in source) else 0
    score += min(source.count("\n"), 40)
    score += min(len(source) // 200, 20)
    return score, len(source), row["id"]


def select_unique(candidates: list[dict], used: set[str], count: int) -> list[dict]:
    selected: list[dict] = []
    for row in sorted(candidates, key=preop_priority, reverse=True):
        key = fingerprint(row)
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) == count:
            return selected
    raise ValueError(f"needed {count} unique rehearsal rows, found {len(selected)}")


def validate(rows: list[dict], schema: dict) -> list[dict]:
    errors: list[dict] = []
    ids: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        if row["id"] in ids:
            errors.append({"id": row["id"], "error": "duplicate_id"})
        ids.add(row["id"])
        fp = fingerprint(row)
        if fp in sources:
            errors.append({"id": row["id"], "error": "duplicate_source"})
        sources.add(fp)
        source = source_from_sample(row)
        document_schema = schema[row["metadata"]["document_type"]]
        allowed = set(document_schema.get("header_fields", [])) | set(document_schema.get("core_fields", [])) | set(document_schema.get("optional_fields", []))
        seen: set[str] = set()
        for item in payload_from_sample(row):
            field = item["slot_attributes"]["field_name"]
            value = item["slot"]
            if field not in allowed:
                errors.append({"id": row["id"], "error": "unknown_field", "field": field})
            if field in seen:
                errors.append({"id": row["id"], "error": "duplicate_field", "field": field})
            seen.add(field)
            if not value or value not in source:
                errors.append({"id": row["id"], "error": "ungrounded", "field": field})
    return errors


def doc_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["metadata"]["document_type"] for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12-dir", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verbatim-train", type=int, default=32)
    parser.add_argument("--verbatim-validation", type=int, default=8)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))["documents"]
    patch_train = load_jsonl(args.v12_dir / "v12_first_course_patch_train.jsonl")
    patch_validation = load_jsonl(args.v12_dir / "v12_first_course_patch_validation.jsonl")
    dedicated_train = load_jsonl(args.v12_dir / "v12_dedicated_section_train.jsonl")
    dedicated_validation = load_jsonl(args.v12_dir / "v12_dedicated_section_validation.jsonl")
    holdout = load_jsonl(args.v12_dir / "v12_real_holdout_relabelled.jsonl")

    trap_train = [verbatim_trap_sample(i, "train", schema) for i in range(args.verbatim_train)]
    trap_validation = [verbatim_trap_sample(10_000 + i, "validation", schema) for i in range(args.verbatim_validation)]

    train_first = [row for row in patch_train if row["metadata"]["document_type"] == "首次病程记录"] + trap_train
    train_preop = [row for row in patch_train if row["metadata"]["document_type"] == "术前讨论"]
    train_used = {fingerprint(row) for row in train_first + train_preop}
    preop_train_candidates = [row for row in dedicated_train if row["metadata"]["document_type"] == "术前讨论"]
    train_preop += select_unique(preop_train_candidates, train_used, len(train_first) - len(train_preop))
    balanced_train = train_first + train_preop

    base_validation_first = [row for row in patch_validation if row["metadata"]["document_type"] == "首次病程记录"]
    validation_preop = [row for row in patch_validation if row["metadata"]["document_type"] == "术前讨论"]
    validation_used = {fingerprint(row) for row in validation_preop + trap_validation}
    preop_validation_candidates = [row for row in dedicated_validation if row["metadata"]["document_type"] == "术前讨论"]
    validation_preop += select_unique(
        preop_validation_candidates,
        validation_used,
        len(preop_validation_candidates) - sum(fingerprint(row) in validation_used for row in preop_validation_candidates),
    )
    target_per_type = len(validation_preop)
    if target_per_type < len(trap_validation):
        raise ValueError("not enough pre-op validation rows to retain verbatim traps")
    validation_first = base_validation_first[: target_per_type - len(trap_validation)] + trap_validation
    balanced_validation = validation_first + validation_preop

    rng = random.Random(20261110)
    rng.shuffle(balanced_train)
    rng.shuffle(balanced_validation)

    splits = {"train": balanced_train, "validation": balanced_validation, "holdout": holdout}
    errors = [
        {"split": split, **error}
        for split, rows in splits.items()
        for error in validate(rows, schema)
    ]
    fingerprints_by_split = {split: {fingerprint(row) for row in rows} for split, rows in splits.items()}
    overlaps = {
        "train_validation": len(fingerprints_by_split["train"] & fingerprints_by_split["validation"]),
        "train_holdout": len(fingerprints_by_split["train"] & fingerprints_by_split["holdout"]),
        "validation_holdout": len(fingerprints_by_split["validation"] & fingerprints_by_split["holdout"]),
    }
    for pair, count in overlaps.items():
        if count:
            errors.append({"error": "source_overlap", "pair": pair, "count": count})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "v13_balanced_patch_train.jsonl", balanced_train)
    write_jsonl(args.output_dir / "v13_balanced_patch_validation.jsonl", balanced_validation)
    write_jsonl(args.output_dir / "v13_real_holdout_relabelled.jsonl", holdout)
    summary = {
        "strategy": "balanced pre-op rehearsal plus character-for-character first-course copy traps",
        "counts": {split: len(rows) for split, rows in splits.items()},
        "document_type_counts": {split: doc_counts(rows) for split, rows in splits.items()},
        "new_synthetic": {"train": len(trap_train), "validation": len(trap_validation)},
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
