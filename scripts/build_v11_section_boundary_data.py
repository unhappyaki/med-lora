#!/usr/bin/env python3
"""Build v11 data by repairing semantic boundaries and adding residual hard cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from section_schema_prompt import build_section_prompt_v2


SYSTEM_PROMPT = "You are a helpful assistant that responds in JSON format."


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_from_sample(sample: dict) -> str:
    content = sample["messages"][1]["content"]
    return content.split("Q: ", 1)[-1].rsplit("\nA:", 1)[0]


def extraction(field: str, value: str) -> dict:
    return {"slot": value, "slot_attributes": {"field_name": field}}


def payload_from_sample(sample: dict) -> list[dict]:
    return json.loads(sample["messages"][-1]["content"])["extractions"]


def normalized_heading(heading: str) -> str:
    return heading.strip().strip("【】").rstrip(":：").strip()


def heading_maps(document_schema: dict) -> tuple[dict[str, str | None], list[str]]:
    fields = (
        document_schema.get("header_fields", [])
        + document_schema.get("core_fields", [])
        + document_schema.get("optional_fields", [])
    )
    mapping: dict[str, str | None] = {normalized_heading(field): field for field in fields}
    for raw, canonical in document_schema.get("aliases", {}).items():
        mapping[normalized_heading(raw)] = canonical
    boundary = [normalized_heading(value) for value in document_schema.get("boundary_only_headings", [])]
    for raw in boundary:
        mapping[raw] = None
    return mapping, boundary


def make_line_heading_pattern(mapping: dict[str, str | None]) -> re.Pattern[str]:
    alternatives = sorted(mapping, key=len, reverse=True)
    joined = "|".join(re.escape(value) for value in alternatives)
    return re.compile(rf"(?m)^[ \t]*(?:【)?(?P<heading>{joined})(?:】)?[ \t]*[:：]")


def split_embedded_peer_headings(
    *,
    field: str,
    value: str,
    mapping: dict[str, str | None],
    pattern: re.Pattern[str],
) -> list[tuple[str, str]]:
    """Split only explicit line-start headings embedded in an old slot value."""
    matches = list(pattern.finditer(value))
    if not matches:
        return [(field, value)]
    first = matches[0]
    result: list[tuple[str, str]] = []
    prefix = value[: first.start()].rstrip(" \t\r\n")
    if prefix:
        result.append((field, prefix))
    elif first.start() > 0:
        result.append((field, value[: first.start()]))
    for index, match in enumerate(matches):
        canonical = mapping[normalized_heading(match.group("heading"))]
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        segment = value[match.end():end].strip(" \t\r\n")
        if canonical and segment:
            result.append((canonical, segment))
    return result


def trim_boundary_only(value: str, boundary: list[str]) -> str:
    if not boundary:
        return value
    alternatives = "|".join(re.escape(item) for item in sorted(boundary, key=len, reverse=True))
    match = re.search(rf"(?m)^[ \t]*(?:【)?(?:{alternatives})(?:】)?[ \t]*[:：]", value)
    return value[: match.start()].rstrip(" \t\r\n") if match else value


def recover_preop_recorder(source: str) -> str | None:
    match = re.search(r"记录者签名[ \t]*[:：]", source)
    if not match:
        return None
    remainder = source[match.end():]
    boundary = re.search(
        r"(?m)(?:\n[ \t]*|[ \t]+)(?:记录日期|主持者签名|主持人签名|审核签名|医师签名)[ \t]*[:：]",
        remainder,
    )
    value = remainder[: boundary.start()] if boundary else remainder
    value = value.strip(" \t\r\n")
    return value or None


def repair_real_sample(sample: dict, schema: dict) -> tuple[dict, list[dict]]:
    row = deepcopy(sample)
    metadata = row.get("metadata") or {}
    document_type = metadata["document_type"]
    document_schema = schema[document_type]
    mapping, boundary = heading_maps(document_schema)
    line_pattern = make_line_heading_pattern(mapping)
    source = source_from_sample(row)
    old_items = payload_from_sample(row)
    candidates: list[tuple[str, str]] = []
    changes: list[dict] = []

    for item in old_items:
        field = item["slot_attributes"]["field_name"]
        old_value = item["slot"]
        value = trim_boundary_only(old_value, boundary)
        if value != old_value:
            changes.append(
                {
                    "id": row["id"],
                    "kind": "trim_boundary_only",
                    "field": field,
                    "old_length": len(old_value),
                    "new_length": len(value),
                }
            )
        pieces = [(field, value)]
        if document_type == "首次病程记录" and value:
            pieces = split_embedded_peer_headings(
                field=field,
                value=value,
                mapping=mapping,
                pattern=line_pattern,
            )
            if len(pieces) > 1 or pieces[0] != (field, value):
                changes.append(
                    {
                        "id": row["id"],
                        "kind": "split_peer_heading",
                        "field": field,
                        "pieces": [piece_field for piece_field, _ in pieces],
                    }
                )
        candidates.extend((piece_field, piece_value) for piece_field, piece_value in pieces if piece_value)

    if document_type == "术前讨论":
        recorder = recover_preop_recorder(source)
        if recorder and not any(field == "记录者" for field, _ in candidates):
            candidates.append(("记录者", recorder))
            changes.append(
                {
                    "id": row["id"],
                    "kind": "recover_recorder_after_boundary_heading",
                    "field": "记录者",
                    "new_length": len(recorder),
                }
            )

    # Preserve order while removing exact duplicates. Prefer the corrected split value
    # when an old field swallowed a peer section and yielded a second occurrence.
    output: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for field, value in candidates:
        if (field, value) in seen_pairs:
            continue
        if value not in source:
            changes.append(
                {
                    "id": row["id"],
                    "kind": "rejected_ungrounded_repair",
                    "field": field,
                    "length": len(value),
                }
            )
            continue
        seen_pairs.add((field, value))
        output.append((field, value))

    row["messages"][-1]["content"] = json.dumps(
        {"extractions": [extraction(field, value) for field, value in output]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    row["metadata"] = {
        **metadata,
        "label_revision": "v11_semantic_boundary",
        "label_revision_changes": len(changes),
    }
    return row, changes


def synthetic_sample(
    *,
    family: str,
    split: str,
    index: int,
    document_type: str,
    source: str,
    pairs: list[tuple[str, str]],
    schema: dict,
    edge_tags: list[str],
) -> dict:
    digest = hashlib.sha1(f"v11|{family}|{split}|{index}|{source}".encode("utf-8")).hexdigest()
    return {
        "id": f"v11-section-synthetic-{digest[:20]}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_section_prompt_v2(source, document_type, schema)},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"extractions": [extraction(field, value) for field, value in pairs]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
        "metadata": {
            "document_type": document_type,
            "source_type": "synthetic_section_boundary_v11_residual",
            "synthetic_family": family,
            "synthetic_index": index,
            "source_chars": len(source),
            "field_count": len(pairs),
            "html_removed": True,
            "template_braces_removed": True,
            "deidentified": True,
            "strict_source_traceable": True,
            "canonical_section_schema": "v1",
            "section_schema_prompt": "v2_dedicated",
            "task_mode": "large_section_boundary",
            "training_role": split,
            "label_revision": "v11_semantic_boundary",
            "edge_tags": edge_tags,
        },
    }


def first_course_residual_sample(index: int, split: str, schema: dict) -> dict:
    year = 2026 if split == "train" else 2027
    age = 30 + index % 48
    disease = ["社区获得性肺炎", "胆囊结石伴胆囊炎", "脑梗死", "子宫腺肌症"][index % 4]
    feature = (
        f"1、患者{age}岁，因相关症状{index % 8 + 1}天入院。\n"
        "2、既往史及个人史：无特殊。\n"
        "3、现病史：起病后症状持续，门诊检查后收入院。\n"
        "4、体格检查：生命体征平稳，心肺腹查体未见明显异常。\n"
        "5、专科情况：神志清楚，专科查体与当前疾病表现一致。\n"
        f"6、辅助检查：血常规及影像学资料支持{disease}，仍需动态复查。"
    )
    diagnosis = disease
    evidence = f"1、临床表现符合；2、专科查体支持；3、辅助检查提示{disease}。"
    plan = "1、完善相关检查；2、给予对症及病因治疗；3、动态观察病情变化。"
    source = (
        f"病历特点：\n{feature}\n"
        f"初步诊断：\n{diagnosis}\n"
        f"诊断依据：\n{evidence}\n"
        f"诊疗计划：\n{plan}\n"
        f"医师签名：医师{chr(65 + index % 6)}\n"
        f"记录日期：{year}年{index % 12 + 1:02d}月{index % 27 + 1:02d}日"
    )
    return synthetic_sample(
        family="first_course_numbered_subsections",
        split=split,
        index=index,
        document_type="首次病程记录",
        source=source,
        pairs=[("病历特点", feature), ("初步诊断", diagnosis), ("诊断依据", evidence), ("诊疗计划", plan)],
        schema=schema,
        edge_tags=[
            "feature_keeps_numbered_exam_subsections",
            "auxiliary_exam_is_not_peer_heading",
            "explicit_peer_heading_stop",
            "boundary_only_signature",
        ],
    )


def first_course_tcm_peer_sample(index: int, split: str, schema: dict) -> dict:
    year = 2026 if split == "train" else 2027
    feature = (
        f"1、患者{40 + index % 32}岁，因反复不适入院。\n"
        "2、体格检查及辅助检查已在本段完整概括，病情暂稳定。"
    )
    tcm = (
        "患者症见相关不适，结合舌脉及整体表现，辨病属气滞血瘀，"
        "病机以气机不畅、瘀血内阻为主。"
    )
    western = "1、临床症状支持；2、查体及辅助检查与当前判断一致。"
    plan = "完善检查，予对症处理，动态复查并根据结果调整方案。"
    raw_heading = ["中医辨病辨证依据与鉴别诊断", "中医辨病辨证依据"][index % 2]
    source = (
        f"病例特点：\n{feature}\n"
        "诊断及鉴别诊断：\n"
        f"{raw_heading}：\n{tcm}\n"
        f"西医辨病辨证依据与鉴别诊断：\n{western}\n"
        f"诊疗计划：\n{plan}\n"
        f"记录日期：{year}年{index % 12 + 1:02d}月{index % 27 + 1:02d}日"
    )
    return synthetic_sample(
        family="first_course_explicit_tcm_peer",
        split=split,
        index=index,
        document_type="首次病程记录",
        source=source,
        pairs=[("病历特点", feature), ("中医辨病辨证依据", tcm), ("诊断依据", western), ("诊疗计划", plan)],
        schema=schema,
        edge_tags=[
            "feature_stops_before_explicit_tcm_peer",
            "wrapper_heading_is_boundary_only",
            "alias_normalization",
            "record_date_is_boundary_only",
        ],
    )


def preop_signature_sample(index: int, split: str, schema: dict) -> dict:
    year = 2026 if split == "train" else 2027
    day = index % 27 + 1
    discussion = (
        "医师乙汇报病史及检查结果，患者诊断较明确，具备择期手术条件。\n"
        "医师甲意见：同意手术方案，术中注意止血及保护邻近组织，术后预防感染。"
    )
    summary = "综合讨论，同意按计划手术；术前完善准备并向患者及家属充分说明风险。"
    recorder = f"医师{chr(67 + index % 4)}"
    if index % 2:
        signature_line = f"主持者签名：医师甲 记录者签名：\n{recorder}"
    else:
        signature_line = f"主持者签名：医师甲\n记录者签名：{recorder}"
    source = (
        f"讨论时间：{year}年11月{day:02d}日 09时00分\n"
        "讨论地点：医师办公室\n"
        "主持者：医师甲\n"
        "参加讨论人员：医师甲、医师乙、医师丙\n"
        f"讨论记录：\n{discussion}\n"
        f"主持人小结意见：\n{summary}\n"
        f"{signature_line}\n审核签名：医师甲"
    )
    return synthetic_sample(
        family="preop_boundary_signature_then_recorder",
        split=split,
        index=index,
        document_type="术前讨论",
        source=source,
        pairs=[
            ("讨论时间", f"{year}年11月{day:02d}日 09时00分"),
            ("讨论地点", "医师办公室"),
            ("主持人", "医师甲"),
            ("参加人员", "医师甲、医师乙、医师丙"),
            ("讨论记录", discussion),
            ("主持人小结意见", summary),
            ("记录者", recorder),
        ],
        schema=schema,
        edge_tags=[
            "summary_stops_before_host_signature",
            "host_signature_is_not_host_field",
            "recorder_after_boundary_heading",
            "audit_signature_is_boundary_only",
        ],
    )


def validate(rows: list[dict], schema: dict) -> list[dict]:
    errors: list[dict] = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen_ids:
            errors.append({"id": row_id, "error": "duplicate_id"})
        seen_ids.add(row_id)
        source = source_from_sample(row)
        source_key = hashlib.sha1(re.sub(r"\s+", "", source).encode("utf-8")).hexdigest()
        if source_key in seen_sources:
            errors.append({"id": row_id, "error": "duplicate_source"})
        seen_sources.add(source_key)
        document_type = row["metadata"]["document_type"]
        allowed = set(schema[document_type].get("header_fields", [])) | set(
            schema[document_type].get("core_fields", [])
        ) | set(schema[document_type].get("optional_fields", []))
        field_counts: Counter[str] = Counter()
        for item in payload_from_sample(row):
            field = item["slot_attributes"]["field_name"]
            value = item["slot"]
            field_counts[field] += 1
            if field not in allowed:
                errors.append({"id": row_id, "error": "unknown_field", "field": field})
            if not value or value not in source:
                errors.append({"id": row_id, "error": "ungrounded", "field": field})
        for field, count in field_counts.items():
            if count > 1:
                errors.append({"id": row_id, "error": "duplicate_field", "field": field, "count": count})
    return errors


def source_fingerprint(row: dict) -> str:
    compact = re.sub(r"\s+", "", source_from_sample(row))
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10-dir", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--new-train-per-family", type=int, default=40)
    parser.add_argument("--new-validation-per-family", type=int, default=8)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))["documents"]
    inputs = {
        "train": load_jsonl(args.v10_dir / "v10_dedicated_section_train.jsonl"),
        "validation": load_jsonl(args.v10_dir / "v10_dedicated_section_validation.jsonl"),
        "holdout": load_jsonl(args.v10_dir / "v10_real_holdout_prompt_v2.jsonl"),
    }
    repaired: dict[str, list[dict]] = {}
    changed_rows: dict[str, list[dict]] = {}
    change_log: list[dict] = []
    for split, rows in inputs.items():
        repaired_rows: list[dict] = []
        changed: list[dict] = []
        for row in rows:
            if row["metadata"].get("source_type") == "section_real_reprompt_v2":
                new_row, changes = repair_real_sample(row, schema)
            else:
                new_row, changes = deepcopy(row), []
            repaired_rows.append(new_row)
            if changes:
                changed.append(new_row)
                change_log.extend({"split": split, **change} for change in changes)
        repaired[split] = repaired_rows
        changed_rows[split] = changed

    families = [first_course_residual_sample, first_course_tcm_peer_sample, preop_signature_sample]
    new_train = [
        family(index, "train", schema)
        for family in families
        for index in range(args.new_train_per_family)
    ]
    new_validation = [
        family(10_000 + index, "validation", schema)
        for family in families
        for index in range(args.new_validation_per_family)
    ]
    train = repaired["train"] + new_train
    validation = repaired["validation"] + new_validation
    rng = random.Random(20261109)
    rng.shuffle(train)
    rng.shuffle(validation)

    patch_train = changed_rows["train"] + new_train
    patch_validation = changed_rows["validation"] + new_validation
    rng.shuffle(patch_train)
    rng.shuffle(patch_validation)

    split_rows = {
        "train": train,
        "validation": validation,
        "holdout": repaired["holdout"],
        "patch_train": patch_train,
        "patch_validation": patch_validation,
    }
    errors = [
        {"split": split, **error}
        for split, rows in split_rows.items()
        for error in validate(rows, schema)
    ]
    fingerprints = {
        split: {source_fingerprint(row) for row in rows}
        for split, rows in {
            "train": train,
            "validation": validation,
            "holdout": repaired["holdout"],
        }.items()
    }
    overlaps = {
        "train_validation": len(fingerprints["train"] & fingerprints["validation"]),
        "train_holdout": len(fingerprints["train"] & fingerprints["holdout"]),
        "validation_holdout": len(fingerprints["validation"] & fingerprints["holdout"]),
    }
    for pair, count in overlaps.items():
        if count:
            errors.append({"error": "source_overlap", "pair": pair, "count": count})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "v11_dedicated_section_train.jsonl", train)
    write_jsonl(args.output_dir / "v11_dedicated_section_validation.jsonl", validation)
    write_jsonl(args.output_dir / "v11_real_holdout_relabelled.jsonl", repaired["holdout"])
    write_jsonl(args.output_dir / "v11_boundary_patch_train.jsonl", patch_train)
    write_jsonl(args.output_dir / "v11_boundary_patch_validation.jsonl", patch_validation)
    (args.output_dir / "label_change_log.json").write_text(
        json.dumps(change_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    change_kinds = Counter(item["kind"] for item in change_log)
    summary = {
        "strategy": "repair old semantic boundaries, keep numbered subsections inside parent, and add explicit residual counterexamples",
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "changed_real_rows": {split: len(rows) for split, rows in changed_rows.items()},
        "change_kinds": dict(change_kinds),
        "new_synthetic": {
            "train": len(new_train),
            "validation": len(new_validation),
            "families": [family.__name__ for family in families],
        },
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
