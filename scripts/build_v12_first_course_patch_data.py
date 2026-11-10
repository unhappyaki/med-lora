#!/usr/bin/env python3
"""Build a focused v12 patch for residual first-course section errors.

The patch keeps the v11 holdout untouched and adds only synthetic, traceable
counterexamples for (1) long 病历特点, (2) long 诊疗计划, and (3) whitespace
faithfulness.  It is intentionally separate from v11 so a low-learning-rate
continuation can be compared with the v11 adapter.
"""

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
from section_schema_prompt import build_section_prompt_v2


SYSTEM_PROMPT = "You are a helpful assistant that responds in JSON format."


def extraction(field: str, value: str) -> dict:
    return {"slot": value, "slot_attributes": {"field_name": field}}


def payload_from_sample(sample: dict) -> list[dict]:
    return json.loads(sample["messages"][-1]["content"])["extractions"]


def synthetic_sample(*, family: str, split: str, index: int, source: str,
                     pairs: list[tuple[str, str]], schema: dict,
                     edge_tags: list[str]) -> dict:
    digest = hashlib.sha1(f"v12|{family}|{split}|{index}|{source}".encode("utf-8")).hexdigest()
    return {
        "id": f"v12-first-course-{digest[:20]}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_section_prompt_v2(source, "首次病程记录", schema)},
            {"role": "assistant", "content": json.dumps(
                {"extractions": [extraction(field, value) for field, value in pairs]},
                ensure_ascii=False, separators=(",", ":"),
            )},
        ],
        "metadata": {
            "document_type": "首次病程记录",
            "source_type": "synthetic_section_boundary_v12_first_course_patch",
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
            "label_revision": "v12_first_course_residual_patch",
            "edge_tags": edge_tags,
        },
    }


def long_feature_sample(index: int, split: str, schema: dict) -> dict:
    age = 28 + index % 55
    disease = ["社区获得性肺炎", "胆囊结石伴胆囊炎", "脑梗死", "子宫腺肌症", "腰椎间盘突出症"][index % 5]
    extra = (
        "患者入院前自行服药效果不确切，症状变化和诱因均已逐项核对；"
        "否认近期输血及重大外伤史，预防接种情况不详，家族中无类似疾病。"
    )
    feature = (
        f"1、患者{age}岁，因反复不适{index % 9 + 1}天入院，入院时精神一般，纳食及睡眠较前下降。\n"
        "2、现病史：起病后症状时轻时重，活动或进食后可加重，休息后部分缓解；"
        "无明显意识障碍，无抽搐，无大小便失禁，曾在门诊接受对症处理。\n"
        "3、既往史及个人史：既往体健，否认高血压、糖尿病、冠心病及传染病史，"
        "否认药物和食物过敏，吸烟饮酒史已详细询问。" + extra + "\n"
        "4、体格检查：T 36.5℃，P 78次/分，R 18次/分，BP 128/78mmHg；"
        "神志清楚，查体合作，心肺腹查体未见明显异常，四肢活动及末梢循环尚可。\n"
        "5、专科情况：专科查体可见局部轻度压痛，未见明显畸形及异常活动，"
        "感觉、肌力或腹部体征与当前疾病表现相符，未引出明确病理征。\n"
        f"6、辅助检查：血常规、炎症指标及影像学检查提示与{disease}相符，"
        "部分结果仍待回报，需结合动态复查及临床变化综合判断。\n"
        "7、病情特点：目前生命体征相对平稳，但症状反复，需观察治疗反应，"
        "并警惕病情进展及并发症。"
    )
    # The real residual misses are mostly 0.9--1.4k characters.  Add a
    # deterministic, clinically neutral continuation so the patch exercises
    # that length range instead of only short toy sections.
    feature += "\n" + "\n".join(
        f"{8 + n}、补充观察：已核对病程记录、护理记录及检查结果，"
        "目前未见新的危险信号；后续需结合复查结果、症状变化和治疗反应持续评估。"
        for n in range(6 + index % 4)
    )
    diagnosis = disease
    evidence = f"1、临床表现与{disease}相符；2、专科查体支持；3、辅助检查结果支持当前判断。"
    plan = "1、完善血常规、生化、凝血及相关影像学检查；2、予对症及病因治疗；3、动态观察病情变化。"
    heading = ["病例特点", "病史特点及入院时情况", "该病例特点"][index % 3]
    source = (
        f"{heading}：\n{feature}\n"
        f"初步诊断：\n{diagnosis}\n"
        f"诊断依据：\n{evidence}\n"
        f"诊疗计划：\n{plan}\n"
        f"医师签名：医师{chr(65 + index % 6)}\n"
        f"记录日期：2027年{index % 12 + 1:02d}月{index % 27 + 1:02d}日"
    )
    return synthetic_sample(
        family="first_course_long_feature", split=split, index=index, source=source,
        pairs=[("病历特点", feature), ("初步诊断", diagnosis), ("诊断依据", evidence), ("诊疗计划", plan)],
        schema=schema,
        edge_tags=["long_parent_section", "numbered_subsections_stay_inside_parent",
                   "explicit_peer_heading_stops_feature", "alias_normalization"],
    )


def long_plan_sample(index: int, split: str, schema: dict) -> dict:
    disease = ["急性胃肠炎", "右侧输尿管结石", "高血压病", "颈椎病"][index % 4]
    feature = (
        f"1、患者因相关症状入院，病程中症状反复；2、查体生命体征平稳，"
        f"专科检查与{disease}相符；3、辅助检查已提示当前疾病，部分结果待回报。"
    )
    diagnosis = disease
    evidence = f"临床表现、查体和辅助检查均支持{disease}，目前诊断基本明确。"
    plan_parts = [
        "1、完善血常规、尿常规、生化、凝血、心电图及相关影像学检查，复核异常结果并记录变化；结合既往资料判断检查结果与当前症状的相关性，必要时补充针对性检查；检查前向患者解释注意事项，检查后及时查看报告并在病程中记录；",
        "2、根据症状给予补液、镇痛、解痉、降压或其他对症处理，观察疗效及不良反应；用药前核对过敏史、禁忌证和相互作用，用药后记录症状、生命体征及实验室指标变化；出现不适及时停用可疑药物并报告上级医师；",
        "3、结合检查结果评估病因，必要时请相关专科会诊，及时调整诊疗方案；对可能需要进一步操作或转科的情况，提前完成风险评估和沟通记录；会诊意见和后续执行情况均应如实记录，避免遗漏关键环节；",
        "4、密切观察体温、脉搏、呼吸、血压、意识、疼痛和腹部体征，出现变化立即复查；重点关注出血、感染、呼吸循环不稳定及病情突然加重等预警表现；如达到预警标准，立即启动相应处置流程并完善记录；",
        "5、向患者及家属说明目前病情、检查目的、治疗方案及可能风险，取得理解和配合；如患者对治疗存在疑问，及时解释并在病程记录中补充沟通内容；重要检查和治疗决定由患者或家属确认后再执行；",
        "6、指导合理饮食、休息及活动，做好跌倒、误吸、出血或其他并发症风险防范；根据病情变化调整护理级别，落实观察、陪护及安全告知措施；每日评估生活自理能力和护理需求并及时反馈；",
        "7、若病情加重或出现新的危险信号，及时升级监护并讨论进一步治疗；必要时完善术前或转诊准备，明确复诊、复查和随访安排；出院前再次核对用药、注意事项和随访时间；向接诊团队交代观察重点和处理优先级。",
        "8、根据每日病程变化复核诊疗目标，检查已完成项目和待办项目，及时补记遗漏内容；在病情稳定后逐步评估出院条件，并安排必要的复查与健康教育。",
    ]
    # Deliberately vary separators and whitespace while keeping each value a
    # single exact source span.
    if index % 3 == 1:
        plan_parts[1] = plan_parts[1].replace("2、", "2\n、")
    if index % 4 == 2:
        plan_parts[4] = "  " + plan_parts[4]
    plan = "\n".join(plan_parts)
    source = (
        f"病历特点：\n{feature}\n初步诊断：\n{diagnosis}\n"
        f"诊断依据：\n{evidence}\n诊疗计划：\n{plan}\n"
        f"医师签名：医师{chr(65 + index % 6)}\n"
        f"记录日期：2027年{index % 12 + 1:02d}月{index % 27 + 1:02d}日"
    )
    return synthetic_sample(
        family="first_course_long_plan", split=split, index=index, source=source,
        pairs=[("病历特点", feature), ("初步诊断", diagnosis), ("诊断依据", evidence), ("诊疗计划", plan)],
        schema=schema,
        edge_tags=["long_plan", "numbered_plan_complete_until_boundary",
                   "signature_and_date_are_boundaries", "whitespace_is_verbatim"],
    )


def whitespace_grounding_sample(index: int, split: str, schema: dict) -> dict:
    disease = ["冠心病", "慢性阻塞性肺疾病", "缺铁性贫血"][index % 3]
    feature = (
        f"1、患者为\n 中年\n男\n性，急性发病。2、患者自述近{index % 6 + 1}天出现不适，"
        "活动后加重，休息后稍缓解。3、查体：T 36.5℃，P 80次/分，"
        "R 18次/分，Bp 128/78mmHg，神志清楚，查体合作。4、专科检查："
        "局部轻压痛，未见明显畸形。5、辅助检查：相关化验及影像资料支持当前判断。"
    )
    diagnosis = disease
    evidence = f"1、症状和体征符合{disease}；2、辅助检查支持该诊断。"
    plan = (
        "1\n、完善相关检查，必要时复查；\n"
        "2、给予对症治疗，观察疗效和不良反应；\n"
        "3、将病情及治疗方案告知患者及家属。"
    )
    source = (
        f"病例特点：\n{feature}\n初步诊断：\n{diagnosis}\n诊断依据：\n{evidence}\n"
        f"诊疗计划：\n{plan}\n医师签名：医师{chr(65 + index % 6)}\n"
        f"记录日期：2027年{index % 12 + 1:02d}月{index % 27 + 1:02d}日"
    )
    return synthetic_sample(
        family="first_course_whitespace_grounding", split=split, index=index, source=source,
        pairs=[("病历特点", feature), ("初步诊断", diagnosis), ("诊断依据", evidence), ("诊疗计划", plan)],
        schema=schema,
        edge_tags=["verbatim_whitespace", "linebreak_inside_numbered_item",
                   "grounding_must_not_normalize", "validator_sensitive"],
    )


def validate(rows: list[dict], schema: dict) -> list[dict]:
    errors: list[dict] = []
    ids: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        if row["id"] in ids:
            errors.append({"id": row["id"], "error": "duplicate_id"})
        ids.add(row["id"])
        source = source_from_sample(row)
        fingerprint = hashlib.sha1(re.sub(r"\s+", "", source).encode()).hexdigest()
        if fingerprint in sources:
            errors.append({"id": row["id"], "error": "duplicate_source"})
        sources.add(fingerprint)
        document_schema = schema[row["metadata"]["document_type"]]
        allowed = set(document_schema.get("header_fields", [])) | set(document_schema.get("core_fields", [])) | set(document_schema.get("optional_fields", []))
        seen_fields: set[str] = set()
        for item in payload_from_sample(row):
            field = item["slot_attributes"]["field_name"]
            value = item["slot"]
            if field not in allowed:
                errors.append({"id": row["id"], "error": "unknown_field", "field": field})
            if field in seen_fields:
                errors.append({"id": row["id"], "error": "duplicate_field", "field": field})
            seen_fields.add(field)
            if not value or value not in source:
                errors.append({"id": row["id"], "error": "ungrounded", "field": field})
    return errors


def fingerprints(rows: list[dict]) -> set[str]:
    return {hashlib.sha1(re.sub(r"\s+", "", source_from_sample(row)).encode()).hexdigest() for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v11-dir", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--new-train-per-family", type=int, default=36)
    parser.add_argument("--new-validation-per-family", type=int, default=12)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))["documents"]
    base_train = load_jsonl(args.v11_dir / "v11_dedicated_section_train.jsonl")
    base_validation = load_jsonl(args.v11_dir / "v11_dedicated_section_validation.jsonl")
    holdout = load_jsonl(args.v11_dir / "v11_real_holdout_relabelled.jsonl")
    base_patch_train = load_jsonl(args.v11_dir / "v11_boundary_patch_train.jsonl")
    base_patch_validation = load_jsonl(args.v11_dir / "v11_boundary_patch_validation.jsonl")
    families = [long_feature_sample, long_plan_sample, whitespace_grounding_sample]
    new_train = [f(i, "train", schema) for f in families for i in range(args.new_train_per_family)]
    new_validation = [f(10_000 + i, "validation", schema) for f in families for i in range(args.new_validation_per_family)]
    train = base_train + new_train
    validation = base_validation + new_validation
    patch_train = base_patch_train + new_train
    patch_validation = base_patch_validation + new_validation
    rng = random.Random(20261110)
    for rows in (train, validation, patch_train, patch_validation):
        rng.shuffle(rows)
    split_rows = {"train": train, "validation": validation, "holdout": holdout,
                  "patch_train": patch_train, "patch_validation": patch_validation}
    errors = [{"split": split, **error} for split, rows in split_rows.items() for error in validate(rows, schema)]
    fps = {split: fingerprints(rows) for split, rows in {"train": train, "validation": validation, "holdout": holdout}.items()}
    overlaps = {"train_validation": len(fps["train"] & fps["validation"]), "train_holdout": len(fps["train"] & fps["holdout"]), "validation_holdout": len(fps["validation"] & fps["holdout"])}
    for pair, count in overlaps.items():
        if count:
            errors.append({"error": "source_overlap", "pair": pair, "count": count})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "v12_dedicated_section_train.jsonl", train)
    write_jsonl(args.output_dir / "v12_dedicated_section_validation.jsonl", validation)
    write_jsonl(args.output_dir / "v12_real_holdout_relabelled.jsonl", holdout)
    write_jsonl(args.output_dir / "v12_first_course_patch_train.jsonl", patch_train)
    write_jsonl(args.output_dir / "v12_first_course_patch_validation.jsonl", patch_validation)
    summary = {
        "strategy": "v11 plus focused first-course long-section and verbatim-whitespace counterexamples",
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "new_synthetic": {"train": len(new_train), "validation": len(new_validation), "families": [f.__name__ for f in families]},
        "source_overlap": overlaps,
        "validation_errors": errors,
        "passed": not errors,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
