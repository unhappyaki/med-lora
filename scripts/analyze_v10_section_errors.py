#!/usr/bin/env python3
"""Analyze residual errors in dedicated large-section extraction predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_pairs(row: dict) -> list[tuple[str, str]]:
    return [
        (item["slot_attributes"]["field_name"], item["slot"])
        for item in row["gold"]["extractions"]
    ]


def shorten(value: str, limit: int = 90) -> str:
    value = value.replace("\r", "").replace("\n", "\\n")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def boundary_kind(predicted: str, gold: str) -> str:
    if gold in predicted:
        return "overrun_contains_gold"
    if predicted in gold:
        return "underrun_inside_gold"
    return "different_span"


def analyze(rows: list[dict]) -> dict:
    summary = Counter()
    by_document: dict[str, Counter] = defaultdict(Counter)
    fp_fields: Counter[str] = Counter()
    fn_fields: Counter[str] = Counter()
    label_confusions: Counter[str] = Counter()
    boundary_fields: Counter[str] = Counter()
    boundary_types: Counter[str] = Counter()
    cases: list[dict] = []

    for row in rows:
        document_type = row["document_type"]
        predicted = [tuple(pair) for pair in row["predicted_pairs"]]
        gold = gold_pairs(row)
        pred_counter = Counter(predicted)
        gold_counter = Counter(gold)
        exact = pred_counter & gold_counter
        unmatched_pred = list((pred_counter - exact).elements())
        unmatched_gold = list((gold_counter - exact).elements())

        local = Counter(
            cases=1,
            exact_pairs=sum(exact.values()),
            fp=len(unmatched_pred),
            fn=len(unmatched_gold),
            case_exact=int(not unmatched_pred and not unmatched_gold),
            json_invalid=int(not row.get("strict_json_valid", row.get("parse_error") is None)),
            schema_invalid=int(not row.get("schema_valid", False)),
            human_review=int(row.get("human_review_required", False)),
            validation_dropped=len(row.get("validation_errors", [])),
        )
        summary.update(local)
        by_document[document_type].update(local)

        gold_by_field: dict[str, list[str]] = defaultdict(list)
        gold_by_value: dict[str, list[str]] = defaultdict(list)
        pred_by_field: dict[str, list[str]] = defaultdict(list)
        for field, value in unmatched_gold:
            gold_by_field[field].append(value)
            gold_by_value[value].append(field)
            fn_fields[field] += 1
        for field, value in unmatched_pred:
            pred_by_field[field].append(value)
            fp_fields[field] += 1

        issue_details: list[dict] = []
        for field, value in unmatched_pred:
            if value in gold_by_value and field not in gold_by_value[value]:
                targets = gold_by_value[value]
                label_confusions[f"{field} -> {targets[0]}"] += 1
                issue_details.append(
                    {
                        "kind": "label_mismatch",
                        "predicted_field": field,
                        "gold_field": targets[0],
                        "value": shorten(value),
                    }
                )
            elif field in gold_by_field:
                candidate = max(gold_by_field[field], key=lambda item: min(len(item), len(value)))
                kind = boundary_kind(value, candidate)
                boundary_fields[field] += 1
                boundary_types[kind] += 1
                issue_details.append(
                    {
                        "kind": kind,
                        "field": field,
                        "predicted_length": len(value),
                        "gold_length": len(candidate),
                        "predicted": shorten(value),
                        "gold": shorten(candidate),
                    }
                )
            else:
                issue_details.append(
                    {
                        "kind": "unexpected_field",
                        "field": field,
                        "value": shorten(value),
                    }
                )

        for field, value in unmatched_gold:
            if value in {item for values in pred_by_field.values() for item in values}:
                continue
            if field not in pred_by_field:
                issue_details.append(
                    {
                        "kind": "missing_field",
                        "field": field,
                        "value": shorten(value),
                    }
                )

        if unmatched_pred or unmatched_gold or local["json_invalid"] or local["validation_dropped"]:
            cases.append(
                {
                    "id": row["id"],
                    "document_type": document_type,
                    "tp": sum(exact.values()),
                    "fp": len(unmatched_pred),
                    "fn": len(unmatched_gold),
                    "json_valid": not bool(local["json_invalid"]),
                    "schema_valid": not bool(local["schema_invalid"]),
                    "human_review_required": bool(local["human_review"]),
                    "generated_tokens": row.get("generated_tokens"),
                    "validation_errors": row.get("validation_errors", []),
                    "issues": issue_details,
                }
            )

    return {
        "summary": dict(summary),
        "by_document_type": {key: dict(value) for key, value in sorted(by_document.items())},
        "fp_fields": fp_fields.most_common(),
        "fn_fields": fn_fields.most_common(),
        "label_confusions": label_confusions.most_common(),
        "boundary_fields": boundary_fields.most_common(),
        "boundary_types": dict(boundary_types),
        "error_cases": cases,
    }


def render_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# v10 大 Section 真实留出集剩余错误分析",
        "",
        f"- 总病例：{s['cases']}；整例正确：{s['case_exact']}；非整例正确：{s['cases'] - s['case_exact']}。",
        f"- 剩余字段级错误：FP {s['fp']}，FN {s['fn']}。",
        f"- JSON 非严格合法：{s['json_invalid']}；Schema 非严格合法：{s['schema_invalid']}。",
        f"- 触发人工复核：{s['human_review']}；验证器删除字段：{s['validation_dropped']}。",
        "",
        "## 分文书",
        "",
        "| 文书 | 病例 | 整例正确 | FP | FN | JSON 非严格合法 | 人工复核 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for document_type, values in report["by_document_type"].items():
        lines.append(
            f"| {document_type} | {values['cases']} | {values['case_exact']} | {values['fp']} | "
            f"{values['fn']} | {values['json_invalid']} | {values['human_review']} |"
        )

    def counter_section(title: str, rows: list[list | tuple], left: str) -> None:
        lines.extend(["", f"## {title}", "", f"| {left} | 次数 |", "|---|---:|"])
        if rows:
            lines.extend(f"| {name} | {count} |" for name, count in rows)
        else:
            lines.append("| 无 | 0 |")

    counter_section("FP 集中的字段", report["fp_fields"], "字段")
    counter_section("FN 集中的字段", report["fn_fields"], "字段")
    counter_section("字段名混淆", report["label_confusions"], "预测 -> Gold")
    counter_section("边界错误集中的字段", report["boundary_fields"], "字段")
    counter_section("边界错误类型", list(report["boundary_types"].items()), "类型")

    lines.extend(["", "## 非整例正确病例", ""])
    for case in report["error_cases"]:
        flags = []
        if not case["json_valid"]:
            flags.append("JSON 非严格合法")
        if case["human_review_required"]:
            flags.append("需人工复核")
        suffix = f"；{'、'.join(flags)}" if flags else ""
        lines.append(
            f"### {case['document_type']} / `{case['id']}`（TP {case['tp']}，FP {case['fp']}，FN {case['fn']}{suffix}）"
        )
        lines.append("")
        for issue in case["issues"]:
            kind = issue["kind"]
            if kind == "label_mismatch":
                lines.append(
                    f"- 字段名错配：`{issue['predicted_field']}` 应为 `{issue['gold_field']}`；值：{issue['value']}"
                )
            elif kind in {"overrun_contains_gold", "underrun_inside_gold", "different_span"}:
                lines.append(
                    f"- 边界 `{issue['field']}` / {kind}：预测 {issue['predicted_length']} 字，Gold {issue['gold_length']} 字。"
                )
            elif kind == "unexpected_field":
                lines.append(f"- 多抽字段 `{issue['field']}`：{issue['value']}")
            else:
                lines.append(f"- 漏抽字段 `{issue['field']}`：{issue['value']}")
        if case["validation_errors"]:
            for error in case["validation_errors"]:
                lines.append(f"- 验证器：{error}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    report = analyze(load_jsonl(args.predictions))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "error_cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
