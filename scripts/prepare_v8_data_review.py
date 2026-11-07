from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


HTML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
TEMPLATE_RE = re.compile(
    r"(?:这里输入|年龄阶段|病程长短|患者自诉情况|非整数|时间日期|"
    r"阴阳性|儿科[^，。；]{0,12}(?:描述|情况|结果)|有无[^，。；]{0,10}|"
    r"\{[^{}]{0,80}\}|\?{1,3}|？{1,3})"
)
IDENTITY_RE = re.compile(r"(?:姓名|住址|电话|身份证|签名|联系人)")
VITAL_RE = re.compile(r"(?:体温|脉率|心率|呼吸频率|收缩压|舒张压|体重|身高)")
GENERIC_LABELS = {
    "主诉",
    "现病史",
    "既往史",
    "家族史",
    "个人史",
    "婚育史",
    "月经史",
    "辅助检查",
    "辅助检查结果",
    "诊疗经过",
    "诊疗过程",
    "入院时情况",
    "入院时病情摘要",
    "妇科入院情况",
    "年龄阶段",
    "病程长短",
    "这里输入内容",
    "辅助检查内容",
}


def strip_html(value: str) -> str:
    value = html.unescape(value or "").replace("\u00a0", " ")
    value = re.sub(
        r"<\s*(?:br|/p|/div|/tr|/li|/h[1-6])\b[^>]*>",
        "\n",
        value,
        flags=re.I,
    )
    value = re.sub(r"<[^>]*>", "", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def compact(value: str) -> str:
    value = html.unescape(value or "").replace("\u00a0", " ")
    return re.sub(r"\s+", "", value).strip()


def plain_label(value: str) -> str:
    value = compact(value)
    value = re.sub(r"^(?:体格检查|体格检査)[-—_:：]*", "", value)
    value = re.sub(r"[（(][^）)]*[）)]", "", value)
    value = re.sub(r"(?:检查结果描述|检查结果|结果描述)$", "", value)
    value = value.replace("检査", "检查").replace("粘膜", "黏膜")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_f1(item: dict) -> float:
    matched = int(item.get("matching_chars", 0))
    prediction_chars = int(item.get("prediction_chars", 0))
    gold_chars = int(item.get("gold_chars", 0))
    precision = matched / prediction_chars if prediction_chars else 0.0
    recall = matched / gold_chars if gold_chars else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def label_prefix_candidate(gold: str, control_name: str, source: str) -> str:
    stripped = gold.strip()
    match = re.match(r"^([^：:\r\n]{1,40})[：:]\s*(.+)$", stripped, flags=re.S)
    if not match:
        return ""
    prefix, suffix = match.group(1), match.group(2).strip()
    prefix_key = plain_label(prefix)
    control_key = plain_label(control_name)
    related = (
        prefix_key in control_key
        or control_key in prefix_key
        or SequenceMatcher(None, prefix_key, control_key, autojunk=False).ratio() >= 0.55
    )
    if related and suffix and compact(suffix) in compact(source):
        return suffix
    return ""


def issue_flags(row: dict, prediction: dict, source: str) -> tuple[list[str], str]:
    gold = (row.get("gold_清洗后") or "").strip()
    control_name = (row.get("控件名") or "").strip()
    gold_key = compact(gold)
    flags: list[str] = []

    if not gold_key:
        flags.append("empty_gold")
    if gold_key and gold_key not in compact(source):
        flags.append("gold_not_in_source")
    if HTML_TAG_RE.search(row.get("原值_value_raw") or ""):
        flags.append("html_in_raw_value")
    if IDENTITY_RE.search(control_name):
        flags.append("identity_field")
    if TEMPLATE_RE.search(gold):
        flags.append("template_or_placeholder_gold")

    gold_label = re.sub(r"[：:，,。；;\-—_\s]+$", "", gold_key)
    if (
        plain_label(gold_label) == plain_label(control_name)
        or gold_label in GENERIC_LABELS
        or re.fullmatch(r"[^：:\r\n]{1,40}[：:]", gold.strip())
    ):
        flags.append("label_only_gold")
    if gold_key and not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", gold_key):
        flags.append("punctuation_only_gold")
    if re.search(r"岁.*岁$", gold_key):
        flags.append("malformed_age")
    if VITAL_RE.search(control_name) and re.fullmatch(
        r"(?:0|0(?:\.0+)?(?:℃|次/分|kg|KG|Kg|cm|mmHg)?)", gold_key, flags=re.I
    ):
        flags.append("suspicious_zero_vital")

    suggested_gold = label_prefix_candidate(gold, control_name, source)
    if suggested_gold:
        flags.append("label_prefix_in_gold")

    if prediction.get("format_wrapped"):
        flags.append("model_format_wrapped")
    if not prediction.get("prediction_grounded"):
        flags.append("model_prediction_not_grounded")
    if sample_f1(prediction) < 0.5:
        flags.append("model_low_char_f1")
    if prediction.get("empty_prediction"):
        flags.append("model_empty_prediction")
    return sorted(set(flags)), suggested_gold


def source_excerpt(source: str, gold: str, prediction: str, radius: int = 260) -> str:
    for needle in (gold, prediction):
        needle = (needle or "").strip()
        if not needle:
            continue
        index = source.find(needle)
        if index >= 0:
            start = max(0, index - radius)
            end = min(len(source), index + len(needle) + radius)
            result = source[start:end]
            if start:
                result = "…" + result
            if end < len(source):
                result += "…"
            return result
    return source[: radius * 2] + ("…" if len(source) > radius * 2 else "")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_gap_summary(eval_rows: list[dict], review_rows: list[dict], train: list[dict], flagged: list[dict]) -> dict:
    single = [
        item
        for item in train
        if str(item.get("metadata", {}).get("source_type", "")).startswith("single_control")
    ]
    hospitals = Counter(item["metadata"].get("hospital", "") for item in single)
    controls = Counter(item["metadata"].get("control_name", "") for item in single)
    source_lengths = [int(item["metadata"].get("source_chars", 0)) for item in single]
    target_lengths = [int(item["metadata"].get("target_chars", 0)) for item in single]
    eval_controls = Counter(row.get("控件名", "") for row in eval_rows)
    eval_hospitals = Counter(row.get("医院", "") for row in eval_rows)
    issue_counts = Counter(
        issue
        for row in review_rows
        for issue in (row.get("issue_flags") or "").split("|")
        if issue
    )
    train_control_keys = {plain_label(name) for name in controls}
    missing_controls = sorted(
        name
        for name in eval_controls
        if plain_label(name) not in train_control_keys and not IDENTITY_RE.search(name)
    )
    return {
        "version": "v8-data-review-01",
        "inputs": {
            "eval_rows": len(eval_rows),
            "train_rows": len(train),
            "single_control_train_rows": len(single),
            "flagged_train_candidates": len(flagged),
        },
        "review": {
            "auto_candidates": sum(row["review_bucket"] == "auto_candidate" for row in review_rows),
            "manual_review": sum(row["review_bucket"] == "manual_review" for row in review_rows),
            "issue_counts": dict(issue_counts),
        },
        "training_distribution": {
            "hospitals": dict(hospitals),
            "unique_control_names": len(controls),
            "source_length": {
                "max": max(source_lengths, default=0),
                "under_2000": sum(value < 2000 for value in source_lengths),
                "2000_to_3999": sum(2000 <= value < 4000 for value in source_lengths),
                "4000_to_7999": sum(4000 <= value < 8000 for value in source_lengths),
                "8000_plus": sum(value >= 8000 for value in source_lengths),
            },
            "target_length": {
                "1_to_5": sum(value <= 5 for value in target_lengths),
                "6_to_20": sum(6 <= value <= 20 for value in target_lengths),
                "21_to_100": sum(21 <= value <= 100 for value in target_lengths),
                "over_100": sum(value > 100 for value in target_lengths),
            },
            "missing_target_count": sum(
                compact(item["messages"][-1].get("content", "")) == "暂缺" for item in single
            ),
        },
        "evaluation_distribution": {
            "hospitals": dict(eval_hospitals),
            "unique_control_names": len(eval_controls),
            "control_names_not_covered_after_basic_alias_normalization": missing_controls,
        },
        "requested_new_data": {
            "target_clean_single_control_examples": 300,
            "hospital_target": {
                "柳城县人民医院": 90,
                "泉州市儿童医院": 75,
                "邵武市中医院": 65,
                "石狮市妇幼保健院": 70,
            },
            "negative_or_placeholder_only_ratio": "20%-30%",
            "long_source_minimum": {"4000_to_7999": 70, "8000_to_16000": 50},
            "long_target_over_100_minimum": 60,
        },
    }


def write_requirements(path: Path, summary: dict) -> None:
    missing_controls = summary["evaluation_distribution"][
        "control_names_not_covered_after_basic_alias_normalization"
    ]
    text = f"""# v8 单控件数据补充要求

## 当前已整理

- 四院评测：{summary['inputs']['eval_rows']} 条。
- 自动候选：{summary['review']['auto_candidates']} 条；仍需人工复核：{summary['review']['manual_review']} 条。
- 现有单控件训练：{summary['inputs']['single_control_train_rows']} 条。
- 现有单控件训练中的 `暂缺` 负样本：{summary['training_distribution']['missing_target_count']} 条。

自动候选仍需人工确认后才能成为正式 Gold，禁止用模型预测直接覆盖标注。

## 还需要的原始数据

目标新增约 300 条经人工确认的单控件样本：柳城 90、泉州 75、邵武 65、石狮 70。

每条至少提供以下字段：

1. `hospital`：医院。
2. `record_id`：稳定记录 ID，可提前哈希。
3. `document_type` / `category_id`：文书类型。
4. `full_text_raw`：完整病历原文；可以保留原始 HTML，清洗脚本会保留可见文字和段落边界。
5. `control_name`、`control_code`、`control_field`。
6. `control_value_raw`：数据库原始控件值。
7. 可选的 `is_filled` / 模板占位符标志。

不要额外提供姓名、电话、身份证等身份字段；如原文含有，入库前统一脱敏。

## 采样要求

- 20%–30% 为真实负样本：字段不存在、只有空模板或只有占位符，Gold 为 `暂缺`。
- 至少 70 条正文长度 4k–8k，至少 50 条正文长度 8k–16k。
- 至少 60 条目标值超过 100 字。
- 同一病历应包含相似控件的成组样本，用于训练字段边界，例如胸部/肺脏、肛门/外生殖器、脊柱/四肢。
- 重复短数字必须带可唯一定位的字段上下文；无法消歧的一律不进入训练。

## 优先补齐的控件

以下名称经过基础别名归一后仍未在训练数据中覆盖：

{chr(10).join('- ' + name for name in missing_controls)}

## 验收规则

- 正样本 Gold 必须在去 HTML 后的原文中逐字可定位。
- Gold 不得包含控件标题、冒号、JSON/Markdown 包装或模板占位符。
- 负样本必须由人工确认全文确实不存在对应事实。
- 单控件验证集与训练集按病例隔离，近重复病例也不得跨集合。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--epoch1-predictions", required=True, type=Path)
    parser.add_argument("--epoch2-predictions", required=True, type=Path)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--flagged-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    with args.eval_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        eval_rows = list(csv.DictReader(handle))
    epoch1 = {item["id"]: item for item in load_jsonl(args.epoch1_predictions)}
    epoch2 = {item["id"]: item for item in load_jsonl(args.epoch2_predictions)}
    train = load_jsonl(args.train_jsonl)
    flagged = load_jsonl(args.flagged_jsonl)

    review_rows: list[dict] = []
    for row in eval_rows:
        prediction_id = f"{row['序号']}::{row['case']}::{row['控件代码']}"
        p1 = epoch1[prediction_id]
        p2 = epoch2[prediction_id]
        source = strip_html(row.get("病历全文") or "")
        flags, suggested_gold = issue_flags(row, p2, source)
        blocking_flags = {
            "empty_gold",
            "gold_not_in_source",
            "identity_field",
            "template_or_placeholder_gold",
            "label_only_gold",
            "punctuation_only_gold",
            "malformed_age",
            "suspicious_zero_vital",
            "label_prefix_in_gold",
        }
        bucket = "manual_review" if blocking_flags.intersection(flags) else "auto_candidate"
        priority = 1
        if "gold_not_in_source" in flags or "label_only_gold" in flags:
            priority = 3
        elif blocking_flags.intersection(flags):
            priority = 2
        review_rows.append(
            {
                "review_bucket": bucket,
                "review_priority": priority,
                "review_status": "pending",
                "reviewer_gold": "",
                "review_note": "",
                "issue_flags": "|".join(flags),
                "序号": row.get("序号", ""),
                "医院": row.get("医院", ""),
                "分类名": row.get("分类名", ""),
                "case": row.get("case", ""),
                "控件名": row.get("控件名", ""),
                "控件代码": row.get("控件代码", ""),
                "gold": row.get("gold_清洗后", ""),
                "suggested_gold_after_label_strip": suggested_gold,
                "gold_grounded": p2.get("gold_grounded", False),
                "epoch1_prediction": p1.get("prediction", ""),
                "epoch2_prediction": p2.get("prediction", ""),
                "epoch1_exact": p1.get("exact_match", False),
                "epoch2_exact": p2.get("exact_match", False),
                "epoch2_char_f1": round(sample_f1(p2), 6),
                "epoch2_prediction_grounded": p2.get("prediction_grounded", False),
                "source_chars_after_html": len(source),
                "source_excerpt": source_excerpt(
                    source, row.get("gold_清洗后", ""), p2.get("prediction", "")
                ),
                "source_sha256": hashlib.sha256(compact(source).encode("utf-8")).hexdigest(),
            }
        )

    review_rows.sort(
        key=lambda row: (-int(row["review_priority"]), row["医院"], int(row["序号"]))
    )
    fields = list(review_rows[0])
    write_csv(args.output_dir / "eval_review_queue.csv", review_rows, fields)
    write_csv(
        args.output_dir / "eval_auto_candidates.csv",
        [row for row in review_rows if row["review_bucket"] == "auto_candidate"],
        fields,
    )
    write_csv(
        args.output_dir / "eval_manual_review.csv",
        [row for row in review_rows if row["review_bucket"] == "manual_review"],
        fields,
    )

    summary = build_gap_summary(eval_rows, review_rows, train, flagged)
    (args.output_dir / "v8_gap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_requirements(args.output_dir / "DATA_REQUIREMENTS.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
