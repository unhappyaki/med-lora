from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REQUIRED_COLUMNS = {
    "医院",
    "分类名",
    "case",
    "控件名",
    "控件代码",
    "gold_清洗后",
    "真实提示词_system",
    "真实提示词_user",
    "病历全文",
}


def chat_template(tokenizer, messages: list[dict], **kwargs):
    try:
        result = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **kwargs)
    return result


def load_text_model(model_path: str):
    common = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_path, **common)
    except (ImportError, ValueError):
        return AutoModelForCausalLM.from_pretrained(model_path, **common)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少列: {sorted(missing)}")
        return list(reader)


def strip_html(value: str) -> str:
    """Remove legacy EMR tags without discarding their visible content."""
    value = html.unescape(value).replace("\u00a0", " ")
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


STRICT_SYSTEM_SUFFIX = """

本次评测的额外约束（优先级最高）：
1) 花括号中的“这里输入内容、患者本人、治则治法、模板说明”等可能是未填写的模板占位符；除非它本身就是明确的真实病历事实，否则禁止把占位符当作控件值。
2) 忽略 HTML 标签、HTML 属性、脚本属性和模板编辑提示，只依据患者真实病历事实抽取。
3) 最终只能输出一个控件值的原文；禁止 JSON、数组、花括号包装、控件名和解释。
4) 找到明确事实时逐字输出；只有不存在真实填写内容时才输出“暂缺”。
""".strip()


def build_messages(
    row: dict[str, str], remove_html: bool, prompt_mode: str
) -> tuple[list[dict], str]:
    source = row["病历全文"]
    user_prompt = row["真实提示词_user"]
    if remove_html:
        cleaned_source = strip_html(source)
        if source not in user_prompt:
            raise ValueError(
                f"case={row['case']} 控件={row['控件名']} 的提示词不包含病历原文"
            )
        user_prompt = user_prompt.replace(source, cleaned_source, 1)
        source = cleaned_source
    system_prompt = row["真实提示词_system"]
    if prompt_mode == "strict":
        system_prompt = f"{system_prompt.rstrip()}\n\n{STRICT_SYSTEM_SUFFIX}"
        user_prompt = (
            f"{user_prompt.rstrip()}\n\n"
            "【输出前自检】答案必须是病历中的真实事实原文，不能是花括号模板占位符。"
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], source


def normalize(value: str) -> str:
    value = html.unescape(value).replace("\u00a0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", "", value).strip()


def matching_chars(prediction: str, gold: str) -> int:
    return sum(
        block.size
        for block in SequenceMatcher(None, prediction, gold, autojunk=False).get_matching_blocks()
    )


def looks_wrapped(value: str) -> bool:
    stripped = value.strip()
    return bool(
        stripped.startswith("```")
        or stripped.startswith("{")
        or stripped.startswith("[")
        or re.match(r"^(?:答案|结果|抽取结果)\s*[:：]", stripped)
    )


def score_prediction(
    row: dict[str, str], response: str, source: str, generated_tokens: int
) -> dict:
    prediction = response.strip()
    gold = row["gold_清洗后"].strip()
    prediction_normalized = normalize(prediction)
    gold_normalized = normalize(gold)
    source_normalized = normalize(source)
    matched = matching_chars(prediction_normalized, gold_normalized)
    return {
        "id": f"{row['序号']}::{row['case']}::{row['控件代码']}",
        "序号": row["序号"],
        "医院": row["医院"],
        "分类名": row["分类名"],
        "case": row["case"],
        "控件名": row["控件名"],
        "控件代码": row["控件代码"],
        "gold": gold,
        "raw_prediction": response,
        "prediction": prediction,
        "exact_match": prediction == gold,
        "normalized_match": prediction_normalized == gold_normalized,
        "prediction_grounded": bool(prediction_normalized)
        and prediction_normalized in source_normalized,
        "gold_grounded": bool(gold_normalized)
        and gold_normalized in source_normalized,
        "predicted_missing": prediction_normalized == "暂缺",
        "gold_missing": gold_normalized == "暂缺",
        "empty_prediction": not prediction_normalized,
        "format_wrapped": looks_wrapped(prediction),
        "matching_chars": matched,
        "prediction_chars": len(prediction_normalized),
        "gold_chars": len(gold_normalized),
        "generated_tokens": generated_tokens,
    }


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate(items: list[dict]) -> dict:
    total = len(items)
    matched = sum(item["matching_chars"] for item in items)
    prediction_chars = sum(item["prediction_chars"] for item in items)
    gold_chars = sum(item["gold_chars"] for item in items)
    char_precision = safe_div(matched, prediction_chars)
    char_recall = safe_div(matched, gold_chars)
    char_f1 = safe_div(2 * char_precision * char_recall, char_precision + char_recall)
    return {
        "samples": total,
        "cases": len({item["case"] for item in items}),
        "exact_matches": sum(item["exact_match"] for item in items),
        "exact_accuracy": round(
            safe_div(sum(item["exact_match"] for item in items), total), 6
        ),
        "normalized_matches": sum(item["normalized_match"] for item in items),
        "normalized_accuracy": round(
            safe_div(sum(item["normalized_match"] for item in items), total), 6
        ),
        "char_precision": round(char_precision, 6),
        "char_recall": round(char_recall, 6),
        "char_f1": round(char_f1, 6),
        "prediction_grounded_rate": round(
            safe_div(sum(item["prediction_grounded"] for item in items), total), 6
        ),
        "gold_grounded_rate": round(
            safe_div(sum(item["gold_grounded"] for item in items), total), 6
        ),
        "missing_accuracy": round(
            safe_div(
                sum(
                    item["predicted_missing"] == item["gold_missing"]
                    for item in items
                ),
                total,
            ),
            6,
        ),
        "empty_predictions": sum(item["empty_prediction"] for item in items),
        "format_wrapped": sum(item["format_wrapped"] for item in items),
    }


def grouped_summary(items: list[dict], key: str) -> dict[str, dict]:
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return {
        name: aggregate(group_items)
        for name, group_items in sorted(groups.items(), key=lambda pair: pair[0])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="单控件病历抽取真实评测")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20261105)
    parser.add_argument(
        "--prompt-mode",
        choices=("original", "strict"),
        default="original",
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help="保留原始 HTML；默认按生产要求去除标签后再推理",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("评测脚本需要 CUDA")
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于 0")

    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    rows = load_csv(Path(args.data))
    if args.sample_size > 0:
        rows = random.Random(args.sample_seed).sample(
            rows, min(args.sample_size, len(rows))
        )
    if args.limit > 0:
        rows = rows[: args.limit]

    completed: dict[str, dict] = {}
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    completed[item["id"]] = item

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_text_model(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.cuda().eval()

    pending: list[tuple[dict, list[dict], str, str]] = []
    all_results = list(completed.values())
    for row in rows:
        row_id = f"{row['序号']}::{row['case']}::{row['控件代码']}"
        if row_id in completed:
            continue
        messages, source = build_messages(
            row,
            remove_html=not args.keep_html,
            prompt_mode=args.prompt_mode,
        )
        prompt = chat_template(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        pending.append((row, messages, source, prompt))

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [item[3] for item in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.03,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response_ids = generated[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        batch_results: list[dict] = []
        for (row, _messages, source, _prompt), response, ids in zip(
            batch, responses, response_ids
        ):
            generated_tokens = int((ids != tokenizer.pad_token_id).sum().item())
            result = score_prediction(row, response, source, generated_tokens)
            batch_results.append(result)
            all_results.append(result)
        with predictions_path.open("a", encoding="utf-8") as handle:
            for result in batch_results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        done = min(start + len(batch), len(pending)) + len(completed)
        print(
            f"[{done}/{len(rows)}] "
            f"exact={sum(item['exact_match'] for item in all_results)} "
            f"normalized={sum(item['normalized_match'] for item in all_results)}",
            flush=True,
        )

    wanted_ids = {
        f"{row['序号']}::{row['case']}::{row['控件代码']}" for row in rows
    }
    final_results = [item for item in all_results if item["id"] in wanted_ids]
    summary = {
        "data": str(Path(args.data)),
        "model": args.model,
        "adapter": args.adapter,
        "html_removed": not args.keep_html,
        "prompt_mode": args.prompt_mode,
        "overall": aggregate(final_results),
        "by_hospital": grouped_summary(final_results, "医院"),
        "by_document_type": grouped_summary(final_results, "分类名"),
        "by_control": grouped_summary(final_results, "控件名"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
