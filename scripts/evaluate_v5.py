from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from medical_text_quality import (
    clean_legacy_spacing_artifacts,
    contains_spacing_artifact,
    has_abnormal_symbol_repetition,
    is_symbol_only_value,
)

try:
    from json_repair import repair_json
except ImportError:  # 允许仅使用严格 JSON 模式
    repair_json = None


def chat_template(tokenizer, messages: list[dict], **kwargs):
    try:
        result = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **kwargs)
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


class JsonObjectCompleteStoppingCriteria(StoppingCriteria):
    """Stop once the first top-level JSON object is syntactically closed."""

    def __init__(self, tokenizer, prompt_tokens: int):
        self.tokenizer = tokenizer
        self.last_token = prompt_tokens
        self.started = False
        self.depth = 0
        self.in_string = False
        self.escaped = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        new_tokens = input_ids[0, self.last_token :]
        self.last_token = int(input_ids.shape[1])
        fragment = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for char in fragment:
            if not self.started:
                if char == "{":
                    self.started = True
                    self.depth = 1
                continue
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
                continue
            if char == '"':
                self.in_string = True
            elif char == "{":
                self.depth += 1
            elif char == "}":
                self.depth -= 1
                if self.depth == 0:
                    return True
        return False


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


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_source(user_content: str) -> str:
    return user_content.split("Q: ", 1)[-1].rsplit("\nA:", 1)[0]


def replace_source(user_content: str, source: str) -> str:
    """Keep the production prompt unchanged while replacing only the EMR body."""
    if "Q: " not in user_content or "\nA:" not in user_content:
        raise ValueError("user prompt does not contain the expected Q:/A: markers")
    prefix = user_content.split("Q: ", 1)[0]
    return f"{prefix}Q: {source}\nA:"


def split_source(source: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries; never cut a clinical paragraph in half."""
    if len(source) <= max_chars:
        return [source]
    # Each unit includes its trailing blank-line separator so joining the chunks
    # always reconstructs the exact source text.
    blocks = [
        block
        for block in re.findall(r".*?(?:\n{2,}|$)", source, flags=re.DOTALL)
        if block
    ]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) > max_chars:
            chunks.append(current)
            current = ""
        current += block
        # A single long paragraph is intentionally kept whole. Cutting it would
        # turn one Gold paragraph slot into two invalid partial slots.
        if len(current) >= max_chars:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def merge_grounded_pairs(
    pair_groups: list[list[tuple[str, str]]], source: str
) -> list[tuple[str, str]]:
    """Union fallback outputs without inventing occurrences or duplicate fields."""
    merged: list[tuple[str, str]] = []
    seen: Counter[tuple[str, str]] = Counter()
    seen_fields: set[str] = set()
    for pairs in pair_groups:
        for field_name, slot in pairs:
            pair = (field_name, slot)
            if not slot or slot not in source:
                continue
            # The section schema requires each canonical field at most once.
            # Chunk outputs may independently rediscover fields such as 年龄 or
            # 记录者; keep the first grounded occurrence instead of concatenating
            # conflicting values from different chunks.
            if field_name in seen_fields:
                continue
            if seen[pair] >= source.count(slot):
                continue
            seen[pair] += 1
            seen_fields.add(field_name)
            merged.append(pair)
    return merged


def recover_unique_whitespace_span(slot: str, source: str) -> str | None:
    """Map a whitespace-only model variant back to one exact source span.

    Recovery is deliberately conservative: the non-whitespace text must be at
    least eight characters and occur exactly once in the normalized source.
    The returned value is always a literal, contiguous substring of ``source``.
    """
    target = normalize_text(slot)
    if len(target) < 8:
        return None

    normalized_chars: list[str] = []
    source_positions: list[int] = []
    for index, character in enumerate(source):
        if character == "\u00a0" or character.isspace():
            continue
        normalized_chars.append(character)
        source_positions.append(index)
    normalized_source = "".join(normalized_chars)

    starts: list[int] = []
    offset = 0
    while True:
        found = normalized_source.find(target, offset)
        if found < 0:
            break
        starts.append(found)
        if len(starts) > 1:
            return None
        offset = found + 1
    if len(starts) != 1:
        return None

    start = starts[0]
    end = start + len(target) - 1
    candidate = source[source_positions[start] : source_positions[end] + 1]
    return candidate if normalize_text(candidate) == target else None


def parse_json_response(text: str) -> tuple[dict | None, str | None, bool]:
    """Parse strict JSON first and use json-repair only as a marked fallback."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned), None, False
    except json.JSONDecodeError as first_error:
        if repair_json is None:
            return None, str(first_error), False
        try:
            repaired = repair_json(cleaned, return_objects=True)
        except Exception as repair_error:
            return None, f"{first_error}; repair_error={repair_error}", True
        if not isinstance(repaired, dict):
            return None, f"{first_error}; repair_result_not_object", True
        return repaired, str(first_error), True


def validate_payload(payload: dict | None) -> tuple[list[tuple[str, str]], bool]:
    if not isinstance(payload, dict) or set(payload) != {"extractions"}:
        return [], False
    if not isinstance(payload["extractions"], list):
        return [], False
    pairs: list[tuple[str, str]] = []
    for item in payload["extractions"]:
        if not isinstance(item, dict) or set(item) != {
            "slot",
            "slot_attributes",
        }:
            return [], False
        attributes = item["slot_attributes"]
        if not isinstance(item["slot"], str) or not isinstance(attributes, dict):
            return [], False
        if set(attributes) != {"field_name"}:
            return [], False
        field_name = attributes.get("field_name")
        if not isinstance(field_name, str) or not field_name:
            return [], False
        pairs.append((field_name, item["slot"]))
    return pairs, True


def salvage_prediction(
    payload: dict | None,
    source: str,
    max_extractions: int,
) -> tuple[list[tuple[str, str]], bool, list[str]]:
    """Keep grounded fields and report every discarded field for production fallback."""
    if not isinstance(payload, dict) or not isinstance(payload.get("extractions"), list):
        return [], False, ["invalid_root"]
    schema_valid = set(payload) == {"extractions"}
    items = payload["extractions"]
    errors: list[str] = []
    if len(items) > max_extractions:
        errors.append(f"too_many_extractions:{len(items)}>{max_extractions}")
        items = items[:max_extractions]

    pairs: list[tuple[str, str]] = []
    seen: Counter[tuple[str, str]] = Counter()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not {
            "slot",
            "slot_attributes",
        }.issubset(item):
            schema_valid = False
            errors.append(f"invalid_item:{index}")
            continue
        if set(item) != {"slot", "slot_attributes"}:
            schema_valid = False
        slot = item["slot"]
        attributes = item["slot_attributes"]
        if not isinstance(slot, str) or not isinstance(attributes, dict):
            schema_valid = False
            errors.append(f"invalid_item_types:{index}")
            continue
        field_name = attributes.get("field_name")
        if not isinstance(field_name, str) or not field_name:
            schema_valid = False
            errors.append(f"invalid_field_name_type:{index}")
            continue
        if set(attributes) != {"field_name"}:
            schema_valid = False
        if not slot:
            errors.append(f"empty_slot:{index}")
            continue
        if contains_spacing_artifact(slot):
            errors.append(f"html_spacing_artifact:{index}")
            continue
        if is_symbol_only_value(slot):
            errors.append(f"symbol_only_slot:{index}")
            continue
        if has_abnormal_symbol_repetition(slot):
            errors.append(f"abnormal_symbol_repetition:{index}")
            continue
        if slot not in source:
            recovered_slot = recover_unique_whitespace_span(slot, source)
            if recovered_slot is None:
                errors.append(f"ungrounded_slot:{index}")
                continue
            slot = recovered_slot
        if (
            len(field_name) > 64
            or "\n" in field_name
            or "\r" in field_name
            or contains_spacing_artifact(field_name)
            or is_symbol_only_value(field_name)
            or has_abnormal_symbol_repetition(field_name)
        ):
            errors.append(f"invalid_field_name:{index}")
            continue
        seen[(field_name, slot)] += 1
        # 同样的字段和值不能比它在原文中实际出现的次数更多。
        if seen[(field_name, slot)] > source.count(slot):
            errors.append(f"duplicate_beyond_source:{index}")
            continue
        pairs.append((field_name, slot))
    return pairs, schema_valid, errors


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("\u00a0", " "))


def clean_source_text(value: str) -> str:
    value = clean_legacy_spacing_artifacts(value).replace("{", " ").replace("}", " ")
    value = re.sub(r"[ \t]{2,}", " ", value)
    return re.sub(r"[ \t]*([，。；：、,.!?！？])[ \t]*", r"\1", value)


def counter_score(
    predicted: list[tuple[str, str]], gold: list[tuple[str, str]]
) -> tuple[int, int, int]:
    predicted_counter = Counter(predicted)
    gold_counter = Counter(gold)
    true_positive = sum((predicted_counter & gold_counter).values())
    return true_positive, len(predicted) - true_positive, len(gold) - true_positive


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def finalize_score(tp: int, fp: int, fn: int) -> dict:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 病历抽取 v5 统一评测")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument(
        "--keep-adapter-unmerged",
        action="store_true",
        help="诊断开关：保留 PEFT 动态 LoRA；默认合并 adapter 以提高推理速度",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-extractions",
        type=int,
        default=128,
        help="生产固定上限，不使用 Gold 数量，默认 128",
    )
    parser.add_argument(
        "--disable-retry",
        action="store_true",
        help="关闭 repair、截断、重复或字段校验失败后的自动重试",
    )
    parser.add_argument("--chunk-chars", type=int, default=2800)
    parser.add_argument(
        "--chunk-all-over-chars",
        type=int,
        default=5000,
        help="超过该长度即使首轮格式正常也做分段复核；0 表示仅异常时分段",
    )
    parser.add_argument("--disable-chunk-fallback", action="store_true")
    parser.add_argument(
        "--fallback-model",
        help="可选的第二层大模型路径；分段仍异常时自动调用，例如 Qwen3.5-4B",
    )
    parser.add_argument("--fallback-adapter")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("评测脚本需要 CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_text_model(args.model).cuda()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        if not args.keep_adapter_unmerged:
            model = model.merge_and_unload(safe_merge=True)
    model.eval()

    rows = load_jsonl(Path(args.data))
    if args.limit > 0:
        rows = rows[: args.limit]
    completed: dict[str, dict] = {}
    if predictions_path.exists():
        for item in load_jsonl(predictions_path):
            completed[item["id"]] = item

    fallback_runtime: dict[str, object | None] = {"model": None, "tokenizer": None}

    def get_fallback_runtime():
        if not args.fallback_model:
            return None, None
        if fallback_runtime["model"] is None:
            fallback_tokenizer = AutoTokenizer.from_pretrained(
                args.fallback_model, trust_remote_code=True
            )
            if fallback_tokenizer.pad_token_id is None:
                fallback_tokenizer.pad_token = fallback_tokenizer.eos_token
            fallback_model = load_text_model(args.fallback_model).cuda()
            if args.fallback_adapter:
                fallback_model = PeftModel.from_pretrained(
                    fallback_model, args.fallback_adapter
                )
                if not args.keep_adapter_unmerged:
                    fallback_model = fallback_model.merge_and_unload(safe_merge=True)
            fallback_runtime["tokenizer"] = fallback_tokenizer
            fallback_runtime["model"] = fallback_model.eval()
        return fallback_runtime["model"], fallback_runtime["tokenizer"]

    def attempt_rank(attempt: dict) -> tuple[int, int, int, int, int, int]:
        return (
            int(attempt["clean"]),
            int(not attempt["degenerate_repetition"]),
            int(not attempt["hit_max_new_tokens"]),
            int(attempt["schema_valid"]),
            int(attempt["strict_json_valid"]),
            len(attempt["predicted_pairs"]),
        )

    def run_attempt(runtime_model, runtime_tokenizer, messages: list[dict], source: str,
                    repetition_penalty: float) -> dict:
        prompt_text = chat_template(
            runtime_tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        inputs = runtime_tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
        stopping_criteria = StoppingCriteriaList(
            [
                JsonObjectCompleteStoppingCriteria(
                    runtime_tokenizer, int(inputs["input_ids"].shape[1])
                )
            ]
        )
        with torch.inference_mode():
            generated = runtime_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=repetition_penalty,
                pad_token_id=runtime_tokenizer.pad_token_id,
                eos_token_id=runtime_tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        response_ids = generated[0, inputs["input_ids"].shape[1] :]
        generated_tokens = int(response_ids.numel())
        response = runtime_tokenizer.decode(response_ids, skip_special_tokens=True)
        payload, parse_error, repair_used = parse_json_response(response)
        pairs, schema_valid, validation_errors = salvage_prediction(
            payload, source, args.max_extractions
        )
        degenerate = has_abnormal_symbol_repetition(response)
        if degenerate:
            validation_errors.append("degenerate_repetition")
        hit_max_new_tokens = generated_tokens >= args.max_new_tokens
        clean = (
            parse_error is None
            and not repair_used
            and schema_valid
            and not validation_errors
            and not hit_max_new_tokens
        )
        raw_pairs, raw_schema_valid = validate_payload(payload)
        if not (parse_error is None and not repair_used and raw_schema_valid):
            raw_pairs = []
        return {
            "response": response,
            "payload": payload,
            "parse_error": parse_error,
            "strict_json_valid": parse_error is None and not repair_used,
            "repair_used": repair_used,
            "schema_valid": schema_valid,
            "validation_errors": validation_errors,
            "raw_pairs": raw_pairs,
            "predicted_pairs": pairs,
            "generated_tokens": generated_tokens,
            "hit_max_new_tokens": hit_max_new_tokens,
            "degenerate_repetition": degenerate,
            "clean": clean,
        }

    def run_with_retry(runtime_model, runtime_tokenizer, messages: list[dict],
                       source: str) -> tuple[list[dict], int]:
        attempts = [run_attempt(runtime_model, runtime_tokenizer, messages, source, 1.0)]
        if not attempts[0]["clean"] and not args.disable_retry:
            attempts.append(
                run_attempt(runtime_model, runtime_tokenizer, messages, source, 1.03)
            )
        selected_index = max(
            range(len(attempts)), key=lambda index: attempt_rank(attempts[index])
        )
        return attempts, selected_index

    results: list[dict] = []
    for position, row in enumerate(rows, start=1):
        if row["id"] in completed:
            results.append(completed[row["id"]])
            continue

        raw_source = extract_source(row["messages"][1]["content"])
        source = clean_source_text(raw_source)
        base_messages = [dict(message) for message in row["messages"][:2]]
        base_messages[1]["content"] = replace_source(
            base_messages[1]["content"], source
        )

        attempts, selected_index = run_with_retry(
            model, tokenizer, base_messages, source
        )
        selected = attempts[selected_index]
        direct_pairs = list(selected["predicted_pairs"])
        predicted_pairs = list(direct_pairs)
        unresolved_quality = not selected["clean"]

        chunk_attempted = False
        chunk_records: list[dict] = []
        chunk_triggered_by_length = bool(
            args.chunk_all_over_chars
            and len(source) > args.chunk_all_over_chars
        )
        if not args.disable_chunk_fallback and (
            unresolved_quality or chunk_triggered_by_length
        ):
            chunk_attempted = True
            selected_chunks: list[dict] = []
            for chunk_index, chunk in enumerate(
                split_source(source, args.chunk_chars), start=1
            ):
                chunk_messages = [dict(message) for message in base_messages]
                chunk_messages[1]["content"] = replace_source(
                    chunk_messages[1]["content"], chunk
                )
                chunk_attempts, chunk_selected_index = run_with_retry(
                    model, tokenizer, chunk_messages, chunk
                )
                chunk_selected = chunk_attempts[chunk_selected_index]
                selected_chunks.append(chunk_selected)
                chunk_records.append(
                    {
                        "chunk_index": chunk_index,
                        "source_chars": len(chunk),
                        "selected_attempt": chunk_selected_index + 1,
                        "attempts": chunk_attempts,
                    }
                )
            chunk_pairs = merge_grounded_pairs(
                [item["predicted_pairs"] for item in selected_chunks], source
            )
            chunks_clean = bool(selected_chunks) and all(
                item["clean"] for item in selected_chunks
            )
            # Clean chunks cover the whole source, so they replace the suspect
            # direct output. Unioning both can double-count repeated values.
            predicted_pairs = (
                chunk_pairs
                if chunks_clean
                else merge_grounded_pairs([direct_pairs, chunk_pairs], source)
            )
            unresolved_quality = not chunks_clean
            if not selected_chunks:
                unresolved_quality = True
        else:
            chunk_pairs = []
        post_chunk_pairs = list(predicted_pairs)

        model_fallback_attempted = False
        model_fallback_record: dict | None = None
        if unresolved_quality and args.fallback_model:
            model_fallback_attempted = True
            fallback_model, fallback_tokenizer = get_fallback_runtime()
            fallback_attempts, fallback_selected_index = run_with_retry(
                fallback_model, fallback_tokenizer, base_messages, source
            )
            fallback_selected = fallback_attempts[fallback_selected_index]
            predicted_pairs = (
                list(fallback_selected["predicted_pairs"])
                if fallback_selected["clean"]
                else merge_grounded_pairs(
                    [predicted_pairs, fallback_selected["predicted_pairs"]], source
                )
            )
            unresolved_quality = not fallback_selected["clean"]
            model_fallback_record = {
                "selected_attempt": fallback_selected_index + 1,
                "attempts": fallback_attempts,
            }

        human_review_required = unresolved_quality or not predicted_pairs
        gold_payload = json.loads(row["messages"][2]["content"])
        result = {
            "id": row["id"],
            "document_type": row["metadata"]["document_type"],
            "source_preprocessed": source != raw_source,
            "response": selected["response"],
            "payload": selected["payload"],
            "parse_error": selected["parse_error"],
            "strict_json_valid": selected["strict_json_valid"],
            "repair_used": selected["repair_used"],
            "schema_valid": selected["schema_valid"],
            "validation_errors": selected["validation_errors"],
            "usable": bool(predicted_pairs),
            "needs_fallback": human_review_required,
            "human_review_required": human_review_required,
            "generated_tokens": selected["generated_tokens"],
            "hit_max_new_tokens": selected["hit_max_new_tokens"],
            "retry_used": len(attempts) > 1,
            "selected_attempt": selected_index + 1,
            "attempts": attempts,
            "chunk_attempted": chunk_attempted,
            "chunk_triggered_by_length": chunk_triggered_by_length,
            "chunk_records": chunk_records,
            "model_fallback_attempted": model_fallback_attempted,
            "model_fallback": model_fallback_record,
            "max_extractions": args.max_extractions,
            "stage_pairs": {
                "raw_strict": attempts[0]["raw_pairs"],
                "repair_salvage": attempts[0]["predicted_pairs"],
                "retry_selected": direct_pairs,
                "chunk_merged": post_chunk_pairs,
                "final": predicted_pairs,
            },
            "predicted_pairs": predicted_pairs,
            "gold": gold_payload,
        }
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)
        print(
            f"[{position}/{len(rows)}] {row['id']} "
            f"attempts={len(attempts)} repair={selected['repair_used']} "
            f"direct={len(direct_pairs)} final={len(predicted_pairs)} "
            f"chunk={chunk_attempted} 4b={model_fallback_attempted} "
            f"human_review={human_review_required}",
            flush=True,
        )

    totals = Counter()
    stage_totals: dict[str, Counter] = defaultdict(Counter)
    by_document: dict[str, Counter] = defaultdict(Counter)
    by_difficulty: dict[str, Counter] = defaultdict(Counter)
    per_case: list[dict] = []
    for row, result in zip(rows, results):
        source = clean_source_text(
            extract_source(row["messages"][1]["content"])
        )
        gold_payload = json.loads(row["messages"][2]["content"])
        gold_pairs = [
            (
                clean_legacy_spacing_artifacts(
                    item["slot_attributes"]["field_name"]
                ),
                clean_legacy_spacing_artifacts(item["slot"]),
            )
            for item in gold_payload["extractions"]
        ]
        predicted_pairs = [tuple(pair) for pair in result["predicted_pairs"]]
        tp, fp, fn = counter_score(predicted_pairs, gold_pairs)
        normalized_predicted = [
            (field, normalize_text(text)) for field, text in predicted_pairs
        ]
        normalized_gold = [(field, normalize_text(text)) for field, text in gold_pairs]
        ntp, nfp, nfn = counter_score(normalized_predicted, normalized_gold)
        grounded = sum(bool(text) and text in source for _, text in predicted_pairs)
        long_gold = [pair for pair in gold_pairs if len(pair[1]) >= 100]
        ltp, _, lfn = counter_score(predicted_pairs, long_gold)
        document_type = row["metadata"]["document_type"]
        values = {
            "cases": 1,
            "json_valid": int(
                result.get("strict_json_valid", result["parse_error"] is None)
            ),
            "repair_used": int(result.get("repair_used", False)),
            "repair_salvaged": int(
                result.get("repair_used", False) and bool(predicted_pairs)
            ),
            "schema_valid": int(result["schema_valid"]),
            "usable": int(result.get("usable", bool(predicted_pairs))),
            "needs_fallback": int(result.get("needs_fallback", False)),
            "validation_dropped_fields": len(result.get("validation_errors", [])),
            "hit_max_new_tokens": int(result.get("hit_max_new_tokens", False)),
            "retry_used": int(result.get("retry_used", False)),
            "chunk_attempted": int(result.get("chunk_attempted", False)),
            "model_fallback_attempted": int(
                result.get("model_fallback_attempted", False)
            ),
            "human_review_required": int(
                result.get("human_review_required", result.get("needs_fallback", False))
            ),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "normalized_tp": ntp,
            "normalized_fp": nfp,
            "normalized_fn": nfn,
            "predicted": len(predicted_pairs),
            "grounded": grounded,
            "case_exact": int(Counter(predicted_pairs) == Counter(gold_pairs)),
            "long_tp": ltp,
            "long_fn": lfn,
        }
        totals.update(values)
        by_document[document_type].update(values)
        difficulty = "hard" if row["metadata"].get("hard_case") else "standard"
        by_difficulty[difficulty].update(values)
        case_stages: dict[str, dict] = {}
        stage_pairs = result.get("stage_pairs", {"final": predicted_pairs})
        for stage_name, pairs in stage_pairs.items():
            typed_pairs = [tuple(pair) for pair in pairs]
            stage_tp, stage_fp, stage_fn = counter_score(typed_pairs, gold_pairs)
            stage_totals[stage_name].update(
                {
                    "cases": 1,
                    "tp": stage_tp,
                    "fp": stage_fp,
                    "fn": stage_fn,
                    "predicted": len(typed_pairs),
                    "usable": int(bool(typed_pairs)),
                }
            )
            case_stages[stage_name] = finalize_score(
                stage_tp, stage_fp, stage_fn
            )
        per_case.append(
            {
                "id": row["id"],
                "document_type": document_type,
                **values,
                "stages": case_stages,
            }
        )

    def metrics(counter: Counter) -> dict:
        return {
            "cases": counter["cases"],
            "json_valid_rate": round(safe_div(counter["json_valid"], counter["cases"]), 6),
            "schema_valid_rate": round(safe_div(counter["schema_valid"], counter["cases"]), 6),
            "usable_rate": round(safe_div(counter["usable"], counter["cases"]), 6),
            "repair_used_cases": counter["repair_used"],
            "repair_salvaged_cases": counter["repair_salvaged"],
            "needs_fallback_cases": counter["needs_fallback"],
            "validation_dropped_fields": counter["validation_dropped_fields"],
            "hit_max_new_tokens_cases": counter["hit_max_new_tokens"],
            "retry_used_cases": counter["retry_used"],
            "chunk_attempted_cases": counter["chunk_attempted"],
            "model_fallback_attempted_cases": counter["model_fallback_attempted"],
            "human_review_required_cases": counter["human_review_required"],
            "human_review_rate": round(
                safe_div(counter["human_review_required"], counter["cases"]), 6
            ),
            "exact": finalize_score(counter["tp"], counter["fp"], counter["fn"]),
            "whitespace_normalized": finalize_score(
                counter["normalized_tp"],
                counter["normalized_fp"],
                counter["normalized_fn"],
            ),
            "grounding_rate": round(safe_div(counter["grounded"], counter["predicted"]), 6),
            "case_exact_rate": round(safe_div(counter["case_exact"], counter["cases"]), 6),
            "long_extraction_recall": round(
                safe_div(counter["long_tp"], counter["long_tp"] + counter["long_fn"]), 6
            ),
        }

    report = {
        "model": args.model,
        "adapter": args.adapter,
        "data": args.data,
        "overall": metrics(totals),
        "stage_metrics": {
            stage_name: {
                "cases": counter["cases"],
                "usable_rate": round(
                    safe_div(counter["usable"], counter["cases"]), 6
                ),
                "exact": finalize_score(
                    counter["tp"], counter["fp"], counter["fn"]
                ),
            }
            for stage_name, counter in stage_totals.items()
        },
        "by_document_type": {
            key: metrics(value) for key, value in sorted(by_document.items())
        },
        "by_difficulty": {
            key: metrics(value) for key, value in sorted(by_difficulty.items())
        },
        "per_case": per_case,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
