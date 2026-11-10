from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from evaluate_v5 import chat_template, load_jsonl, parse_json_response


MODEL = "/root/medical-v5/inference/v131/model"
ADAPTER = "/root/medical-v5/inference/v131/adapter"
DATA = Path("/root/medical-v5/inference/v131/data/v131_real_holdout_relabelled.jsonl")
REFERENCE = Path(
    "/root/medical-v5/inference/v131/outputs/smoke-20261110/predictions.jsonl"
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    row = load_jsonl(DATA)[0]
    reference = load_jsonl(REFERENCE)[0]
    if row["id"] != reference["id"]:
        raise RuntimeError("Holdout and reference prediction rows do not match")

    prompt = chat_template(
        tokenizer,
        row["messages"][:2],
        tokenize=False,
        add_generation_prompt=True,
    )
    started = time.perf_counter()
    engine = LLM(
        model=MODEL,
        enable_lora=True,
        max_lora_rank=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.55,
        max_model_len=8192,
        trust_remote_code=True,
    )
    load_seconds = time.perf_counter() - started
    sampling = SamplingParams(
        temperature=0,
        max_tokens=4096,
        repetition_penalty=1.05,
    )
    started = time.perf_counter()
    output = engine.generate(
        [prompt],
        sampling,
        lora_request=LoRARequest("v131", 1, ADAPTER),
    )[0].outputs[0]
    generation_seconds = time.perf_counter() - started
    response = output.text.strip()
    reference_response = reference["response"].strip()
    payload, parse_error, repair_used = parse_json_response(response)

    print(
        json.dumps(
            {
                "engine_load_seconds": round(load_seconds, 3),
                "prompt_tokens": len(
                    tokenizer(prompt, add_special_tokens=False).input_ids
                ),
                "generated_tokens": len(output.token_ids),
                "generation_seconds": round(generation_seconds, 3),
                "tokens_per_second": round(
                    len(output.token_ids) / generation_seconds, 3
                ),
                "finish_reason": output.finish_reason,
                "strict_json_valid": parse_error is None and not repair_used,
                "parse_error": parse_error,
                "repair_used": repair_used,
                "response_exact_match": response == reference_response,
                "payload_exact_match": payload == reference["payload"],
                "response_sha256": digest(response),
                "reference_sha256": digest(reference_response),
                "reference_generated_tokens": reference["generated_tokens"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
