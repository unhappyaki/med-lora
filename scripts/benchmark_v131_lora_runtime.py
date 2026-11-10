from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer

from evaluate_v5 import chat_template, load_jsonl, load_text_model


MODEL = "/root/medical-v5/inference/v131/model"
ADAPTER = "/root/medical-v5/inference/v131/adapter"
DATA = "/root/medical-v5/inference/v131/data/v131_real_holdout_relabelled.jsonl"


def generate_fixed(model, inputs: dict[str, torch.Tensor], tokens: int = 128):
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            min_new_tokens=tokens,
            max_new_tokens=tokens,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return output[:, inputs["input_ids"].shape[1] :]


def benchmark(model, inputs: dict[str, torch.Tensor]) -> tuple[list[dict], torch.Tensor]:
    generate_fixed(model, inputs, 1)
    torch.cuda.synchronize()
    runs: list[dict] = []
    last_output = None
    for _ in range(2):
        started = time.perf_counter()
        last_output = generate_fixed(model, inputs, 128)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = int(last_output.shape[1])
        runs.append(
            {
                "seconds": round(elapsed, 3),
                "generated_tokens": generated,
                "tokens_per_second": round(generated / elapsed, 3),
            }
        )
    return runs, last_output.detach().cpu()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    row = load_jsonl(Path(DATA))[0]
    prompt = chat_template(
        tokenizer,
        row["messages"][:2],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {name: value.cuda() for name, value in inputs.items()}

    base = load_text_model(MODEL).cuda().eval()
    base_runs, _ = benchmark(base, inputs)

    adapter_config = PeftConfig.from_pretrained(ADAPTER)
    live_lora = PeftModel.from_pretrained(base, ADAPTER).cuda().eval()
    live_runs, live_output = benchmark(live_lora, inputs)

    merge_started = time.perf_counter()
    merged = live_lora.merge_and_unload(safe_merge=True).cuda().eval()
    torch.cuda.synchronize()
    merge_seconds = time.perf_counter() - merge_started
    merged_runs, merged_output = benchmark(merged, inputs)

    print(
        json.dumps(
            {
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "adapter_rank": adapter_config.r,
                "target_modules": sorted(adapter_config.target_modules),
                "base": base_runs,
                "live_peft_lora": live_runs,
                "merge_seconds": round(merge_seconds, 3),
                "merged_lora": merged_runs,
                "live_merged_tokens_identical": bool(
                    torch.equal(live_output, merged_output)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
