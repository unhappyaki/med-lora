from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


TARGET_LEAVES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
}
EXCLUDED_PATH_PARTS = ("visual", "vision", "merger", "lm_head", "embed")


def chat_template(tokenizer, messages: list[dict], **kwargs):
    try:
        result = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **kwargs)
    # transformers 5.x may return a BatchEncoding from apply_chat_template,
    # while older releases return the input-id list directly.
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


def discover_target_modules(model: torch.nn.Module) -> list[str]:
    targets: list[str] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        lowered = name.lower()
        if not isinstance(module, torch.nn.Linear):
            continue
        if leaf not in TARGET_LEAVES:
            continue
        if any(part in lowered for part in EXCLUDED_PATH_PARTS):
            continue
        targets.append(name)
    if not targets:
        linear_leaves = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear)
            }
        )
        raise RuntimeError(
            "没有找到可注入 LoRA 的文本线性层。实际线性层叶子名："
            + ", ".join(linear_leaves)
        )
    return targets


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def encode_sample(tokenizer, row: dict, max_length: int) -> dict:
    messages = row["messages"]
    prompt_ids = chat_template(
        tokenizer,
        messages[:2],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prefix = longest_common_prefix(prompt_ids, full_ids)
    if prefix < len(prompt_ids) - 2:
        raise ValueError(
            f"{row['id']}: prompt/full chat template 前缀不一致 "
            f"({prefix}/{len(prompt_ids)})"
        )
    if len(full_ids) > max_length:
        raise ValueError(
            f"{row['id']}: {len(full_ids)} tokens 超过 max_length={max_length}。"
            "请先运行 analyze_v5_lengths.py，再提高 --max-length；不要静默截断病历。"
        )
    labels = [-100] * prefix + full_ids[prefix:]
    if all(value == -100 for value in labels):
        raise ValueError(f"{row['id']}: assistant 监督标签为空")
    return {
        "id": row["id"],
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "tokens": len(full_ids),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_loss(model, encoded: list[dict], max_samples: int) -> float:
    if not encoded or max_samples == 0:
        return float("nan")
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for item in encoded[:max_samples]:
            batch = {
                key: value.cuda(non_blocking=True)
                for key, value in item.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B 病历抽取 v5 LoRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--validation-loss-samples", type=int, default=12)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="仅用于烟雾测试；0 表示使用全部训练样本",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=0,
        help="仅用于烟雾测试；0 表示使用全部验证样本",
    )
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA；此脚本按 RTX 4090 bf16 训练配置编写")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 bf16")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_jsonl(Path(args.train_data))
    validation_rows = load_jsonl(Path(args.validation_data))
    if args.train_limit > 0:
        train_rows = train_rows[: args.train_limit]
    if args.validation_limit > 0:
        validation_rows = validation_rows[: args.validation_limit]
    print(f"编码训练集 {len(train_rows)} 条、验证集 {len(validation_rows)} 条")
    train_data = [
        encode_sample(tokenizer, row, args.max_length) for row in train_rows
    ]
    validation_data = [
        encode_sample(tokenizer, row, args.max_length) for row in validation_rows
    ]
    lengths = sorted(item["tokens"] for item in train_data)
    print(
        "训练 token 长度："
        f"min={lengths[0]}, p50={lengths[len(lengths)//2]}, max={lengths[-1]}"
    )

    model = load_text_model(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.resume_adapter:
        model = PeftModel.from_pretrained(
            model, args.resume_adapter, is_trainable=True
        )
        print(f"续训 adapter：{args.resume_adapter}")
    else:
        target_modules = discover_target_modules(model)
        print(f"LoRA 注入 {len(target_modules)} 个文本线性层")
        print(json.dumps(target_modules, ensure_ascii=False, indent=2))
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, config)

    model.print_trainable_parameters()
    model.cuda()
    model.train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    updates_per_epoch = math.ceil(
        len(train_data) / args.gradient_accumulation
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = max(1, round(total_updates * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_updates
    )

    run_config = vars(args).copy()
    run_config.update(
        {
            "cuda_device": torch.cuda.get_device_name(0),
            "train_samples": len(train_data),
            "validation_samples": len(validation_data),
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log_path = output_dir / "train_log.jsonl"
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_data)))
        random.Random(args.seed + epoch).shuffle(order)
        accumulated_loss = 0.0
        accumulated = 0
        for position, index in enumerate(order, start=1):
            item = train_data[index]
            batch = {
                key: value.cuda(non_blocking=True)
                for key, value in item.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation
            loss.backward()
            accumulated_loss += float(output.loss.detach().cpu())
            accumulated += 1

            should_update = (
                accumulated == args.gradient_accumulation
                or position == len(order)
            )
            if not should_update:
                continue
            clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            record = {
                "epoch": epoch,
                "update": global_update,
                "total_updates": total_updates,
                "loss": accumulated_loss / accumulated,
                "learning_rate": scheduler.get_last_lr()[0],
                "max_memory_gb": round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                ),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            accumulated_loss = 0.0
            accumulated = 0

        validation_loss = evaluate_loss(
            model, validation_data, args.validation_loss_samples
        )
        checkpoint = output_dir / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        epoch_record = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "checkpoint": str(checkpoint),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"训练完成：{final_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
