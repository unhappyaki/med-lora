from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from peft import PeftModel
from transformers import AutoTokenizer

from evaluate_v5 import load_text_model


MODEL = "/root/medical-v5/inference/v131/model"
ADAPTER = "/root/medical-v5/inference/v131/adapter"
OUTPUT = Path("/root/medical-v5/inference/v131/merged-model")


def main() -> None:
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base_model = load_text_model(MODEL)
    merged_model = PeftModel.from_pretrained(base_model, ADAPTER).merge_and_unload(
        safe_merge=True
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        OUTPUT,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(OUTPUT)
    for name in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "merges.txt",
        "vocab.json",
    ):
        source = Path(MODEL).resolve() / name
        if source.exists():
            shutil.copy2(source, OUTPUT / name)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
