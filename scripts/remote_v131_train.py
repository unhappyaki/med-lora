#!/usr/bin/env python3
"""Upload, train, and evaluate the targeted V13.1 boundary patch."""

from __future__ import annotations

import argparse
import json
import posixpath
import shlex
from pathlib import Path

from remote_v12_pipeline import REMOTE_PYTHON, REMOTE_ROOT, connect, local_sha256, run


LOCAL_DATA = Path("outputs/medical_sft_v131_targeted_20261110")
REMOTE_DATA = f"{REMOTE_ROOT}/data/medical_sft_v131_targeted_20261110"
REMOTE_RUN = f"{REMOTE_ROOT}/runs/lora-r16-v131-targeted-from-v13-01"
REMOTE_EVAL_SHARDS = f"{REMOTE_ROOT}/runs/eval-v131-targeted-e1-real-holdout-shards"
V13_ADAPTER = f"{REMOTE_ROOT}/runs/lora-r16-v13-balanced-rehearsal-from-v11-01/checkpoint-epoch-1"


def upload(client) -> dict:
    required = [
        LOCAL_DATA / "v131_targeted_train.jsonl",
        LOCAL_DATA / "v131_targeted_validation.jsonl",
        LOCAL_DATA / "v131_real_holdout_relabelled.jsonl",
        LOCAL_DATA / "summary.json",
        Path("train_lora_v5.py"),
        Path("evaluate_v5.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing files: {missing}")

    result = run(client, f"mkdir -p {shlex.quote(REMOTE_DATA)} {shlex.quote(REMOTE_ROOT + '/scripts')}")
    if result["exit_code"]:
        raise RuntimeError(result)

    uploaded = []
    sftp = client.open_sftp()
    try:
        for path in required:
            remote = (
                posixpath.join(REMOTE_DATA, path.name)
                if path.parent == LOCAL_DATA
                else posixpath.join(REMOTE_ROOT, "scripts", path.name)
            )
            sftp.put(str(path), remote)
            uploaded.append({"local": str(path), "remote": remote, "sha256": local_sha256(path)})
    finally:
        sftp.close()

    remote_files = " ".join(shlex.quote(item["remote"]) for item in uploaded)
    hashes = run(client, f"sha256sum {remote_files}", timeout=300)
    remote_hashes = {
        line.split()[1]: line.split()[0]
        for line in hashes["stdout"].splitlines()
        if len(line.split()) >= 2
    }
    mismatches = [item for item in uploaded if remote_hashes.get(item["remote"]) != item["sha256"]]
    return {"uploaded": uploaded, "mismatches": mismatches, "remote_data": REMOTE_DATA}


def inspect(client) -> dict:
    command = f"""
echo '---gpu'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo '---training-processes'
pgrep -af '^/usr/local/miniconda3/envs/py312/bin/python /root/medical-v5/scripts/train_lora_v5.py' || true
echo '---v13-adapter'
test -d {shlex.quote(V13_ADAPTER)} && echo READY || echo MISSING
echo '---data'
wc -l {shlex.quote(posixpath.join(REMOTE_DATA, 'v131_targeted_train.jsonl'))} {shlex.quote(posixpath.join(REMOTE_DATA, 'v131_targeted_validation.jsonl'))} 2>/dev/null || true
"""
    return run(client, command)


def start_train(client) -> dict:
    train = posixpath.join(REMOTE_DATA, "v131_targeted_train.jsonl")
    validation = posixpath.join(REMOTE_DATA, "v131_targeted_validation.jsonl")
    log = posixpath.join(REMOTE_RUN, "train.log")
    pid_file = posixpath.join(REMOTE_RUN, "train.pid")
    command = f"""
set -e
test -d {shlex.quote(V13_ADAPTER)}
test -f {shlex.quote(train)}
test -f {shlex.quote(validation)}
if pgrep -af '^/usr/local/miniconda3/envs/py312/bin/python /root/medical-v5/scripts/train_lora_v5.py' >/dev/null; then
  echo TRAIN_ALREADY_RUNNING
  pgrep -af '^/usr/local/miniconda3/envs/py312/bin/python /root/medical-v5/scripts/train_lora_v5.py'
  exit 3
fi
mkdir -p {shlex.quote(REMOTE_RUN)}
nohup {shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_ROOT + '/scripts/train_lora_v5.py')} \
  --model /root/qwen35-0.8b-base \
  --resume-adapter {shlex.quote(V13_ADAPTER)} \
  --train-data {shlex.quote(train)} \
  --validation-data {shlex.quote(validation)} \
  --output-dir {shlex.quote(REMOTE_RUN)} \
  --epochs 1 \
  --max-length 6144 \
  --gradient-accumulation 8 \
  --learning-rate 5e-6 \
  --warmup-ratio 0.08 \
  --weight-decay 0.01 \
  --validation-loss-samples 32 \
  --seed 20261110 > {shlex.quote(log)} 2>&1 &
echo $! > {shlex.quote(pid_file)}
echo STARTED_PID=$(cat {shlex.quote(pid_file)})
echo LOG={shlex.quote(log)}
"""
    return run(client, command)


def status(client) -> dict:
    command = f"""
echo '---process'
pgrep -af '^/usr/local/miniconda3/envs/py312/bin/python /root/medical-v5/scripts/train_lora_v5.py' || true
echo '---gpu'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo '---train-log'
tail -n 100 {shlex.quote(posixpath.join(REMOTE_RUN, 'train.log'))} 2>/dev/null || true
echo '---checkpoints'
find {shlex.quote(REMOTE_RUN)} -maxdepth 2 -type d -name 'checkpoint-*' -print 2>/dev/null | sort
"""
    return run(client, command)


def start_sharded_eval(client) -> dict:
    rows = [
        json.loads(line)
        for line in (LOCAL_DATA / "v131_real_holdout_relabelled.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    shard_dir = LOCAL_DATA / "eval_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for shard_index in range(4):
        path = shard_dir / f"holdout_shard_{shard_index}.jsonl"
        shard_rows = rows[shard_index::4]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in shard_rows),
            encoding="utf-8",
        )
        shard_paths.append(path)

    result = run(client, f"mkdir -p {shlex.quote(REMOTE_DATA + '/eval_shards')} {shlex.quote(REMOTE_EVAL_SHARDS)}")
    if result["exit_code"]:
        raise RuntimeError(result)
    sftp = client.open_sftp()
    try:
        for path in shard_paths:
            sftp.put(str(path), posixpath.join(REMOTE_DATA, "eval_shards", path.name))
    finally:
        sftp.close()

    adapter = posixpath.join(REMOTE_RUN, "checkpoint-epoch-1")
    commands = ["set -e", f"test -d {shlex.quote(adapter)}"]
    for shard_index in range(4):
        shard_output = posixpath.join(REMOTE_EVAL_SHARDS, f"shard-{shard_index}")
        shard_data = posixpath.join(REMOTE_DATA, "eval_shards", f"holdout_shard_{shard_index}.jsonl")
        commands.extend(
            [
                f"mkdir -p {shlex.quote(shard_output)}",
                f"rm -f {shlex.quote(posixpath.join(shard_output, 'predictions.jsonl'))} {shlex.quote(posixpath.join(shard_output, 'metrics.json'))}",
                (
                    f"nohup {shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_ROOT + '/scripts/evaluate_v5.py')} "
                    f"--model /root/qwen35-0.8b-base --adapter {shlex.quote(adapter)} "
                    f"--data {shlex.quote(shard_data)} --output-dir {shlex.quote(shard_output)} "
                    f"--max-new-tokens 2048 > {shlex.quote(posixpath.join(shard_output, 'eval.log'))} 2>&1 &"
                ),
                f"echo $! > {shlex.quote(posixpath.join(shard_output, 'eval.pid'))}",
            ]
        )
    commands.append("echo SHARDS_STARTED")
    return run(client, "\n".join(commands))


def sharded_status(client) -> dict:
    lines = [
        "echo '---processes'",
        "pgrep -af '^/usr/local/miniconda3/envs/py312/bin/python /root/medical-v5/scripts/evaluate_v5.py' || true",
        "echo '---gpu'",
        "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader",
    ]
    for shard_index in range(4):
        shard_output = posixpath.join(REMOTE_EVAL_SHARDS, f"shard-{shard_index}")
        lines.extend(
            [
                f"echo '---shard-{shard_index}'",
                f"wc -l {shlex.quote(posixpath.join(shard_output, 'predictions.jsonl'))} 2>/dev/null || true",
                f"tail -n 5 {shlex.quote(posixpath.join(shard_output, 'eval.log'))} 2>/dev/null || true",
                f"test -f {shlex.quote(posixpath.join(shard_output, 'metrics.json'))} && echo METRICS_READY || true",
            ]
        )
    return run(client, "\n".join(lines))


def download_shards(client) -> dict:
    local_root = Path("outputs/eval-v131-targeted-e1-real-holdout-shards")
    local_root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    sftp = client.open_sftp()
    try:
        for shard_index in range(4):
            local_dir = local_root / f"shard-{shard_index}"
            local_dir.mkdir(parents=True, exist_ok=True)
            for name in ("predictions.jsonl", "metrics.json", "eval.log"):
                remote = posixpath.join(REMOTE_EVAL_SHARDS, f"shard-{shard_index}", name)
                local = local_dir / name
                sftp.get(remote, str(local))
                downloaded.append({"remote": remote, "local": str(local), "sha256": local_sha256(local)})
    finally:
        sftp.close()
    return {"downloaded": downloaded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["upload", "inspect", "start-train", "status", "start-sharded-eval", "sharded-status", "download-shards"],
    )
    args = parser.parse_args()
    client = connect()
    try:
        actions = {
            "upload": upload,
            "inspect": inspect,
            "start-train": start_train,
            "status": status,
            "start-sharded-eval": start_sharded_eval,
            "sharded-status": sharded_status,
            "download-shards": download_shards,
        }
        result = actions[args.action](client)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("exit_code", 0):
            raise SystemExit(result["exit_code"])
        if result.get("mismatches"):
            raise SystemExit(2)
    finally:
        client.close()


if __name__ == "__main__":
    main()
