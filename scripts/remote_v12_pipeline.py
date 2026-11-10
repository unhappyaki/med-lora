#!/usr/bin/env python3
"""Small SSH/SFTP helper for the v12 first-course patch experiment.

Credentials are read from TRAIN_HOST, TRAIN_PORT, TRAIN_USER and TRAIN_PASS.
No secret is stored in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
import time
from pathlib import Path

import paramiko


LOCAL_DATA = Path("outputs/medical_sft_v12_first_course_patch_20261110")
REMOTE_ROOT = "/root/medical-v5"
REMOTE_DATA = f"{REMOTE_ROOT}/data/medical_sft_v12_first_course_patch_20261110"
REMOTE_RUN = f"{REMOTE_ROOT}/runs/lora-r16-v12-first-course-patch-from-v11-01"
REMOTE_EVAL = f"{REMOTE_ROOT}/runs/eval-v12-first-course-patch-e1-real-holdout-combined"
REMOTE_EVAL_SHARDS = f"{REMOTE_ROOT}/runs/eval-v12-first-course-patch-e1-real-holdout-shards"
V11_ADAPTER = f"{REMOTE_ROOT}/runs/lora-r16-v11-boundary-repair-from-base-01/checkpoint-epoch-2"
REMOTE_PYTHON = "/usr/local/miniconda3/envs/py312/bin/python"


def connect() -> paramiko.SSHClient:
    last_error: Exception | None = None
    for attempt in range(5):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                os.environ["TRAIN_HOST"],
                port=int(os.environ.get("TRAIN_PORT", "22")),
                username=os.environ.get("TRAIN_USER", "root"),
                password=os.environ["TRAIN_PASS"],
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
            )
            return client
        except (EOFError, OSError, paramiko.SSHException) as error:
            client.close()
            last_error = error
            if attempt < 4:
                time.sleep(3 + attempt * 2)
    raise RuntimeError(f"SSH connection failed after retries: {last_error}")


def run(client: paramiko.SSHClient, command: str, timeout: int = 120) -> dict:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return {
        "exit_code": exit_code,
        "stdout": stdout.read().decode("utf-8", "replace"),
        "stderr": stderr.read().decode("utf-8", "replace"),
    }


def local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload(client: paramiko.SSHClient) -> dict:
    required = [
        LOCAL_DATA / "v12_dedicated_section_train.jsonl",
        LOCAL_DATA / "v12_dedicated_section_validation.jsonl",
        LOCAL_DATA / "v12_real_holdout_relabelled.jsonl",
        LOCAL_DATA / "v12_first_course_patch_train.jsonl",
        LOCAL_DATA / "v12_first_course_patch_validation.jsonl",
        LOCAL_DATA / "summary.json",
        LOCAL_DATA / "strict_audit.json",
        Path("train_lora_v5.py"),
        Path("evaluate_v5.py"),
        Path("analyze_v10_section_errors.py"),
        Path("rescore_section_predictions.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing files: {missing}")
    result = run(client, f"mkdir -p {shlex.quote(REMOTE_DATA)} {shlex.quote(REMOTE_ROOT + '/scripts')}")
    if result["exit_code"]:
        raise RuntimeError(result)
    sftp = client.open_sftp()
    uploaded: list[dict] = []
    try:
        for path in required:
            if path.parent == LOCAL_DATA:
                remote = posixpath.join(REMOTE_DATA, path.name)
            else:
                remote = posixpath.join(REMOTE_ROOT, "scripts", path.name)
            sftp.put(str(path), remote)
            uploaded.append({"local": str(path), "remote": remote, "sha256": local_sha256(path)})
    finally:
        sftp.close()
    remote_files = " ".join(shlex.quote(item["remote"]) for item in uploaded)
    hashes = run(client, f"sha256sum {remote_files}", timeout=300)
    if hashes["exit_code"]:
        raise RuntimeError(hashes)
    remote_hashes = {
        line.split()[1]: line.split()[0]
        for line in hashes["stdout"].splitlines()
        if len(line.split()) >= 2
    }
    mismatches = [item for item in uploaded if remote_hashes.get(item["remote"]) != item["sha256"]]
    return {"uploaded": uploaded, "mismatches": mismatches, "remote_data": REMOTE_DATA}


def inspect(client: paramiko.SSHClient) -> dict:
    command = " && ".join(
        [
            "echo CONNECTED",
            "hostname",
            "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader",
            f"test -d {shlex.quote(V11_ADAPTER)}",
            f"find {shlex.quote(REMOTE_ROOT + '/runs/lora-r16-v11-boundary-repair-from-base-01')} -maxdepth 2 -type d | sort",
            "pgrep -af 'train_lora_v5.py|evaluate_v5.py' || true",
        ]
    )
    return run(client, command)


def runtime(client: paramiko.SSHClient) -> dict:
    command = fr"""
echo '---path'
printf '%s\n' "$PATH"
echo '---commands'
command -v python3 || true
command -v python || true
command -v conda || true
echo '---candidate-python'
find /root /opt /usr/local -maxdepth 7 \( -type f -o -type l \) \( -name python -o -name python3 \) 2>/dev/null | head -50
echo '---torch-locations'
find /root /opt /usr/local -maxdepth 8 -type d -name torch 2>/dev/null | head -30
echo '---python-modules'
{REMOTE_PYTHON} -c 'import torch,transformers,peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())' 2>&1 || true
echo '---medical-root'
ls -la /root/medical-v5
"""
    return run(client, command)


def start_train(client: paramiko.SSHClient) -> dict:
    train = posixpath.join(REMOTE_DATA, "v12_first_course_patch_train.jsonl")
    validation = posixpath.join(REMOTE_DATA, "v12_first_course_patch_validation.jsonl")
    log = posixpath.join(REMOTE_RUN, "train.log")
    command = f"""
set -e
test -d {shlex.quote(V11_ADAPTER)}
test -f {shlex.quote(train)}
if pgrep -af 'train_lora_v5.py' | grep -v grep >/dev/null; then
  echo TRAIN_ALREADY_RUNNING
  pgrep -af 'train_lora_v5.py'
  exit 3
fi
mkdir -p {shlex.quote(REMOTE_RUN)}
nohup {shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_ROOT + '/scripts/train_lora_v5.py')} \
  --model /root/qwen35-0.8b-base \
  --resume-adapter {shlex.quote(V11_ADAPTER)} \
  --train-data {shlex.quote(train)} \
  --validation-data {shlex.quote(validation)} \
  --output-dir {shlex.quote(REMOTE_RUN)} \
  --epochs 1 \
  --max-length 6144 \
  --gradient-accumulation 8 \
  --learning-rate 3e-5 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --validation-loss-samples 24 \
  --seed 20261110 > {shlex.quote(log)} 2>&1 &
echo $! > {shlex.quote(posixpath.join(REMOTE_RUN, 'train.pid'))}
echo STARTED_PID=$(cat {shlex.quote(posixpath.join(REMOTE_RUN, 'train.pid'))})
echo LOG={shlex.quote(log)}
"""
    return run(client, command)


def status(client: paramiko.SSHClient) -> dict:
    command = f"""
echo '---process'
pgrep -af 'train_lora_v5.py|evaluate_v5.py' || true
echo '---gpu'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo '---train-log'
tail -n 60 {shlex.quote(posixpath.join(REMOTE_RUN, 'train.log'))} 2>/dev/null || true
echo '---checkpoints'
find {shlex.quote(REMOTE_RUN)} -maxdepth 2 -type d -name 'checkpoint-*' -print 2>/dev/null | sort
"""
    return run(client, command)


def start_eval(client: paramiko.SSHClient) -> dict:
    adapter = posixpath.join(REMOTE_RUN, "checkpoint-epoch-1")
    data = posixpath.join(REMOTE_DATA, "v12_real_holdout_relabelled.jsonl")
    log = posixpath.join(REMOTE_EVAL, "eval.log")
    command = f"""
set -e
test -d {shlex.quote(adapter)}
test -f {shlex.quote(data)}
if pgrep -af 'evaluate_v5.py' | grep -v grep >/dev/null; then
  echo EVAL_ALREADY_RUNNING
  pgrep -af 'evaluate_v5.py'
  exit 3
fi
mkdir -p {shlex.quote(REMOTE_EVAL)}
rm -f {shlex.quote(posixpath.join(REMOTE_EVAL, 'predictions.jsonl'))} {shlex.quote(posixpath.join(REMOTE_EVAL, 'metrics.json'))}
nohup {shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_ROOT + '/scripts/evaluate_v5.py')} \
  --model /root/qwen35-0.8b-base \
  --adapter {shlex.quote(adapter)} \
  --data {shlex.quote(data)} \
  --output-dir {shlex.quote(REMOTE_EVAL)} \
  --max-new-tokens 2048 > {shlex.quote(log)} 2>&1 &
echo $! > {shlex.quote(posixpath.join(REMOTE_EVAL, 'eval.pid'))}
echo STARTED_PID=$(cat {shlex.quote(posixpath.join(REMOTE_EVAL, 'eval.pid'))})
echo LOG={shlex.quote(log)}
"""
    return run(client, command)


def eval_status(client: paramiko.SSHClient) -> dict:
    command = f"""
echo '---process'
pgrep -af 'evaluate_v5.py' || true
echo '---gpu'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo '---progress'
wc -l {shlex.quote(posixpath.join(REMOTE_EVAL, 'predictions.jsonl'))} 2>/dev/null || true
echo '---eval-log'
tail -n 35 {shlex.quote(posixpath.join(REMOTE_EVAL, 'eval.log'))} 2>/dev/null || true
echo '---metrics'
test -f {shlex.quote(posixpath.join(REMOTE_EVAL, 'metrics.json'))} && cat {shlex.quote(posixpath.join(REMOTE_EVAL, 'metrics.json'))} || true
"""
    return run(client, command)


def download_eval(client: paramiko.SSHClient) -> dict:
    local_dir = Path("outputs/eval-v12-first-course-patch-e1-real-holdout-combined")
    local_dir.mkdir(parents=True, exist_ok=True)
    files = ["predictions.jsonl", "metrics.json", "eval.log"]
    sftp = client.open_sftp()
    downloaded = []
    try:
        for name in files:
            remote = posixpath.join(REMOTE_EVAL, name)
            local = local_dir / name
            sftp.get(remote, str(local))
            downloaded.append({"remote": remote, "local": str(local), "sha256": local_sha256(local)})
    finally:
        sftp.close()
    return {"downloaded": downloaded}


def start_sharded_eval(client: paramiko.SSHClient) -> dict:
    rows = [json.loads(line) for line in (LOCAL_DATA / "v12_real_holdout_relabelled.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shard_dir = LOCAL_DATA / "eval_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    for shard_index in range(4):
        shard_rows = rows[shard_index::4]
        path = shard_dir / f"holdout_shard_{shard_index}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in shard_rows), encoding="utf-8")
        shard_paths.append(path)
    run(client, f"mkdir -p {shlex.quote(REMOTE_DATA + '/eval_shards')} {shlex.quote(REMOTE_EVAL_SHARDS)}")
    sftp = client.open_sftp()
    try:
        for path in shard_paths:
            sftp.put(str(path), posixpath.join(REMOTE_DATA, "eval_shards", path.name))
    finally:
        sftp.close()
    adapter = posixpath.join(REMOTE_RUN, "checkpoint-epoch-1")
    commands = [
        "set -e",
        f"test -d {shlex.quote(adapter)}",
        f"if test -f {shlex.quote(posixpath.join(REMOTE_EVAL, 'predictions.jsonl'))}; then cp {shlex.quote(posixpath.join(REMOTE_EVAL, 'predictions.jsonl'))} {shlex.quote(posixpath.join(REMOTE_EVAL, 'predictions.sequential-partial.jsonl'))}; fi",
        f"if test -f {shlex.quote(posixpath.join(REMOTE_EVAL, 'eval.pid'))}; then kill $(cat {shlex.quote(posixpath.join(REMOTE_EVAL, 'eval.pid'))}) 2>/dev/null || true; fi",
        "sleep 2",
    ]
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
    commands.append("pgrep -af '/root/medical-v5/scripts/[e]valuate_v5.py' || true")
    return run(client, "\n".join(commands))


def sharded_status(client: paramiko.SSHClient) -> dict:
    lines = [
        "echo '---processes'",
        "pgrep -af '/root/medical-v5/scripts/[e]valuate_v5.py' || true",
        "echo '---gpu'",
        "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader",
    ]
    for shard_index in range(4):
        shard_output = posixpath.join(REMOTE_EVAL_SHARDS, f"shard-{shard_index}")
        lines.extend(
            [
                f"echo '---shard-{shard_index}'",
                f"wc -l {shlex.quote(posixpath.join(shard_output, 'predictions.jsonl'))} 2>/dev/null || true",
                f"tail -n 4 {shlex.quote(posixpath.join(shard_output, 'eval.log'))} 2>/dev/null || true",
                f"test -f {shlex.quote(posixpath.join(shard_output, 'metrics.json'))} && echo METRICS_READY || true",
            ]
        )
    return run(client, "\n".join(lines))


def download_shards(client: paramiko.SSHClient) -> dict:
    local_root = Path("outputs/eval-v12-first-course-patch-e1-real-holdout-shards")
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
        choices=["inspect", "runtime", "upload", "start-train", "status", "start-eval", "eval-status", "download-eval", "start-sharded-eval", "sharded-status", "download-shards"],
    )
    args = parser.parse_args()
    client = connect()
    try:
        if args.action == "inspect":
            result = inspect(client)
        elif args.action == "runtime":
            result = runtime(client)
        elif args.action == "upload":
            result = upload(client)
        elif args.action == "start-train":
            result = start_train(client)
        elif args.action == "status":
            result = status(client)
        elif args.action == "start-eval":
            result = start_eval(client)
        elif args.action == "eval-status":
            result = eval_status(client)
        elif args.action == "download-eval":
            result = download_eval(client)
        elif args.action == "start-sharded-eval":
            result = start_sharded_eval(client)
        elif args.action == "sharded-status":
            result = sharded_status(client)
        else:
            result = download_shards(client)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("exit_code", 0):
            raise SystemExit(result["exit_code"])
        if result.get("mismatches"):
            raise SystemExit(2)
    finally:
        client.close()


if __name__ == "__main__":
    main()
