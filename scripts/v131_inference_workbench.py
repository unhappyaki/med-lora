from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, StoppingCriteriaList

from evaluate_v5 import (
    JsonObjectCompleteStoppingCriteria,
    clean_source_text,
    load_text_model,
    parse_json_response,
    salvage_prediction,
)
from section_schema_prompt import build_section_prompt_v2


MODEL_DIR = os.environ.get("MODEL_DIR", "/root/medical-v5/inference/v131/model")
ADAPTER_DIR = os.environ.get(
    "ADAPTER_DIR", "/root/medical-v5/inference/v131/adapter"
)
SCHEMA_PATH = Path(
    os.environ.get(
        "SCHEMA_PATH", "/root/medical-v5/schemas/section_extraction_schema_v1.json"
    )
)
API_TOKEN = os.environ.get("WORKBENCH_TOKEN", "")
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "medical-section-v131")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", MODEL_NAME)
SYSTEM_PROMPT = "You are a helpful assistant that responds in JSON format."

app = FastAPI(title="V13.1 医疗大 Section 推理工作台", version="1.0")
model_lock = threading.Lock()
schema: dict[str, Any] = {}
tokenizer = None
model = None
startup_error: str | None = None
loaded_at: float | None = None


class ExtractRequest(BaseModel):
    document_type: str
    text: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=4096, ge=64, le=4096)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    max_tokens: int | None = Field(default=4096, ge=1, le=4096)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False


def authorize(
    token: str | None,
    x_api_key: str | None,
    authorization: str | None = None,
) -> None:
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    if API_TOKEN and token != API_TOKEN and x_api_key != API_TOKEN and bearer != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid workbench token")


def ensure_ready() -> None:
    if startup_error:
        raise HTTPException(status_code=503, detail=startup_error)
    if tokenizer is None or (not VLLM_BASE_URL and model is None):
        raise HTTPException(status_code=503, detail="Model is still loading")


def run_vllm_generation(
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = len(
        tokenizer(prompt, add_special_tokens=False).input_ids
    )
    body = json.dumps(
        {
            "model": VLLM_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": 1.05,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    request = Request(
        f"{VLLM_BASE_URL}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=600) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"vLLM backend HTTP {error.code}: {detail}",
        ) from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM backend unavailable: {error}",
        ) from error
    elapsed = time.perf_counter() - started
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    response_text = choice["message"]["content"] or ""
    generated_tokens = int(
        usage.get("completion_tokens")
        or len(tokenizer(response_text, add_special_tokens=False).input_ids)
    )
    return {
        "response": response_text,
        "prompt_tokens": int(usage.get("prompt_tokens") or prompt_tokens),
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_second": round(generated_tokens / elapsed, 3) if elapsed else None,
        "hit_max_new_tokens": choice.get("finish_reason") == "length",
    }


def run_generation(
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    ensure_ready()
    if VLLM_BASE_URL:
        return run_vllm_generation(
            messages,
            max_new_tokens,
            temperature,
            top_p,
        )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {name: value.cuda() for name, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    stopping = StoppingCriteriaList(
        [JsonObjectCompleteStoppingCriteria(tokenizer, prompt_tokens)]
    )

    started = time.perf_counter()
    with model_lock, torch.inference_mode():
        generation_kwargs: dict[str, Any] = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.05,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopping,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        output = model.generate(
            **inputs,
            **generation_kwargs,
        )
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    response_ids = output[0, prompt_tokens:]
    generated_tokens = int(response_ids.numel())
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return {
        "response": response,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_second": round(generated_tokens / elapsed, 3) if elapsed else None,
        "hit_max_new_tokens": generated_tokens >= max_new_tokens,
    }


@app.on_event("startup")
def startup() -> None:
    global schema, tokenizer, model, startup_error, loaded_at
    try:
        raw_schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        schema = raw_schema.get("documents", raw_schema)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not VLLM_BASE_URL:
            base_model = load_text_model(MODEL_DIR).cuda()
            model = (
                PeftModel.from_pretrained(base_model, ADAPTER_DIR)
                .merge_and_unload(safe_merge=True)
                .eval()
            )
            torch.backends.cuda.matmul.allow_tf32 = True
        loaded_at = time.time()
    except Exception as error:  # pragma: no cover - surfaced through /health
        startup_error = f"{type(error).__name__}: {error}"


@app.get("/health")
def health() -> dict[str, Any]:
    backend_error = None
    backend_ready = True
    if VLLM_BASE_URL:
        health_url = f"{VLLM_BASE_URL.removesuffix('/v1')}/health"
        try:
            with urlopen(health_url, timeout=2) as response:
                backend_ready = response.status == 200
        except (HTTPError, URLError, TimeoutError) as error:
            backend_ready = False
            backend_error = f"vLLM backend unavailable: {error}"
    ready = (
        tokenizer is not None
        and backend_ready
        and (bool(VLLM_BASE_URL) or model is not None)
    )
    effective_error = startup_error or backend_error
    return {
        "status": "ok" if ready and not effective_error else "loading_or_error",
        "ready": ready and not effective_error,
        "error": effective_error,
        "model": MODEL_NAME,
        "backend": (
            "vllm-merged-persistent-proxy"
            if VLLM_BASE_URL
            else "transformers-merged-lora-persistent"
        ),
        "adapter_merged": True,
        "backend_ready": backend_ready,
        "device": "cuda:0" if torch.cuda.is_available() else "unavailable",
        "loaded_at": loaded_at,
        "document_types": list(schema),
    }


@app.post("/extract")
def extract(
    request: ExtractRequest,
    token: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    authorize(token, x_api_key, authorization)
    ensure_ready()
    if request.document_type not in schema:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type: {request.document_type}",
        )

    source = clean_source_text(request.text)
    user_prompt = build_section_prompt_v2(source, request.document_type, schema)
    result = run_generation(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        request.max_new_tokens,
    )
    payload, parse_error, repair_used = parse_json_response(result["response"])
    pairs, schema_valid, validation_errors = salvage_prediction(payload, source, 128)
    extractions = [
        {"slot": slot, "slot_attributes": {"field_name": field_name}}
        for field_name, slot in pairs
    ]
    return {
        "document_type": request.document_type,
        "extractions": extractions,
        "quality": {
            "strict_json_valid": parse_error is None and not repair_used,
            "repair_used": repair_used,
            "schema_valid": schema_valid,
            "grounded": all(slot in source for _, slot in pairs),
            "validation_errors": validation_errors,
            "hit_max_new_tokens": result["hit_max_new_tokens"],
        },
        "usage": {
            "source_chars": len(source),
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["generated_tokens"],
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_per_second": result["tokens_per_second"],
        },
        "raw_response": result["response"],
    }


@app.get("/v1/models")
def models(
    token: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    authorize(token, x_api_key, authorization)
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatRequest,
    token: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    authorize(token, x_api_key, authorization)
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="stream=true is not enabled; use the non-streaming OpenAI endpoint",
        )
    completion_limit = request.max_completion_tokens or request.max_tokens or 4096
    result = run_generation(
        [message.model_dump() for message in request.messages],
        completion_limit,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    now = int(time.time())
    return {
        "id": f"chatcmpl-v131-{now}",
        "object": "chat.completion",
        "created": now,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["response"]},
                "finish_reason": "length" if result["hit_max_new_tokens"] else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["generated_tokens"],
            "total_tokens": result["prompt_tokens"] + result["generated_tokens"],
        },
        "performance": {
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_per_second": result["tokens_per_second"],
        },
    }


def page_html() -> str:
    options = "".join(f'<option value="{name}">{name}</option>' for name in schema)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V13.1 医疗大 Section 推理工作台</title>
<style>
:root{{--bg:#f4f7fb;--card:#fff;--ink:#172033;--muted:#68738a;--accent:#2f6fed;--ok:#16845b;--bad:#c13b45}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:26px}}h1{{margin:0 0 5px;font-size:24px}}.sub{{color:var(--muted);margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.card{{background:var(--card);border:1px solid #e4e9f2;border-radius:12px;padding:16px;box-shadow:0 4px 16px #23345a0d}}
label{{display:block;font-weight:650;margin:0 0 7px}}select,textarea,button{{font:inherit}}select{{width:100%;padding:9px;border:1px solid #ccd5e4;border-radius:8px;background:#fff}}
textarea{{width:100%;height:460px;resize:vertical;padding:12px;border:1px solid #ccd5e4;border-radius:8px;line-height:1.6}}button{{margin-top:12px;border:0;border-radius:8px;padding:10px 18px;background:var(--accent);color:#fff;font-weight:700;cursor:pointer}}button:disabled{{opacity:.55;cursor:wait}}
.status{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}}.pill{{padding:5px 9px;background:#edf2fa;border-radius:999px;color:var(--muted)}}.ok{{color:var(--ok)}}.bad{{color:var(--bad)}}
pre{{margin:0;white-space:pre-wrap;word-break:break-word;max-height:540px;overflow:auto;background:#0f172a;color:#dce8ff;border-radius:8px;padding:13px;min-height:460px}}
.note{{margin-top:12px;color:var(--muted)}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head>
<body><main class="wrap"><h1>V13.1 医疗大 Section 推理工作台</h1>
<div class="sub">常驻 Qwen3.5-0.8B + V13.1 LoRA · 原文去 HTML 后粘贴 · 不保存输入内容</div>
<div class="grid"><section class="card"><label for="docType">文书类型</label><select id="docType">{options}</select>
<label for="record" style="margin-top:12px">病历正文</label><textarea id="record" placeholder="粘贴去 HTML 后的单份病历，建议不超过 4K 字符"></textarea>
<button id="run">开始抽取</button><div class="note">长 Section 需要逐 token 返回全文，页面会显示实时等待时间；请勿重复提交。</div></section>
<section class="card"><div class="status"><span class="pill" id="state">待测试</span><span class="pill" id="timer">0.0s</span><span class="pill" id="speed">- tok/s</span></div><pre id="result">结果将在这里显示</pre></section></div></main>
<script>
const token=new URLSearchParams(location.search).get('token')||'';
const run=document.getElementById('run'), state=document.getElementById('state'), timer=document.getElementById('timer'), speed=document.getElementById('speed'), result=document.getElementById('result');
run.onclick=async()=>{{
 const text=document.getElementById('record').value.trim(); if(!text){{alert('请先粘贴病历正文');return}}
 run.disabled=true; state.textContent='推理中'; state.className='pill'; result.textContent='模型正在抽取，请等待…';
 const started=performance.now(); const tick=setInterval(()=>timer.textContent=((performance.now()-started)/1000).toFixed(1)+'s',100);
 try{{const res=await fetch('/extract',{{method:'POST',headers:{{'Content-Type':'application/json','X-API-Key':token}},body:JSON.stringify({{document_type:document.getElementById('docType').value,text,max_new_tokens:4096}})}}); const data=await res.json(); if(!res.ok)throw new Error(data.detail||JSON.stringify(data));
 state.textContent=data.quality.schema_valid?'完成':'需复核'; state.className='pill '+(data.quality.schema_valid?'ok':'bad'); speed.textContent=(data.usage.tokens_per_second??'-')+' tok/s'; result.textContent=JSON.stringify(data,null,2);
 }}catch(e){{state.textContent='失败';state.className='pill bad';result.textContent=String(e)}}finally{{clearInterval(tick);timer.textContent=((performance.now()-started)/1000).toFixed(1)+'s';run.disabled=false}}
}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index(
    token: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> str:
    authorize(token, x_api_key, authorization)
    return page_html()
