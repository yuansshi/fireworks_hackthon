"""Async Fireworks client: one classification call with timing, server headers, usage, cost, retries."""

from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .prompt import RESPONSE_FORMAT, parse_label

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://api.fireworks.ai/inference/v1"
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}

load_dotenv(ROOT / ".env")


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "models.yaml").read_text())


def model_cfg(key: str, override_id: str | None = None) -> dict:
    cfg = load_config()
    if key not in cfg["models"]:
        raise SystemExit(f"unknown model {key!r}; choose from {sorted(cfg['models'])}")
    m = {**cfg["defaults"], **cfg["models"][key], "key": key}
    m["id"] = override_id or m.get("id") or os.environ.get(m.get("id_env", ""), "")
    if not m["id"]:
        raise SystemExit(f"{key}: no model id; set {m.get('id_env')} in .env (after deploy) or pass --model-id")
    return m


def make_client(timeout: float = 90.0) -> AsyncOpenAI:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY is not set: copy .env.example to .env and paste your key")
    return AsyncOpenAI(api_key=key, base_url=os.environ.get("FIREWORKS_BASE_URL", BASE_URL),
                       timeout=timeout, max_retries=0)


def call_cost(m: dict, prompt_tokens: int, cached: int, completion: int) -> float:
    """Per-call $ for serverless models. Dedicated deployments are costed per GPU-hour in the report."""
    if m.get("serving") != "serverless":
        return 0.0
    return ((prompt_tokens - cached) * m["price_in"] + cached * m.get("price_cached_in", m["price_in"])
            + completion * m["price_out"]) / 1e6


def _header_float(headers, name: str) -> float | None:
    v = headers.get(name)
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


async def classify(client: AsyncOpenAI, m: dict, messages: list[dict], max_attempts: int = 8) -> dict:
    kwargs = {
        "model": m["id"],
        "messages": messages,
        "temperature": m.get("temperature", 0),
        "max_tokens": m.get("max_tokens", 16),
        "extra_body": {**(m.get("extra_body") or {}), "perf_metrics_in_response": True},
    }
    if m.get("structured_output", True):
        kwargs["response_format"] = RESPONSE_FORMAT

    err = None
    for attempt in range(max_attempts):
        t0 = time.perf_counter()
        try:
            raw = await client.chat.completions.with_raw_response.create(**kwargs)
            latency = time.perf_counter() - t0
            resp = raw.parse()
            h = raw.headers
            usage = resp.usage
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
            choice = resp.choices[0]
            content = choice.message.content or ""
            perf = (resp.model_extra or {}).get("perf_metrics") or {}
            return {
                "pred": parse_label(content),
                "raw": content[-200:],
                "finish_reason": choice.finish_reason,
                "latency_s": round(latency, 4),
                "server_ttft_s": _header_float(h, "fireworks-server-time-to-first-token")
                                 or perf.get("server-time-to-first-token"),
                "server_processing_s": _header_float(h, "fireworks-server-processing-time")
                                       or perf.get("server-processing-time"),
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": cached,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": reasoning,
                "cost_usd": round(call_cost(m, usage.prompt_tokens, cached, usage.completion_tokens), 8),
                "attempts": attempt + 1,
                "error": None,
            }
        except openai.APIStatusError as e:
            err = f"{e.status_code}: {str(e)[:300]}"
            if e.status_code not in RETRYABLE:
                break
            retry_after = _header_float(e.response.headers, "retry-after")
            wait = retry_after if retry_after is not None else min(30.0, 1.5 * 2 ** attempt)
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            wait = min(30.0, 1.5 * 2 ** attempt)
        await asyncio.sleep(wait + random.random())
    return {"pred": None, "raw": "", "error": err, "latency_s": None, "prompt_tokens": 0,
            "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}
