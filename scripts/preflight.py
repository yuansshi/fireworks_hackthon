"""One test call per model: is the id valid, is reasoning really off, does json_schema work,
which timing headers come back, and what rate limit does the account have?

    uv run python scripts/preflight.py                       # all serverless models
    uv run python scripts/preflight.py kimi-k3 nemotron-base # a subset
"""

from __future__ import annotations

import asyncio
import sys

from phish.client import classify, load_config, make_client, model_cfg
from phish.prompt import build_messages

PROBE = {
    "channel": "sms",
    "text": "USPS: Your package is on hold due to an unpaid $1.99 fee. Pay within 24 hours or it will be "
            "returned: https://usps-redelivery.top/pay",
    "meta": {"n_urls": 1, "url_domains": ["usps-redelivery.top"], "suspicious_tld": True,
             "urgency_cues": ["within 24 hours"], "money_or_prize": True},
    "label": 1,
}


async def probe(key: str) -> None:
    m = model_cfg(key)
    client = make_client()
    raw = await client.chat.completions.with_raw_response.create(
        model=m["id"], messages=build_messages(PROBE, "meta"), max_tokens=m.get("max_tokens", 16),
        temperature=0, extra_body={**(m.get("extra_body") or {}), "perf_metrics_in_response": True},
        **({"response_format": __import__("phish.prompt").prompt.RESPONSE_FORMAT} if m.get("structured_output", True) else {}),
    )
    limits = {k: v for k, v in raw.headers.items() if k.lower().startswith(("x-ratelimit", "fireworks-server"))}
    res = await classify(client, m, build_messages(PROBE, "meta"))
    verdict = "OK " if res["pred"] == 1 else "?? "
    print(f"{verdict}{key:22s} pred={res['pred']} raw={res['raw']!r:40.40} in={res['prompt_tokens']} "
          f"out={res['completion_tokens']} reasoning={res['reasoning_tokens']} lat={res['latency_s']}s "
          f"ttft={res.get('server_ttft_s')} err={res['error']}")
    if res["reasoning_tokens"] and m.get("structured_output", True):
        print(f"   WARNING: {key} still spends reasoning tokens; adjust extra_body in config/models.yaml")
    if limits:
        print("   headers:", limits)


async def main(keys: list[str]) -> None:
    for k in keys:
        try:
            await probe(k)
        except Exception as e:  # report and keep going: one bad id should not hide the others
            print(f"ERR {k:22s} {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    cfg = load_config()
    keys = sys.argv[1:] or [k for k, v in cfg["models"].items() if v.get("serving") == "serverless"
                            and k not in ("glm-5p3", "qwen3p8-max")]
    asyncio.run(main(keys))
