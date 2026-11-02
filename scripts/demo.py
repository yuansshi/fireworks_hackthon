"""Side-by-side live demo for the video: the same held-out items through our model and a frontier model.

    uv run python scripts/demo.py --models ft-meta kimi-k3
    uv run python scripts/demo.py --models nemotron-base kimi-k3 --channel sms --text "USPS: package on hold..."
"""

from __future__ import annotations

import argparse
import asyncio

from phish.client import classify, make_client, model_cfg
from phish.evaluate import DATA, read_jsonl
from phish.features import content_cues, link_summary, sms_meta, voice_meta
from phish.prompt import build_messages


def held_out_examples() -> list[dict]:
    test = read_jsonl(DATA / "test.jsonl")
    prefix = read_jsonl(DATA / "voice_prefix.jsonl")
    pick = lambda recs, f: next(r for r in sorted(recs, key=lambda r: r["id"]) if f(r))
    return [
        pick(test, lambda r: r["channel"] == "sms" and r["label"] == 1 and r["meta"].get("n_urls")),
        pick(test, lambda r: r["channel"] == "email" and r["label"] == 0),
        pick(test, lambda r: r["channel"] == "url" and r["label"] == 1),
        pick(prefix, lambda r: r["label"] == 1 and r["prefix_frac"] == 0.5),
    ]


def custom_example(channel: str, text: str) -> dict:
    meta = {"sms": sms_meta, "voice": voice_meta}.get(channel, lambda t: {**link_summary(t), **content_cues(t)})(text)
    return {"id": "custom", "channel": channel, "text": text, "meta": meta, "label": None}


async def main(models: list[str], recs: list[dict]) -> None:
    client = make_client()
    cfgs = [model_cfg(m) for m in models]
    for r in recs:
        truth = {1: "MALICIOUS", 0: "benign", None: "?"}[r["label"]]
        where = " (first 50% of the call)" if r.get("prefix_frac") else ""
        print("=" * 104)
        print(f"[{r['channel'].upper()}{where}]  ground truth: {truth}")
        snippet = r["text"].replace("\n", " ")
        print("  " + snippet[:220] + ("…" if len(snippet) > 220 else ""))
        results = await asyncio.gather(*(classify(client, m, build_messages(r, "meta")) for m in cfgs))
        for m, res in zip(cfgs, results):
            verdict = {1: "MALICIOUS", 0: "benign", None: "??"}[res["pred"]]
            mark = " " if r["label"] is None else ("✔" if res["pred"] == r["label"] else "✘")
            cost = f"${res['cost_usd'] * 1e6:,.0f} per 1M msgs" if m.get("serving") == "serverless" else "dedicated GPU"
            lat = f"{res['latency_s']:.2f}s" if res["latency_s"] else "error"
            print(f"  {mark} {m['label']:42s} {verdict:10s} {lat:>7s}  {res['prompt_tokens']:4d} in / "
                  f"{res['completion_tokens']} out  {cost}")
    print("=" * 104)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ft-meta", "kimi-k3"])
    ap.add_argument("--channel", default="sms", choices=["sms", "email", "chat", "url", "voice"])
    ap.add_argument("--text")
    args = ap.parse_args()
    examples = [custom_example(args.channel, args.text)] if args.text else held_out_examples()
    asyncio.run(main(args.models, examples))
