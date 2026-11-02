"""Live demo for the video: the same held-out items through our system and a frontier model.

    uv run python scripts/demo.py --models ft-nemotron-super kimi-k3   # ours (recorded) vs Kimi K3 (live)
    uv run python scripts/demo.py --router                       # Nemotron voters, Kimi K3 only on disagreement
    uv run python scripts/demo.py --models nemotron-base+knn kimi-k3
    uv run python scripts/demo.py --router --channel sms --text "USPS: package on hold..."
"""

from __future__ import annotations

import argparse
import asyncio

from phish.client import classify, make_client, model_cfg
from phish.evaluate import DATA, RESULTS, read_jsonl
from phish.features import content_cues, link_summary, sms_meta, voice_meta
from phish.prompt import build_messages
from phish.retrieval import build_knn_messages


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


def cfg_and_builder(key: str):
    """`<model>+knn` = the model with the retrieval-augmented few-shot harness."""
    if key.endswith("+knn"):
        m = model_cfg(key[:-4])
        return {**m, "label": m["label"] + " + harness"}, build_knn_messages
    m = model_cfg(key)
    return m, lambda r: build_messages(r, "meta")


async def router(recs: list[dict], voters: list[str], judge: str) -> None:
    client = make_client()
    vs = [cfg_and_builder(v) for v in voters]
    jm, jb = cfg_and_builder(judge)
    for r in recs:
        truth = {1: "MALICIOUS", 0: "benign", None: "?"}[r["label"]]
        print("=" * 104)
        print(f"[{r['channel'].upper()}{' (first 50% of the call)' if r.get('prefix_frac') else ''}]  ground truth: {truth}")
        snippet = r["text"].replace("\n", " ")
        print("  " + snippet[:220] + ("…" if len(snippet) > 220 else ""))
        votes = await asyncio.gather(*(classify(client, m, b(r)) for m, b in vs))
        lat = max(v["latency_s"] or 0 for v in votes)
        cost = sum(v["cost_usd"] for v in votes)
        for (m, _), v in zip(vs, votes):
            print(f"    vote  {m['label']:44s} {({1: 'MALICIOUS', 0: 'benign', None: '??'})[v['pred']]:10s} {v['latency_s'] or 0:.2f}s")
        preds = {v["pred"] for v in votes}
        if len(preds) == 1 and None not in preds:
            final, route = votes[0]["pred"], "voters agree, answered by Nemotron"
        else:
            j = await classify(client, jm, jb(r))
            final, route = j["pred"], f"voters disagree, escalated to {jm['label']}"
            lat += j["latency_s"] or 0
            cost += j["cost_usd"]
        mark = " " if r["label"] is None else ("✔" if final == r["label"] else "✘")
        print(f"  {mark} ROUTER: {({1: 'MALICIOUS', 0: 'benign', None: '??'})[final]:10s} {lat:.2f}s  ${cost * 1e6:,.0f} per 1M msgs  ({route})")
    print("=" * 104)


def recorded(model_key: str) -> dict[str, dict]:
    """Saved predictions for a model that has no live endpoint (sampled inside a training session)."""
    rows = {}
    for path in RESULTS.glob(f"{model_key}__meta__*.jsonl"):
        for r in read_jsonl(path):
            rows[r["id"]] = r
    return rows


async def replay(rec: dict, key: str, cache: dict) -> dict:
    row = cache.setdefault(key, recorded(key)).get(rec["id"])
    if row is None:
        return {"pred": None, "latency_s": None, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    return {**row, "latency_s": None}


async def main(models: list[str], recs: list[dict]) -> None:
    client = make_client()
    pairs = [cfg_and_builder(m) for m in models]
    cfgs = [m for m, _ in pairs]
    cache: dict = {}
    for r in recs:
        truth = {1: "MALICIOUS", 0: "benign", None: "?"}[r["label"]]
        where = " (first 50% of the call)" if r.get("prefix_frac") else ""
        print("=" * 104)
        print(f"[{r['channel'].upper()}{where}]  ground truth: {truth}")
        snippet = r["text"].replace("\n", " ")
        print("  " + snippet[:220] + ("…" if len(snippet) > 220 else ""))
        results = await asyncio.gather(*(
            replay(r, m["key"], cache) if m.get("serving") == "serverless-training" else classify(client, m, b(r))
            for m, b in pairs))
        for m, res in zip(cfgs, results):
            verdict = {1: "MALICIOUS", 0: "benign", None: "??"}[res["pred"]]
            mark = " " if r["label"] is None else ("✔" if res["pred"] == r["label"] else "✘")
            cost = (f"${res['cost_usd'] * 1e6:,.0f} per 1M msgs" if str(m.get("serving", "")).startswith("serverless")
                    else "dedicated GPU")
            lat = (f"{res['latency_s']:.2f}s" if res["latency_s"]
                   else ("recorded" if m.get("serving") == "serverless-training" else "error"))
            print(f"  {mark} {m['label']:42s} {verdict:10s} {lat:>7s}  {res['prompt_tokens']:4d} in / "
                  f"{res['completion_tokens']} out  {cost}")
    print("=" * 104)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nemotron-base+knn", "kimi-k3"])
    ap.add_argument("--router", action="store_true", help="Nemotron voters, escalate disagreements to Kimi K3")
    ap.add_argument("--voters", nargs="+", default=["nemotron-base", "nemotron-base+knn", "deepseek-v4p1-flash"])
    ap.add_argument("--channel", default="sms", choices=["sms", "email", "chat", "url", "voice"])
    ap.add_argument("--text")
    args = ap.parse_args()
    examples = [custom_example(args.channel, args.text)] if args.text else held_out_examples()
    if args.router:
        asyncio.run(router(examples, args.voters, "kimi-k3"))
    else:
        asyncio.run(main(args.models, examples))
