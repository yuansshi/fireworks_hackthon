"""Run one model x prompt variant over frozen splits. Resumable and cost-guarded.

    uv run python -m phish.evaluate --model kimi-k3 --variant meta --split test test_temporal
    uv run python -m phish.evaluate --model kimi-k3-reasoning --split test --per-cell 25   # 200-case ceiling
    uv run python -m phish.evaluate --model ft-meta --mode latency                         # concurrency 1
    uv run python -m phish.evaluate --model ft-meta --mode throughput                      # concurrency 16
"""

from __future__ import annotations

import argparse
import os
import asyncio
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from .client import ROOT, classify, make_client, model_cfg
from .prompt import build_messages
from .retrieval import build_knn_messages

DATA = ROOT / "data"
RESULTS = Path(os.environ.get("PHISH_RESULTS", ROOT / "results"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def stratified(recs: list[dict], per_cell: int) -> list[dict]:
    """First `per_cell` records (by id, deterministic) for every channel x label cell."""
    cells: dict[tuple, list[dict]] = {}
    for r in sorted(recs, key=lambda r: r["id"]):
        cells.setdefault((r["channel"], r["label"]), []).append(r)
    return [r for group in cells.values() for r in group[:per_cell]]


def messages_for(r: dict, variant: str) -> list[dict]:
    return build_knn_messages(r) if variant == "knn" else build_messages(r, variant)


def projected_cost(m: dict, recs: list[dict], variant: str) -> float:
    if m.get("serving") != "serverless":
        return 0.0
    tokens_in = sum(len(json.dumps(messages_for(r, variant))) / 3.5 for r in recs)
    return (tokens_in * m["price_in"] + len(recs) * m.get("est_out_tokens", 8) * m["price_out"]) / 1e6


async def run(m: dict, recs: list[dict], variant: str, split: str, out: Path, concurrency: int, test_sha: str) -> float:
    client = make_client()
    sem = asyncio.Semaphore(concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a")

    async def one(r: dict):
        async with sem:
            res = await classify(client, m, messages_for(r, variant))
        row = {
            "id": r["id"], "channel": r["channel"], "label": r["label"], **res,
            "parse_ok": res["pred"] is not None, "model": m["key"], "model_id": m["id"],
            "variant": variant, "split": split, "serving": m.get("serving"), "test_sha256": test_sha,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        for k in ("prefix_frac", "base_id", "tactics", "confusable_class", "scam_type"):
            if k in r:
                row[k] = r[k]
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        return row

    t0 = time.perf_counter()
    rows = await tqdm_asyncio.gather(*(one(r) for r in recs), desc=f"{m['key']}/{variant}/{split}")
    wall = time.perf_counter() - t0
    fh.close()
    await client.close()
    ok = [x for x in rows if x["error"] is None]
    if ok:
        acc = sum(x["pred"] == x["label"] for x in ok) / len(ok)
        lat = statistics.median(x["latency_s"] for x in ok)
        cost = sum(x["cost_usd"] for x in rows)
        print(f"  done {len(ok)}/{len(rows)} ok | acc={acc:.3f} | parse={sum(x['parse_ok'] for x in ok)/len(ok):.3f} "
              f"| p50={lat:.2f}s | cost=${cost:.4f} | wall={wall:.1f}s")
    errs = [x["error"] for x in rows if x["error"]]
    if errs:
        print(f"  {len(errs)} errors, e.g. {errs[0]}")
    return wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", help="override the model id (e.g. a deployed adapter)")
    ap.add_argument("--variant", choices=["meta", "text", "knn"], default="meta")
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--per-cell", type=int, help="stratified subset: N per channel x label")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-cost", type=float, default=5.0, help="abort if projected $ exceeds this")
    ap.add_argument("--mode", choices=["eval", "latency", "throughput"], default="eval")
    ap.add_argument("--yes", action="store_true", help="skip the cost guard")
    args = ap.parse_args()

    m = model_cfg(args.model, args.model_id)
    if m.get("serving") == "serverless-training":
        raise SystemExit(f"{args.model} is sampled inside a training session: use scripts/train_serverless.py eval")
    splits = args.split
    if args.mode != "eval":
        splits = ["test"]
        args.per_cell = 13  # ~104 cases, identical for every model
        args.concurrency = 1 if args.mode == "latency" else 16

    for split in splits:
        path = DATA / f"{split}.jsonl"
        recs = read_jsonl(path)
        if not recs:
            raise SystemExit(f"missing {path}; run `make data` first")
        test_sha = sha256(path)
        if args.per_cell:
            recs = stratified(recs, args.per_cell)
        if args.limit:
            recs = recs[: args.limit]
        tag = split if args.mode == "eval" else args.mode
        out = RESULTS / f"{args.model}__{args.variant}__{tag}.jsonl"
        prev = read_jsonl(out)
        if prev and prev[0].get("test_sha256") != test_sha:
            raise SystemExit(f"{out.name} was produced from a different {split}.jsonl; move it aside first")
        if args.mode == "eval":
            done = {r["id"] for r in prev if r.get("error") is None}
            recs = [r for r in recs if r["id"] not in done]
        elif prev:
            out.unlink()  # timing runs are always fresh
        if not recs:
            print(f"{out.name}: already complete")
            continue
        est = projected_cost(m, recs, args.variant)
        print(f"{m['key']} [{m['id']}] {args.variant}/{tag}: {len(recs)} requests, projected ${est:.3f}")
        if est > args.max_cost and not args.yes:
            raise SystemExit(f"projected cost ${est:.2f} > --max-cost {args.max_cost}; rerun with --yes to proceed")
        wall = asyncio.run(run(m, recs, args.variant, split, out, args.concurrency, test_sha))
        if args.mode == "throughput":
            (RESULTS / f"{args.model}__throughput.json").write_text(json.dumps(
                {"model": args.model, "n": len(recs), "concurrency": args.concurrency, "wall_s": wall,
                 "rps": len(recs) / wall, "ts": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
