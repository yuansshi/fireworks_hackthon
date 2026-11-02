"""LoRA SFT on Fireworks Serverless Training: no dataset upload, no deployment.

Trains Nemotron on the frozen data/ft_meta_train.jsonl, saves a sampler checkpoint before training
(the untuned base) and after every epoch, and samples any checkpoint in-session on the frozen eval
sets, writing results in the same schema as phish.evaluate.

    uv run python scripts/train_serverless.py smoke                       # 1 step on 32 rows + 8 samples
    uv run python scripts/train_serverless.py train --epochs 2            # base + per-epoch checkpoints
    uv run python scripts/train_serverless.py eval --ckpt base --model-key nemotron-super-base
    uv run python scripts/train_serverless.py eval --ckpt ep2  --model-key ft-nemotron-super
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet CDN is blocked on some corporate networks

import truststore

truststore.inject_into_ssl()  # the SDK sampler uses default certs; corporate TLS proxies need the OS trust store
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv
from fireworks.training.sdk import AdamParams, Datum, FiretitanServiceClient, ModelInput, SamplingParams

from phish.evaluate import DATA, RESULTS, read_jsonl, sha256
from phish.prompt import build_messages, parse_label

load_dotenv(DATA.parent / ".env")
BASE = os.environ.get("TRAIN_BASE_MODEL", "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16")
TOKENIZER = os.environ.get("TRAIN_TOKENIZER", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16")  # from the model's huggingFaceUrl
ENDPOINT = "https://api.fireworks.ai/training/v1/serverless"
PRICES = {"prefill": 0.57, "sample": 1.44, "train": 2.32}  # $/1M tokens (Serverless Training page, 2026-11-02)
EVAL_SPLITS = ["test", "test_temporal", "test_ood", "voice_prefix", "test_realcalls"]
STATE = RESULTS / "_ft_state.json"
EXTRAS = ("prefix_frac", "base_id", "tactics", "confusable_class", "scam_type")


def service() -> FiretitanServiceClient:
    return FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=ENDPOINT)


def load_tokenizer():
    """We send token ids and decode outputs ourselves, so the tokenizer comes straight from the model's HF repo."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def render(tok, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    """Chat-template token ids with thinking off (the label is the whole answer)."""
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                      enable_thinking=False)
    except TypeError:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def to_datum(tok, row: dict) -> Datum:
    msgs = row["messages"]
    prompt = render(tok, msgs[:-1], True)
    full = render(tok, msgs, False)
    if full[: len(prompt)] != prompt:  # template re-renders history differently: append the answer ourselves
        full = prompt + tok.encode(msgs[-1]["content"], add_special_tokens=False) + [tok.eos_token_id]
    weights = [0.0] * len(prompt) + [1.0] * (len(full) - len(prompt))
    return Datum(model_input=ModelInput.from_ints(full[:-1]),
                 loss_fn_inputs={"target_tokens": full[1:], "weights": weights[1:]})


def save_state(**kv) -> dict:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state.update(kv)
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    return state


def train(epochs: int, batch_size: int, lr: float, rank: int, limit: int | None, smoke: bool) -> None:
    svc = service()
    tc = svc.create_lora_training_client(base_model=BASE, rank=rank)
    print(f"session={svc.training_session_id} run={tc.run_id} base={BASE}", flush=True)
    tok = load_tokenizer()
    rows = read_jsonl(DATA / "ft_meta_train.jsonl")[: limit or None]
    print("rendered prompt tail:", repr(tok.decode(render(tok, rows[0]["messages"][:-1], True))[-160:]), flush=True)
    datums = [to_datum(tok, r) for r in rows]
    n_tok = sum(d.model_input.length for d in datums)
    print(f"{len(datums)} examples, {n_tok:,} tokens/epoch, est train cost ${n_tok * epochs * PRICES['train'] / 1e6:.2f}", flush=True)

    ckpts = {}
    if not smoke:  # the zero-initialised LoRA is the untuned base: the before-checkpoint
        ckpts["base"] = tc.save_weights_for_sampler("phish-base").result().path
        save_state(session=svc.training_session_id, run=tc.run_id, base_model=BASE, checkpoints=ckpts)
        print("checkpoint base:", ckpts["base"], flush=True)

    adam = AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
    rng, step, t0 = random.Random(42), 0, time.time()
    steps_per_epoch = (len(datums) + batch_size - 1) // batch_size
    for ep in range(1, epochs + 1):
        order = list(range(len(datums)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [datums[j] for j in order[i:i + batch_size]]
            fb = tc.forward_backward(batch, "cross_entropy")
            op = tc.optim_step(adam)
            out = fb.result()
            op.result()
            step += 1
            metrics = getattr(out, "metrics", None) or {}
            if step == 1 or step % 5 == 0 or smoke:
                shown = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))}
                print(f"epoch {ep} step {step}/{steps_per_epoch * epochs} {time.time() - t0:.0f}s {shown}", flush=True)
            if smoke:
                break
        name = "phish-smoke" if smoke else f"phish-ep{ep}"
        ckpts["smoke" if smoke else f"ep{ep}"] = tc.save_weights_for_sampler(name).result().path
        save_state(session=svc.training_session_id, run=tc.run_id, base_model=BASE, checkpoints=ckpts)
        print(f"checkpoint {name}: {ckpts['smoke' if smoke else f'ep{ep}']}", flush=True)
        if smoke:
            break
    if smoke:
        evaluate("smoke", "smoke-check", ["test"], limit=8, svc=svc, tok=tok)


def evaluate(ckpt: str, model_key: str, splits: list[str], limit: int | None = None, max_tokens: int = 24,
             concurrency: int = 24, svc=None, tok=None, path_override: str | None = None) -> None:
    state = json.loads(STATE.read_text())
    path = path_override or state["checkpoints"][ckpt]
    if not path.startswith("accounts/"):  # the sampler addresses a checkpoint through its training session
        path = f"accounts/{os.environ['FIREWORKS_ACCOUNT_ID']}/trainingSessions/{state['session']}/checkpoints/{path}"
    svc = svc or service()
    sampler = svc.create_sampling_client(model_path=path)
    tok = tok or load_tokenizer()
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    def one(rec: dict, split: str, test_sha: str) -> dict:
        prompt = render(tok, build_messages(rec, "meta"), True)
        t0 = time.perf_counter()
        err, text, n_out, stop = None, "", 0, None
        for attempt in range(4):
            try:
                res = sampler.sample(prompt=ModelInput.from_ints(prompt), num_samples=1, sampling_params=params).result()
                seq = res.sequences[0]
                n_out, stop = len(seq.tokens), str(getattr(seq, "stop_reason", ""))
                text = tok.decode(seq.tokens, skip_special_tokens=True).strip()
                err = None
                break
            except Exception as e:  # transient sampler errors: back off and retry
                err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(2 * (attempt + 1))
        pred = parse_label(text) if err is None else None
        row = {"id": rec["id"], "channel": rec["channel"], "label": rec["label"], "pred": pred, "raw": text[-200:],
               "finish_reason": stop, "latency_s": round(time.perf_counter() - t0, 4), "server_ttft_s": None,
               "server_processing_s": None, "prompt_tokens": len(prompt), "cached_tokens": 0, "completion_tokens": n_out,
               "reasoning_tokens": 0, "cost_usd": round((len(prompt) * PRICES["prefill"] + n_out * PRICES["sample"]) / 1e6, 8),
               "attempts": attempt + 1, "error": err, "parse_ok": pred is not None, "model": model_key, "model_id": path,
               "variant": "meta", "split": split, "serving": "serverless-training", "test_sha256": test_sha,
               "ts": datetime.now(timezone.utc).isoformat()}
        row.update({k: rec[k] for k in EXTRAS if k in rec})
        return row

    for split in splits:
        src = DATA / f"{split}.jsonl"
        recs, test_sha = read_jsonl(src)[: limit or None], sha256(src)
        out = RESULTS / f"{model_key}__meta__{split}.jsonl"
        done = {r["id"] for r in read_jsonl(out) if r.get("error") is None}
        todo = [r for r in recs if r["id"] not in done]
        if not todo:
            print(f"{out.name}: complete", flush=True)
            continue
        rows, t0 = [], time.time()
        with out.open("a") as fh, ThreadPoolExecutor(concurrency) as pool:
            futs = [pool.submit(one, r, split, test_sha) for r in todo]
            for f in as_completed(futs):
                row = f.result()
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
        ok = [r for r in rows if r["error"] is None]
        acc = sum(r["pred"] == r["label"] for r in ok) / max(1, len(ok))
        print(f"{model_key}/{split}: {len(ok)}/{len(rows)} ok | acc={acc:.3f} | parse={sum(r['parse_ok'] for r in ok) / max(1, len(ok)):.3f} "
              f"| cost=${sum(r['cost_usd'] for r in rows):.3f} | {time.time() - t0:.0f}s", flush=True)
        if model_key == "smoke-check":
            for r in rows[:4]:
                print("   sample output:", repr(r["raw"]), "gold", r["label"], "error", r["error"])
    sampler.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "train", "eval"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ckpt", default="ep2")
    ap.add_argument("--model-key", default="ft-nemotron-super")
    ap.add_argument("--split", nargs="+", default=EVAL_SPLITS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--path", help="fully qualified sampler path (accounts/<a>/trainingSessions/<ts>/checkpoints/<ckpt>)")
    args = ap.parse_args()
    if args.cmd == "smoke":
        train(1, 32, args.lr, args.rank, limit=32, smoke=True)
    elif args.cmd == "train":
        train(args.epochs, args.batch_size, args.lr, args.rank, args.limit, smoke=False)
    else:
        evaluate(args.ckpt, args.model_key, args.split, args.limit, concurrency=args.concurrency, path_override=args.path)
