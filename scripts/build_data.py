"""Build frozen train/val/test files, Fireworks fine-tuning JSONL, and a manifest.

    uv run python scripts/build_data.py            # build + check
    uv run python scripts/build_data.py --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from phish.data import DATASETS, SEED, build_all
from phish.prompt import LABELS, build_messages, label_json

DATA = Path(__file__).resolve().parents[1] / "data"
FORBIDDEN_IN_PROMPT = ["receiver", "dataset_name", "page_rank", "google_index", "web_traffic", "statistical_report"]
EVAL_SPLITS = ["test", "test_temporal", "test_ood", "voice_prefix", "test_realcalls"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def ft_rows(recs: list[dict], variant: str) -> list[dict]:
    return [
        {"messages": build_messages(r, variant) + [{"role": "assistant", "content": label_json(r["label"])}]}
        for r in recs
    ]


def smoke_rows(recs: list[dict], n: int = 200) -> list[dict]:
    """A small balanced slice across channels for the plumbing check job."""
    by_key: dict[tuple, list[dict]] = {}
    for r in recs:
        by_key.setdefault((r["channel"], r["label"]), []).append(r)
    per = max(1, n // max(1, len(by_key)))
    out = [r for group in by_key.values() for r in group[:per]]
    return ft_rows(out[:n], "meta")


def build() -> None:
    DATA.mkdir(exist_ok=True)
    splits = build_all(SEED)
    for name, recs in splits.items():
        write_jsonl(DATA / f"{name}.jsonl", recs)
    for variant in ("meta", "text"):
        write_jsonl(DATA / f"ft_{variant}_train.jsonl", ft_rows(splits["train"], variant))
        write_jsonl(DATA / f"ft_{variant}_val.jsonl", ft_rows(splits["val"], variant))
    write_jsonl(DATA / "ft_smoke_train.jsonl", smoke_rows(splits["train"]))

    revisions = {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        revisions = {d: api.dataset_info(d).sha for d in DATASETS}
    except Exception as e:  # revisions are nice-to-have provenance, not a hard requirement
        print(f"warning: could not fetch dataset revisions: {e}", file=sys.stderr)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "datasets": revisions,
        "files": {p.name: {"rows": sum(1 for _ in p.open()), "sha256": sha256(p)} for p in sorted(DATA.glob("*.jsonl"))},
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['files'])} files to {DATA}")


def check() -> bool:
    ok = True
    splits = {p.stem: read_jsonl(p) for p in DATA.glob("*.jsonl") if not p.stem.startswith("ft_")}

    print("\nCounts (channel x label) per split:")
    for name in ["train", "val", *EVAL_SPLITS]:
        recs = splits.get(name, [])
        c = Counter((r["channel"], r["label"]) for r in recs)
        cells = ", ".join(f"{ch}:{lab}={n}" for (ch, lab), n in sorted(c.items()))
        print(f"  {name:15s} n={len(recs):5d}  {cells}")

    # 1. No normalized-text overlap across splits (prefixes are cut from test calls by design).
    hashes = {name: {r["hash"] for r in recs} for name, recs in splits.items() if name != "voice_prefix"}
    names = list(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = hashes[a] & hashes[b]
            if inter:
                ok = False
                print(f"  FAIL overlap {a} x {b}: {len(inter)} rows")

    # 2. URL domains never shared between train and test.
    dom = {s: {r.get("domain") for r in splits.get(s, []) if r["channel"] == "url"} for s in ("train", "val", "test")}
    shared = (dom["train"] & dom["test"]) - {None}
    if shared:
        ok = False
        print(f"  FAIL {len(shared)} URL domains shared by train and test, e.g. {sorted(shared)[:3]}")

    # 3. Source-identifying or near-label fields never reach a prompt.
    for name in ("train", "test"):
        for r in splits.get(name, []):
            prompt = build_messages(r, "meta")[1]["content"]
            meta_block = prompt.split("Content:")[0]  # transcripts legitimately contain "receiver:" turns
            bad = [k for k in FORBIDDEN_IN_PROMPT if k in meta_block or k in (r.get("meta") or {})]
            if bad:
                ok = False
                print(f"  FAIL {r['id']} exposes {bad}")
                break

    # 4. Fine-tuning JSONL schema.
    for p in sorted(DATA.glob("ft_*.jsonl")):
        rows = read_jsonl(p)
        for row in rows:
            roles = [m["role"] for m in row["messages"]]
            target = json.loads(row["messages"][-1]["content"])["label"]
            if roles != ["system", "user", "assistant"] or target not in LABELS:
                ok = False
                print(f"  FAIL bad row in {p.name}")
                break
        print(f"  {p.name:22s} rows={len(rows)}")

    # 5. Rough token-length stats (chars/4) for the meta prompt.
    lens = sorted(len(build_messages(r, "meta")[1]["content"]) // 4 for r in splits.get("train", []))
    if lens:
        print(f"\nUser-prompt tokens (approx) train: median={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")
        total = sum(lens) + len(lens) * (len(build_messages(splits['train'][0], 'meta')[0]['content']) // 4 + 10)
        print(f"Approx training tokens per epoch (meta): {total:,}")

    print("\nCHECK", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.check_only:
        build()
    sys.exit(0 if check() else 1)
