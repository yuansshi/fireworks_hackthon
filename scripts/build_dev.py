"""Round-2 dev slices for checkpoint selection, disjoint from every frozen split.

    uv run python scripts/build_dev.py

Writes data/dev_temporal.jsonl (2008 CEAS emails that are not in test_temporal) and data/dev_ood.jsonl
(ScamGuardBench English items that are not in test_ood). Selecting checkpoints on these keeps the frozen
test files clean. The frozen train/val/test files are not touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phish.data import EMAIL_DS, OOD_DS, _balanced, _email_rows_to_records, _load_all_splits, make_record, norm_hash
from phish.features import content_cues, link_summary

DATA = Path(__file__).resolve().parents[1] / "data"
PER_LABEL = 100


def used_hashes() -> set[str]:
    hashes = set()
    for path in DATA.glob("*.jsonl"):
        if path.stem.startswith(("ft_", "dev_")):
            continue
        hashes |= {json.loads(line)["hash"] for line in path.open()}
    return hashes


def dev_temporal(rng, used: set[str]) -> list[dict]:
    df = _load_all_splits(EMAIL_DS)
    df = df[df["text"].fillna("").str.strip().str.len() > 20].copy()
    df["label"] = df["label"].astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["hash"] = df["text"].map(norm_hash)
    df = df.drop_duplicates("hash")
    pool = df[(df["dataset_name"] == "CEAS-08") & (df["date"].dt.year == 2008) & ~df["hash"].isin(used)]
    # Frozen records hash the formatted "Subject + body" text, so oversample and filter on the record hash too.
    return fresh(_email_rows_to_records(_balanced(pool, PER_LABEL + 25, rng)), used)


def dev_ood(rng, used: set[str]) -> list[dict]:
    df = pd.concat([_load_all_splits(OOD_DS, cfg) for cfg in ("standard", "hard", "legit_confusable")], ignore_index=True)
    df = df[(df["language"] == "en") & df["verdict"].isin(["scam_likely", "no_indicators"])].copy()
    df["label"] = (df["verdict"] == "scam_likely").astype(int)
    df["hash"] = df["text"].map(norm_hash)
    df = df[~df["hash"].isin(used)].drop_duplicates("hash")
    recs = []
    for r in _balanced(df, PER_LABEL, rng).itertuples(index=False):
        channel = r.channel if r.channel in ("sms", "email", "chat") else "sms"
        recs.append(make_record(channel, r.text, {**link_summary(r.text), **content_cues(r.text)}, r.label,
                                f"{OOD_DS}:{r.subset}", tactics=str(r.tactics), confusable_class=str(r.confusable_class)))
    return fresh(recs, used)


def fresh(recs: list[dict], used: set[str]) -> list[dict]:
    """Drop records already in a frozen split, then keep at most PER_LABEL per label."""
    out, counts = [], {0: 0, 1: 0}
    for r in recs:
        if r["hash"] not in used and counts[r["label"]] < PER_LABEL:
            out.append(r)
            counts[r["label"]] += 1
    return out


def main() -> None:
    rng = np.random.default_rng(2026)  # different stream from the frozen build (seed 42)
    used = used_hashes()
    for name, recs in (("dev_temporal", dev_temporal(rng, used)), ("dev_ood", dev_ood(rng, used))):
        overlap = sum(r["hash"] in used for r in recs)
        assert overlap == 0, f"{name}: {overlap} rows overlap frozen splits"
        with (DATA / f"{name}.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        pos = sum(r["label"] for r in recs)
        print(f"{name}: {len(recs)} rows ({pos} malicious / {len(recs) - pos} benign), 0 overlap with frozen splits")


if __name__ == "__main__":
    main()
