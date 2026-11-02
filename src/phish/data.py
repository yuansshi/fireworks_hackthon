"""Hugging Face loaders -> one record schema, plus balanced, leakage-checked splits.

Record: {id, channel, text, meta, timestamp, label (1=malicious), source, hash, ...extras}
"""

from __future__ import annotations

import hashlib
import math
import os
import re

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the Xet CDN is blocked on some corporate networks

import numpy as np
import pandas as pd
from datasets import load_dataset

from .features import (
    content_cues,
    email_meta,
    link_summary,
    sms_meta,
    split_turns,
    url_info,
    url_meta,
    voice_meta,
)

SEED = 42
PER_LABEL = {"train": 300, "val": 25, "test": 75}
TEMPORAL_PER_LABEL = 100
OOD_PER_LABEL = 100
VOICE_PREFIX_PER_LABEL = 50
PREFIX_FRACS = (0.25, 0.5, 0.75)

EMAIL_DS = "puyang2025/seven-phishing-email-datasets"
SMS_DS = "ucirvine/sms_spam"
URL_DS = "pirocheto/phishing-url"
VOICE_DS = "BothBosu/scam-dialogue"
REALCALLS_DS = "BothBosu/youtube-scam-conversations"
OOD_DS = "flowxai/scamguardbench"
DATASETS = [EMAIL_DS, SMS_DS, URL_DS, VOICE_DS, REALCALLS_DS, OOD_DS]


def norm_hash(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "")[:5000].lower()).strip()
    return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()


def make_record(channel, text, meta, label, source, ts=None, **extra) -> dict:
    h = norm_hash(text)
    rec = {
        "id": f"{channel}-{h[:12]}",
        "channel": channel,
        "text": text,
        "meta": meta,
        "timestamp": ts.isoformat() if ts is not None else None,
        "label": int(label),
        "source": source,
        "hash": h,
    }
    rec.update(extra)
    return rec


def _load_all_splits(name: str, config: str | None = None) -> pd.DataFrame:
    ds = load_dataset(name, config) if config else load_dataset(name)
    return pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)


def _sample_splits(df: pd.DataFrame, rng: np.random.Generator, per_label=PER_LABEL) -> dict[str, pd.DataFrame]:
    """Balanced per-label sample, carved into train/val/test with no overlap."""
    out = {s: [] for s in per_label}
    need = sum(per_label.values())
    for lab in (0, 1):
        sub = df[df["label"] == lab]
        if len(sub) < need:
            raise ValueError(f"only {len(sub)} rows with label={lab}, need {need}")
        sub = sub.sample(n=need, random_state=int(rng.integers(1 << 31)))
        start = 0
        for split, n in per_label.items():
            out[split].append(sub.iloc[start:start + n])
            start += n
    return {s: pd.concat(parts) for s, parts in out.items()}


def _balanced(df: pd.DataFrame, n_per_label: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for lab in (0, 1):
        sub = df[df["label"] == lab]
        parts.append(sub.sample(n=min(n_per_label, len(sub)), random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts)


# ---------------------------------------------------------------- email

def _email_rows_to_records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for r in df.itertuples(index=False):
        ts = r.date.to_pydatetime() if pd.notna(r.date) else None
        body = r.text if isinstance(r.text, str) else ""
        subject = r.subject if isinstance(r.subject, str) else ""
        text = f"Subject: {subject}\n\n{body}" if subject else body
        sender = r.sender if isinstance(r.sender, str) else None
        meta = email_meta(sender, subject, body, ts)
        # `receiver` and `dataset_name` are deliberately not used: both identify the source corpus.
        recs.append(make_record("email", text, meta, r.label, f"{EMAIL_DS}:{r.dataset_name}", ts))
    return recs


def build_email(rng) -> tuple[dict[str, list[dict]], list[dict]]:
    df = _load_all_splits(EMAIL_DS)
    df = df[df["text"].fillna("").str.strip().str.len() > 20].copy()
    df["label"] = df["label"].astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["hash"] = df["text"].map(norm_hash)
    df = df.drop_duplicates("hash")
    year = df["date"].dt.year
    is_ceas = df["dataset_name"] == "CEAS-08"
    # Train/val/test come from the pre-2008 corpora; the whole CEAS-08 corpus is held out and its
    # 2008-dated rows form the temporal test. Bogus dates (2038/2099/2100) fall in neither pool.
    in_pool = df[~is_ceas & ((year < 2008) | df["date"].isna())]
    temporal_pool = df[is_ceas & (year == 2008)]
    splits = _sample_splits(in_pool, rng)
    used = set(pd.concat(splits.values())["hash"])
    temporal = _balanced(temporal_pool[~temporal_pool["hash"].isin(used)], TEMPORAL_PER_LABEL, rng)
    return {s: _email_rows_to_records(d) for s, d in splits.items()}, _email_rows_to_records(temporal)


# ---------------------------------------------------------------- sms

def build_sms(rng) -> dict[str, list[dict]]:
    raw = _load_all_splits(SMS_DS)
    df = pd.DataFrame({"text": raw["sms"].astype(str).str.strip(), "label": raw["label"].astype(int)})
    df["hash"] = df["text"].map(norm_hash)
    df = df.drop_duplicates("hash")
    splits = _sample_splits(df, rng)
    return {
        s: [make_record("sms", r.text, sms_meta(r.text), r.label, SMS_DS) for r in d.itertuples(index=False)]
        for s, d in splits.items()
    }


# ---------------------------------------------------------------- url

def build_url(rng) -> dict[str, list[dict]]:
    df = _load_all_splits(URL_DS)
    df["label"] = (df["status"].astype(str).str.lower() == "phishing").astype(int)
    df["hash"] = df["url"].map(norm_hash)
    df = df.drop_duplicates("hash")
    df["domain"] = df["url"].map(lambda u: url_info(u)["registered_domain"])
    # Group split by registered domain so no domain appears in more than one split.
    domains = np.array(sorted(df["domain"].unique()), dtype=object)
    rng.shuffle(domains)
    counts = df["domain"].value_counts()
    total, cum, assign = len(df), 0, {}
    for d in domains:
        frac = cum / total
        assign[d] = "train" if frac < 0.70 else ("val" if frac < 0.78 else "test")
        cum += counts[d]
    df["split"] = df["domain"].map(assign)
    out = {}
    for split, n in PER_LABEL.items():
        part = _balanced(df[df["split"] == split], n, rng)
        out[split] = [
            make_record("url", r["url"], url_meta(r), r["label"], URL_DS, domain=r["domain"])
            for r in part.to_dict("records")
        ]
    return out


# ---------------------------------------------------------------- voice

def _fmt_dialogue(text: str) -> str:
    return re.sub(r"\s*\b(caller|receiver):", r"\n\1:", text or "").strip()


def build_voice(rng) -> tuple[dict[str, list[dict]], list[dict]]:
    ds = load_dataset(VOICE_DS)
    tr = ds["train"].to_pandas()
    te = ds["test"].to_pandas()
    for d in (tr, te):
        d["label"] = d["label"].astype(int)
        d["text"] = d["dialogue"].map(_fmt_dialogue)
        d["hash"] = d["text"].map(norm_hash)
    tr = tr.drop_duplicates("hash")
    te = te[~te["hash"].isin(set(tr["hash"]))].drop_duplicates("hash")
    trv = _sample_splits(tr, rng, per_label={"train": PER_LABEL["train"], "val": PER_LABEL["val"]})
    test = _balanced(te, PER_LABEL["test"], rng)

    def recs(d):
        return [
            make_record("voice", r.text, voice_meta(r.text), r.label, VOICE_DS, scam_type=r.type)
            for r in d.itertuples(index=False)
        ]

    out = {"train": recs(trv["train"]), "val": recs(trv["val"]), "test": recs(test)}

    # Early-warning set: the same test calls cut after 25/50/75% of turns.
    prefix = []
    base = [r for r in out["test"] if r["label"] == 1][:VOICE_PREFIX_PER_LABEL]
    base += [r for r in out["test"] if r["label"] == 0][:VOICE_PREFIX_PER_LABEL]
    for b in base:
        turns = split_turns(b["text"])
        for frac in PREFIX_FRACS:
            k = max(1, math.ceil(frac * len(turns)))
            text = "\n".join(turns[:k])
            rec = make_record("voice", text, voice_meta(text), b["label"], VOICE_DS,
                              base_id=b["id"], prefix_frac=frac)
            rec["id"] = f"{b['id']}-p{int(frac * 100)}"
            prefix.append(rec)
    return out, prefix


def build_realcalls() -> list[dict]:
    df = _load_all_splits(REALCALLS_DS)
    return [
        make_record("voice", _fmt_dialogue(r.dialogue), voice_meta(_fmt_dialogue(r.dialogue)), int(r.labels), REALCALLS_DS)
        for r in df.itertuples(index=False)
    ]


# ---------------------------------------------------------------- out-of-distribution (recent, synthetic)

def build_ood(rng, exclude_hashes: set[str]) -> list[dict]:
    frames = [_load_all_splits(OOD_DS, cfg) for cfg in ("standard", "hard", "legit_confusable")]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["language"] == "en") & df["verdict"].isin(["scam_likely", "no_indicators"])].copy()
    df["label"] = (df["verdict"] == "scam_likely").astype(int)
    df["hash"] = df["text"].map(norm_hash)
    df = df[~df["hash"].isin(exclude_hashes)].drop_duplicates("hash")
    part = _balanced(df, OOD_PER_LABEL, rng)
    recs = []
    for r in part.itertuples(index=False):
        channel = r.channel if r.channel in ("sms", "email", "chat") else "sms"
        meta = {**link_summary(r.text), **content_cues(r.text)}
        recs.append(make_record(channel, r.text, meta, r.label, f"{OOD_DS}:{r.subset}",
                                tactics=str(r.tactics), confusable_class=str(r.confusable_class)))
    return recs


# ---------------------------------------------------------------- everything

def build_all(seed: int = SEED) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    email, temporal = build_email(rng)
    sms = build_sms(rng)
    url = build_url(rng)
    voice, voice_prefix = build_voice(rng)
    splits = {s: email[s] + sms[s] + url[s] + voice[s] for s in PER_LABEL}
    for s in splits:
        rng.shuffle(splits[s])
    splits["test_temporal"] = temporal
    seen = {r["hash"] for s in ("train", "val", "test", "test_temporal") for r in splits[s]}
    splits["test_ood"] = build_ood(rng, seen)
    splits["voice_prefix"] = voice_prefix
    splits["test_realcalls"] = build_realcalls()
    return splits
