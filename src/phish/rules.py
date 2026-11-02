"""Zero-cost baseline: a hand-weighted score over the metadata features, one threshold per channel
fit on the train split. It shows how far metadata alone gets, with no model at all.

    uv run python -m phish.rules
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from .evaluate import DATA, RESULTS, read_jsonl, sha256

EVAL_SPLITS = ["test", "test_temporal", "test_ood", "voice_prefix", "test_realcalls"]


def score(rec: dict) -> float:
    m = rec.get("meta") or {}
    s = 1.0 * len(m.get("urgency_cues") or [])
    s += sum(1.0 for k in ("deadline", "money_or_prize", "credential_request", "authority_claim",
                           "shortener", "ip_url", "ip_in_url", "suspicious_tld", "page_login_form",
                           "hyphen_in_domain", "brand_in_subdomain", "brand_in_path", "brand_not_in_domain") if m.get(k))
    s += 1.5 if m.get("remote_access") else 0.0
    s += 0.5 if m.get("link_domain_differs_from_sender") else 0.0
    s += 0.5 if m.get("display_name_brand_mismatch") else 0.0
    s += 0.5 * min(m.get("n_urls") or 0, 3)
    s += 0.5 if (m.get("phish_hint_words") or 0) > 0 else 0.0
    s += 0.5 if (m.get("caps_ratio") or 0) > 0.3 else 0.0
    age = m.get("domain_age_days")
    if isinstance(age, int):
        s += 1.5 if age < 365 else -1.0
    elif age == "unknown":
        s += 1.0
    return s


def fit_thresholds(train: list[dict]) -> dict[str, float]:
    th = {}
    for ch in sorted({r["channel"] for r in train}):
        rows = [(score(r), r["label"]) for r in train if r["channel"] == ch]
        cands = sorted({s for s, _ in rows})
        th[ch] = max(cands, key=lambda t: sum((s >= t) == bool(y) for s, y in rows))
    return th


def main() -> None:
    train = read_jsonl(DATA / "train.jsonl")
    th = fit_thresholds(train)
    print("thresholds:", th)
    RESULTS.mkdir(exist_ok=True)
    for split in EVAL_SPLITS:
        path = DATA / f"{split}.jsonl"
        recs = read_jsonl(path)
        if not recs:
            continue
        test_sha = sha256(path)
        out = RESULTS / f"rules__meta__{split}.jsonl"
        with out.open("w") as fh:
            for r in recs:
                t0 = time.perf_counter()
                pred = int(score(r) >= th.get(r["channel"], th.get("sms", 2.0)))
                row = {"id": r["id"], "channel": r["channel"], "label": r["label"], "pred": pred, "parse_ok": True,
                       "raw": "", "error": None, "latency_s": time.perf_counter() - t0, "prompt_tokens": 0,
                       "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0,
                       "model": "rules", "model_id": "rules", "variant": "meta", "split": split,
                       "serving": "local", "test_sha256": test_sha, "ts": datetime.now(timezone.utc).isoformat()}
                for k in ("prefix_frac", "base_id", "tactics", "confusable_class", "scam_type"):
                    if k in r:
                        row[k] = r[k]
                fh.write(json.dumps(row) + "\n")
        acc = sum(score(r) >= th.get(r["channel"], 2.0) for r in recs if r["label"] == 1)
        print(f"{split:15s} n={len(recs)} wrote {out.name}")


if __name__ == "__main__":
    main()
