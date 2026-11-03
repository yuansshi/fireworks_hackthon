"""Round-3 extra training data: recent-style emails plus hard legitimate look-alikes, from a source that
none of the test sets use.

    uv run python scripts/build_extra.py

Source: AreLit/PhishNChips (synthetic emails, MIT; embedded third-party URL indicators are cleared for
academic research with attribution, see the dataset's SOURCE_LICENSES.md). Writes data/extra_train.jsonl
(records) and data/ft_{meta,text}_extra.jsonl (fine-tuning rows). Frozen files are not touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phish.data import make_record
from phish.features import email_meta
from phish.prompt import build_messages, label_json

from datasets import load_dataset  # after phish.data, which disables the blocked Xet CDN

DATA = Path(__file__).resolve().parents[1] / "data"
SRC = "AreLit/PhishNChips"
N_PHISH, N_LEGIT, N_HARD = 300, 150, 150


def to_records(df: pd.DataFrame, split: str) -> list[dict]:
    recs = []
    for r in df.itertuples(index=False):
        mail = json.loads(r.email_content)
        subject, body = mail.get("subject") or "", mail.get("body") or ""
        if r.url_raw and r.url_raw not in body:
            body = f"{body}\n\n{r.url_raw}"  # the link the email points to, as a recipient would see it
        sender = f"{mail.get('sender', '')} <{mail.get('from', '')}>"
        text = f"Subject: {subject}\n\n{body}"
        recs.append(make_record("email", text, email_meta(sender, subject, body, None), int(r.phish_label),
                                f"{SRC}:{split}", strategy=str(r.strategy)))
    return recs


def main() -> None:
    rng = np.random.default_rng(3)
    used = set()
    for path in DATA.glob("*.jsonl"):
        if not path.stem.startswith(("ft_", "extra_")):
            used |= {json.loads(line)["hash"] for line in path.open()}
    ds = load_dataset(SRC, "emails")
    core = ds["core"].to_pandas()  # balanced phishing / legitimate
    core["label"] = core["phish_label"].astype(int)
    hard = ds["cross_domain_legitimate_v5"].to_pandas()  # legitimate, but the link domain differs from the sender's
    hard["label"] = 0
    phish = core[core["label"] == 1].sample(n=N_PHISH, random_state=int(rng.integers(1 << 31)))
    legit = core[core["label"] == 0].sample(n=N_LEGIT, random_state=int(rng.integers(1 << 31)))
    hard = hard.sample(n=min(N_HARD, len(hard)), random_state=int(rng.integers(1 << 31)))
    recs = to_records(phish, "core") + to_records(legit, "core") + to_records(hard, "cross_domain_legitimate_v5")
    recs = [r for r in recs if r["hash"] not in used]
    rng.shuffle(recs)
    with (DATA / "extra_train.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    for variant in ("meta", "text"):
        with (DATA / f"ft_{variant}_extra.jsonl").open("w") as f:
            for r in recs:
                row = {"messages": build_messages(r, variant) + [{"role": "assistant", "content": label_json(r["label"])}]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pos = sum(r["label"] for r in recs)
    print(f"extra_train: {len(recs)} rows ({pos} phishing, {len(recs) - pos} legitimate incl. hard negatives), "
          f"0 overlap with frozen splits")


if __name__ == "__main__":
    main()
