"""Retrieval-augmented few-shot harness: the nearest *training* examples (same channel, balanced
by label) are shown to the model as prior turns. Only train.jsonl is indexed, so held-out cases
can never be retrieved.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .prompt import build_messages, label_json, user_message

EXAMPLE_CHARS = 600  # keep each shot short: k shots add ~k*170 input tokens
CHANNEL_POOL = {"chat": "sms"}  # channels absent from training borrow the closest pool


class Retriever:
    def __init__(self, train: list[dict]):
        self.train = train
        self.by_channel: dict[str, np.ndarray] = {}
        texts = [r["text"] for r in train]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200_000)
        self.matrix = self.vec.fit_transform(texts)
        channels = np.array([r["channel"] for r in train])
        labels = np.array([r["label"] for r in train])
        for ch in set(channels):
            for lab in (0, 1):
                self.by_channel[(ch, lab)] = np.where((channels == ch) & (labels == lab))[0]

    def neighbors(self, rec: dict, per_label: int = 3) -> list[dict]:
        ch = CHANNEL_POOL.get(rec["channel"], rec["channel"])
        q = self.vec.transform([rec["text"]])
        picked: list[tuple[float, int]] = []
        for lab in (0, 1):
            idx = self.by_channel.get((ch, lab))
            if idx is None or not len(idx):
                continue
            sims = linear_kernel(q, self.matrix[idx]).ravel()
            for j in np.argsort(-sims)[:per_label]:
                picked.append((float(sims[j]), int(idx[j])))
        picked.sort()  # most similar shot sits closest to the query
        return [self.train[i] for _, i in picked]


@lru_cache(maxsize=1)
def default_retriever() -> Retriever:
    from .evaluate import DATA

    return Retriever([json.loads(line) for line in (DATA / "train.jsonl").open()])


def build_knn_messages(rec: dict, per_label: int = 3, system_suffix: str = "") -> list[dict]:
    msgs = build_messages(rec, "meta", system_suffix)
    shots = []
    for ex in default_retriever().neighbors(rec, per_label):
        short = {**ex, "text": ex["text"][:EXAMPLE_CHARS]}
        shots += [{"role": "user", "content": user_message(short, "meta")},
                  {"role": "assistant", "content": label_json(ex["label"])}]
    return [msgs[0], *shots, msgs[1]]
