"""One prompt template for every model and channel, in two variants: meta and text-only."""

from __future__ import annotations

import json
import re

LABELS = ("malicious", "benign")

SYSTEM_PROMPT = (
    "You are a fraud-screening classifier for a security team. You receive one item: an email, "
    "an SMS or chat message, a URL with page/domain facts, or a phone-call transcript.\n"
    'Label it "malicious" if it is phishing, a scam, fraud, or unsolicited bulk spam that tries to '
    "get the recipient to click, pay, reply, call back, install something, or disclose information. "
    'Otherwise label it "benign".\n'
    "When a Metadata block is present, use it together with the content: it lists sender/link/domain "
    "facts and timing cues such as urgency language, deadlines, send time, and domain age.\n"
    'Respond with JSON only: {"label": "malicious"} or {"label": "benign"}.'
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "schema": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(LABELS)}},
            "required": ["label"],
        },
    },
}

CHANNEL_NAMES = {
    "email": "email",
    "sms": "SMS",
    "chat": "chat message",
    "url": "URL",
    "voice": "phone-call transcript",
}

MAX_CONTENT_CHARS = 2000  # ~500 tokens


def _fmt_value(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return ", ".join(f'"{x}"' if " " in str(x) else str(x) for x in v) if v else "none"
    return str(v)


def format_meta(meta: dict) -> str:
    lines = []
    for k, v in meta.items():
        if v in (None, "", [], False) and k not in ("n_urls",):
            continue  # keep the block short: only facts that are present
        lines.append(f"- {k}: {_fmt_value(v)}")
    return "\n".join(lines) if lines else "- (no notable signals)"


def truncate(text: str, limit: int = MAX_CONTENT_CHARS) -> str:
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text if len(text) <= limit else text[:limit].rstrip() + " …[truncated]"


def user_message(rec: dict, variant: str) -> str:
    channel = CHANNEL_NAMES.get(rec["channel"], rec["channel"])
    parts = [f"Channel: {channel}"]
    if variant == "meta":
        parts.append("Metadata:\n" + format_meta(rec.get("meta") or {}))
    parts.append("Content:\n" + truncate(rec["text"]))
    return "\n".join(parts)


def build_messages(rec: dict, variant: str, system_suffix: str = "") -> list[dict]:
    system = SYSTEM_PROMPT + (f"\n{system_suffix}" if system_suffix else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(rec, variant)},
    ]


def label_json(label: int) -> str:
    return json.dumps({"label": LABELS[0] if label == 1 else LABELS[1]})


def parse_label(raw: str | None) -> int | None:
    """1 = malicious, 0 = benign, None = unparseable (counted as wrong in the report)."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("label") in LABELS:
            return 1 if obj["label"] == "malicious" else 0
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\b(malicious|benign)\b", raw.lower())
    if m:
        return 1 if m.group(1) == "malicious" else 0
    return None
