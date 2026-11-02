"""Cheap, offline metadata and timing features.

Everything here is derived from fields the dataset already ships with (no WHOIS or
URL fetching), so features are free, fast, and reproducible.
"""

from __future__ import annotations

import re
from datetime import datetime
from email.utils import parseaddr
from functools import lru_cache

import tldextract

# Bundled public-suffix snapshot only, so feature extraction never touches the network.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>\"'()\[\]]+"
    r"|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:com|net|org|info|biz|ru|cn|xyz|top|co|io|ly|me|us|uk|de|tk|ml|ga|cf|gq|click|link|online|site)"
    r"(?:/[^\s<>\"'()\[\]]*)?"
)
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

SHORTENERS = {
    "bit.ly", "bitly.com", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "tiny.cc", "rb.gy", "shorturl.at", "lnkd.in", "s.id", "t.ly",
}
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "ru", "cn", "info", "biz", "click", "link",
    "work", "loan", "win", "zip", "mov", "icu", "buzz", "rest", "country", "online", "site",
}
BRANDS = [
    "paypal", "apple", "amazon", "microsoft", "netflix", "chase", "wellsfargo", "citibank",
    "irs", "usps", "fedex", "dhl", "ups", "google", "facebook", "instagram", "whatsapp",
    "outlook", "office365", "docusign", "coinbase", "binance", "ebay", "bankofamerica",
]

URGENCY_PHRASES = [
    "urgent", "immediately", "right away", "as soon as possible", "asap", "act now",
    "final notice", "last chance", "expires", "expire", "suspended", "suspension",
    "locked", "deactivated", "unauthorized", "verify your", "confirm your", "within 24 hours",
    "within 48 hours", "today only", "limited time", "don't delay", "respond now",
    "account will be", "failure to", "legal action", "arrest", "warrant",
]
DEADLINE_RE = re.compile(
    r"(?i)\b(?:within|in the next|before|by)\s+\d+\s*(?:minutes?|mins?|hours?|hrs?|days?)\b"
)
MONEY_RE = re.compile(
    r"(?i)(?:[$£€]\s?\d|\b(?:usd|cash|prize|won|winner|reward|refund|gift ?cards?|bitcoin|crypto|wire transfer|lottery|claim)\b)"
)
CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:password|passcode|otp|one[- ]time (?:code|password)|verification code|security code|pin|ssn|"
    r"social security(?: number)?|account number|card number|cvv|log ?in|sign ?in|credentials)\b"
)
REMOTE_RE = re.compile(r"(?i)\b(?:anydesk|teamviewer|remote access|install (?:this|the) app|download (?:this|the) app)\b")
AUTHORITY_RE = re.compile(
    r"(?i)\b(?:irs|social security administration|police|officer|federal|government|"
    r"fraud (?:department|team)|security (?:department|team)|tech(?:nical)? support|customs|court)\b"
)
ATTACHMENT_RE = re.compile(r"(?i)\b(?:attach(?:ed|ment)|\.pdf|\.zip|\.docx?|\.xlsm?|\.exe|invoice)\b")


@lru_cache(maxsize=65536)
def _split_domain(host: str) -> tuple[str, str, str]:
    ext = _EXTRACT(host)
    return ext.subdomain, ext.domain, ext.suffix


def extract_urls(text: str, limit: int = 20) -> list[str]:
    return URL_RE.findall(text or "")[:limit]


def url_info(url: str) -> dict:
    """Lexical facts about one URL."""
    u = url.strip().lower()
    host = re.sub(r"^[a-z]+://", "", u).split("/")[0].split("?")[0].split("@")[-1].split(":")[0]
    sub, dom, suffix = _split_domain(host)
    registered = f"{dom}.{suffix}" if suffix else dom
    path = u[len(u.split("/")[0]):] if "://" not in u else "/".join(u.split("/")[3:])
    brand_elsewhere = any(b in (sub + " " + path) and b not in dom for b in BRANDS)
    return {
        "registered_domain": registered,
        "is_ip": bool(IP_HOST_RE.match(host)),
        "punycode": "xn--" in host,
        "shortener": registered in SHORTENERS or host in SHORTENERS,
        "suspicious_tld": suffix.split(".")[-1] in SUSPICIOUS_TLDS if suffix else False,
        "at_sign": "@" in u,
        "brand_not_in_domain": brand_elsewhere,
        "subdomain_depth": len([s for s in sub.split(".") if s]),
    }


def link_summary(text: str) -> dict:
    urls = extract_urls(text)
    infos = [url_info(u) for u in urls]
    domains = []
    for i in infos:
        if i["registered_domain"] and i["registered_domain"] not in domains:
            domains.append(i["registered_domain"])
    return {
        "n_urls": len(urls),
        "url_domains": domains[:4],
        "shortener": any(i["shortener"] for i in infos),
        "ip_url": any(i["is_ip"] for i in infos),
        "punycode": any(i["punycode"] for i in infos),
        "suspicious_tld": any(i["suspicious_tld"] for i in infos),
        "brand_not_in_domain": any(i["brand_not_in_domain"] for i in infos),
    }


def urgency_hits(text: str, limit: int = 4) -> list[str]:
    low = (text or "").lower()
    hits = [p for p in URGENCY_PHRASES if p in low]
    hits += [m.group(0).lower() for m in DEADLINE_RE.finditer(low)]
    seen: list[str] = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    return seen[:limit]


def content_cues(text: str) -> dict:
    """Scam-pattern cues shared by every channel, including the timing/urgency cues."""
    t = text or ""
    return {
        "urgency_cues": urgency_hits(t),
        "deadline": bool(DEADLINE_RE.search(t)),
        "money_or_prize": bool(MONEY_RE.search(t)),
        "credential_request": bool(CREDENTIAL_RE.search(t)),
        "remote_access": bool(REMOTE_RE.search(t)),
        "authority_claim": bool(AUTHORITY_RE.search(t)),
    }


def time_features(ts: datetime | None) -> dict:
    if ts is None:
        return {}
    off = ts.hour < 7 or ts.hour >= 21 or ts.weekday() >= 5
    return {"sent": ts.strftime("%a %H:%M"), "off_hours": off}


def email_meta(sender: str | None, subject: str | None, body: str, ts: datetime | None) -> dict:
    name, addr = parseaddr(sender or "")
    sender_domain = ""
    if "@" in addr:
        _, dom, suffix = _split_domain(addr.split("@", 1)[1].lower())
        sender_domain = f"{dom}.{suffix}" if suffix else dom
    name_low = (name or "").lower()
    display_mismatch = bool(sender_domain) and any(
        b in name_low and b not in sender_domain for b in BRANDS
    )
    links = link_summary(body)
    link_mismatch = bool(sender_domain) and any(d != sender_domain for d in links["url_domains"])
    meta = {
        "sender_domain": sender_domain or "unknown",
        "display_name_brand_mismatch": display_mismatch,
        **links,
        "link_domain_differs_from_sender": link_mismatch,
        "attachment_mention": bool(ATTACHMENT_RE.search(body or "")),
        **content_cues(f"{subject or ''}\n{body}"),
        **time_features(ts),
    }
    return meta


def sms_meta(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    return {
        **link_summary(text),
        "has_phone_number": bool(re.search(r"\b\d{5,}\b|\+\d{6,}", text)),
        "caps_ratio": round(caps, 2),
        **content_cues(text),
    }


# pirocheto/phishing-url columns we expose, with readable names. Excluded on purpose:
# page_rank, google_index, web_traffic, statistical_report (near-label proxies that are
# not available at click time for a brand-new domain).
URL_FEATURES = {
    "domain_age": "domain_age_days",
    "domain_registration_length": "registration_length_days",
    "nb_subdomains": "subdomains",
    "prefix_suffix": "hyphen_in_domain",
    "shortening_service": "shortener",
    "punycode": "punycode",
    "ip": "ip_in_url",
    "suspecious_tld": "suspicious_tld",
    "phish_hints": "phish_hint_words",
    "brand_in_subdomain": "brand_in_subdomain",
    "brand_in_path": "brand_in_path",
    "nb_redirection": "redirections",
    "nb_hyperlinks": "page_hyperlinks",
    "ratio_extHyperlinks": "page_external_link_ratio",
    "login_form": "page_login_form",
    "external_favicon": "page_external_favicon",
    "iframe": "page_iframe",
    "popup_window": "page_popup",
    "empty_title": "page_empty_title",
    "submit_email": "page_form_submits_to_email",
}
URL_BOOL_FEATURES = {
    "prefix_suffix", "shortening_service", "punycode", "ip", "suspecious_tld", "brand_in_subdomain",
    "brand_in_path", "login_form", "external_favicon", "iframe", "popup_window", "empty_title",
    "submit_email",
}


def url_meta(row: dict) -> dict:
    meta = {}
    for col, name in URL_FEATURES.items():
        if col not in row or row[col] is None:
            continue
        v = row[col]
        if col in URL_BOOL_FEATURES:
            meta[name] = bool(int(v))
        elif col in ("domain_age", "domain_registration_length"):
            meta[name] = int(v) if int(v) >= 0 else "unknown"
        elif isinstance(v, float):
            meta[name] = round(v, 2)
        else:
            meta[name] = v
    return meta


def voice_meta(dialogue: str) -> dict:
    turns = split_turns(dialogue)
    caller = " ".join(t for t in turns if t.lower().startswith("caller:"))
    return {
        "turns": len(turns),
        **content_cues(caller or dialogue),
    }


TURN_RE = re.compile(r"(?i)(?=\b(?:caller|receiver|agent|customer|speaker \w+):)")


def split_turns(dialogue: str) -> list[str]:
    parts = [p.strip() for p in TURN_RE.split(dialogue or "") if p.strip()]
    return parts or [dialogue]
