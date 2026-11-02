"""Aggregate results/*.jsonl into metrics with 95% CIs, paired tests, results/summary.md and charts.

    uv run python -m phish.report
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .client import load_config
from .evaluate import DATA, RESULTS, read_jsonl, sha256

OURS = "ft-meta"  # replaced at runtime by pick_ours()
N_BOOT = 1000
SPLITS = ["test", "test_temporal", "test_ood", "voice_prefix", "test_realcalls"]
# Fixed categorical order (validated default palette): a model keeps its color in every chart.
COLORS = {
    "ft-nemotron-super": "#2a78d6", "ft-meta": "#2a78d6", "kimi-k3": "#eb6834", "deepseek-v4p1-flash": "#1baf7a", "nemotron-base": "#eda100",
    "ft-text": "#e87ba4", "rules": "#008300", "kimi-k3-reasoning": "#6250d6", "nemotron-base+knn": "#e34948", "cascade": "#52514e", "cascade3": "#8a6d3b", "nemotron-super-base": "#c98500",
}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
ORDER = list(COLORS)


# ---------------------------------------------------------------- loading

def load_results() -> tuple[pd.DataFrame, list[str]]:
    notes, frames = [], []
    current = {s: sha256(DATA / f"{s}.jsonl") for s in SPLITS if (DATA / f"{s}.jsonl").exists()}
    for p in sorted(RESULTS.glob("*__*__*.jsonl")):
        model, variant, tag = p.stem.split("__")
        if model.startswith("smoke"):
            continue  # plumbing checks, not results
        rows = read_jsonl(p)
        if not rows:
            continue
        split = rows[0].get("split")
        if rows[0].get("test_sha256") != current.get(split):
            notes.append(f"skipped {p.name}: built from a different {split}.jsonl")
            continue
        df = pd.DataFrame(rows)
        df["model"], df["variant"], df["tag"] = model, variant, tag
        if variant == "knn":  # retrieval-augmented harness = a separate system on the meta prompt
            df["model"], df["variant"] = f"{model}+knn", "meta"
        # resumed runs may hold an error row and a later success for the same id: keep the success
        df["_ok"] = df["error"].isna()
        df = df.sort_values("_ok").drop_duplicates("id", keep="last")
        frames.append(df)
    if not frames:
        raise SystemExit("no results yet: run phish.rules or phish.evaluate first")
    return pd.concat(frames, ignore_index=True), notes


# Routing across open models (rules fixed a priori, nothing tuned on test): cheap voters answer when
# they all agree; any disagreement escalates the item to the frontier model.
CASCADES = {
    "cascade": (("nemotron-base", "nemotron-base+knn"), "kimi-k3"),
    "cascade3": (("nemotron-base", "nemotron-base+knn", "deepseek-v4p1-flash"), "kimi-k3"),
}


def add_cascades(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta = df[df["tag"].isin(SPLITS) & (df["variant"] == "meta")]
    for name, (voters, judge) in CASCADES.items():
        for split, g in meta.groupby("split"):
            frames = [g[g["model"] == m].set_index("id") for m in (*voters, judge)]
            ids = frames[0].index
            for f in frames[1:]:
                ids = ids.intersection(f.index)
            for i in ids:
                vs = [f.loc[i] for f in frames[:-1]]
                rj = frames[-1].loc[i]
                preds = [v["pred"] for v in vs]
                esc = any(pd.isna(x) for x in preds) or len(set(preds)) > 1
                row = vs[-1].to_dict()
                row.update(
                    id=i, model=name, pred=rj["pred"] if esc else preds[0], escalated=bool(esc),
                    parse_ok=bool(rj["parse_ok"] if esc else vs[0]["parse_ok"]),
                    cost_usd=sum(v["cost_usd"] for v in vs) + (rj["cost_usd"] if esc else 0.0),
                    latency_s=max(v["latency_s"] for v in vs) + (rj["latency_s"] if esc else 0.0),
                    prompt_tokens=sum(v["prompt_tokens"] for v in vs) + (rj["prompt_tokens"] if esc else 0),
                    completion_tokens=sum(v["completion_tokens"] for v in vs) + (rj["completion_tokens"] if esc else 0),
                    reasoning_tokens=0, server_ttft_s=None,
                )
                rows.append(row)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if rows else df


# ---------------------------------------------------------------- metrics

def _counts(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p != 1) & (y == 1)).sum())
    neg = int((y == 0).sum())
    false_alarm = int(((p != 0) & (y == 0)).sum())  # unparseable negatives count as false alarms
    return {"tp": tp, "fp": fp, "fn": fn, "neg": neg, "false_alarm": false_alarm}


def _f1(y: np.ndarray, p: np.ndarray) -> float:
    c = _counts(y, p)
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _boot(y: np.ndarray, p: np.ndarray, fn, rng) -> tuple[float, float]:
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    vals = np.array([fn(y[i], p[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def quality(g: pd.DataFrame, rng, ci: bool = True) -> dict:
    y = g["label"].to_numpy(int)
    p = g["pred"].fillna(-1).to_numpy(int)
    c = _counts(y, p)
    acc_fn = lambda yy, pp: float((yy == pp).mean())
    out = {
        "n": len(g),
        "acc": acc_fn(y, p),
        "f1": _f1(y, p),
        "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0,
        "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0,
        "fpr": c["false_alarm"] / c["neg"] if c["neg"] else float("nan"),
        "parse_fail": float((~g["parse_ok"].astype(bool)).mean()),
        "errors": int(g["error"].notna().sum()),
    }
    if ci and len(g) >= 20:
        out["acc_ci"] = _boot(y, p, acc_fn, rng)
        out["f1_ci"] = _boot(y, p, _f1, rng)
    return out


def efficiency(g: pd.DataFrame, m: dict, throughput: dict | None) -> dict:
    out = {
        "in_tokens": float(g["prompt_tokens"].mean()),
        "out_tokens": float(g["completion_tokens"].mean()),
        "reasoning_tokens": float(g["reasoning_tokens"].mean()),
    }
    if str(m.get("serving", "")).startswith("serverless"):
        out["usd_per_1m_msgs"] = float(g["cost_usd"].mean() * 1e6)
        out["cost_basis"] = "serverless per-token price"
    elif m.get("serving") == "dedicated" and throughput:
        out["usd_per_1m_msgs"] = m["gpu_hour_usd"] / (throughput["rps"] * 3600) * 1e6
        out["cost_basis"] = f"${m['gpu_hour_usd']}/GPU-h at measured {throughput['rps']:.1f} req/s (concurrency {throughput['concurrency']})"
    elif m.get("serving") == "local":
        out["usd_per_1m_msgs"] = 0.0
        out["cost_basis"] = "local CPU"
    else:
        out["usd_per_1m_msgs"] = float("nan")
        out["cost_basis"] = "dedicated: run --mode throughput to cost it"
    return out


def latency(g: pd.DataFrame, budgets: dict) -> dict:
    lat = g["latency_s"].dropna()
    if lat.empty:
        return {}
    within = g.apply(lambda r: r["latency_s"] is not None and r["latency_s"] <= budgets.get(r["channel"], 1.0), axis=1)
    ttft = g["server_ttft_s"].dropna() if "server_ttft_s" in g else pd.Series(dtype=float)
    return {
        "p50_s": float(lat.quantile(0.5)),
        "p95_s": float(lat.quantile(0.95)),
        "within_budget": float(within.mean()),
        "server_ttft_p50_s": float(ttft.quantile(0.5)) if not ttft.empty else None,
    }


def paired(a: pd.DataFrame, b: pd.DataFrame, rng) -> dict:
    """Ours (a) vs baseline (b) on the identical ids: McNemar exact test + paired-bootstrap dF1."""
    j = a[["id", "label", "pred"]].merge(b[["id", "pred"]], on="id", suffixes=("_a", "_b"))
    y = j["label"].to_numpy(int)
    pa = j["pred_a"].fillna(-1).to_numpy(int)
    pb = j["pred_b"].fillna(-1).to_numpy(int)
    ca, cb = pa == y, pb == y
    b10, b01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pval = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    d = np.array([_f1(y[i], pa[i]) - _f1(y[i], pb[i]) for i in idx])
    return {"n": len(y), "ours_only_correct": b10, "baseline_only_correct": b01, "mcnemar_p": float(pval),
            "delta_f1": _f1(y, pa) - _f1(y, pb),
            "delta_f1_ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))}


# ---------------------------------------------------------------- formatting

def _ci(v, ci=None, pct=True) -> str:
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(v)} [{f(ci[0])}, {f(ci[1])}]" if ci else f(v)


def _size(m: dict) -> str:
    t, a = m.get("params_total_b"), m.get("params_active_b")
    if t is None:
        return "n/a"
    t_s = f"{t / 1000:.2f}T" if t >= 1000 else f"{t:g}B"
    return f"{t_s} / {a:g}B" if a else t_s


def _money(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"${v:,.2f}" if v >= 1 else f"${v:.3f}"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- report

def main() -> None:
    cfg = load_config()
    global OURS
    models = {**cfg["models"], "rules": {"label": "Rules on metadata (no model)", "serving": "local",
                                          "params_total_b": None, "role": "baseline"}}
    budgets = cfg["latency_budget_s"]
    rng = np.random.default_rng(0)
    df, notes = load_results()
    df = add_cascades(df)
    throughput = {}
    for p in RESULTS.glob("*__throughput.json"):
        t = json.loads(p.read_text())
        throughput[t["model"]] = t

    for name, (voters, judge) in CASCADES.items():
        sel = (df["model"] == name) & (df["split"] == "test")
        rate = df.loc[sel, "escalated"].mean() if "escalated" in df and sel.any() else float("nan")
        who = " + ".join("DeepSeek" if "deepseek" in v else ("Nemotron+harness" if v.endswith("+knn") else "Nemotron") for v in voters)
        models[name] = {"label": f"Router: {who} agree, else Kimi K3 ({100 * rate:.0f}% escalated)", "serving": "serverless",
                        "params_total_b": None, "role": "ours-router"}
    for k in sorted(set(df["model"])):
        if k.endswith("+knn") and k not in models:
            base = models.get(k[:-4], {})
            models[k] = {**base, "label": base.get("label", k[:-4]) + " + retrieval harness", "role": "ours-harness"}
    OURS = os.environ.get("PHISH_OURS") or next(
        (k for k in ("ft-nemotron-super", "ft-meta", "nemotron-base+knn") if k in set(df.loc[df["split"] == "test", "model"])), "ft-meta")
    if OURS in COLORS and OURS != "ft-meta":  # the hero series wears the first categorical slot
        COLORS[OURS], COLORS["ft-meta"] = COLORS["ft-meta"], COLORS[OURS]
    if not models.get(OURS, {}).get("label", "").endswith("[ours]"):
        models[OURS] = {**models.get(OURS, {}), "label": models.get(OURS, {}).get("label", OURS) + " [ours]"}

    def label(k):
        return models.get(k, {}).get("label", k)

    ev = df[df["tag"].isin(SPLITS)]
    test = ev[(ev["split"] == "test")]
    present = [k for k in ORDER if k in set(test["model"])] + sorted(set(test["model"]) - set(ORDER))
    present = sorted(present, key=lambda k: k != OURS)

    metrics: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "models": {}, "paired": {},
                     "notes": notes}
    head_rows = []
    for k in present:
        g = test[(test["model"] == k) & (test["variant"] == "meta")]
        if g.empty:
            g = test[test["model"] == k]
        q = quality(g, rng)
        e = efficiency(g, models.get(k, {}), throughput.get(k))
        lat_rows = df[(df["model"] == k) & (df["tag"] == "latency")]
        lat = latency(lat_rows if not lat_rows.empty else g, budgets)
        if models.get(k, {}).get("serving") == "serverless-training":
            lat = {"basis": "n/a: sampled inside a training session, not a serving endpoint"}
        lat.setdefault("basis", "concurrency 1 (latency run)" if not lat_rows.empty else "eval run (concurrent)")
        per_ch = {ch: quality(gc, rng, ci=False)["f1"] for ch, gc in g.groupby("channel")}
        metrics["models"][k] = {"label": label(k), "size": _size(models.get(k, {})), "serving": models.get(k, {}).get("serving"),
                                "variant": g["variant"].iloc[0], "quality": q, "efficiency": e, "latency": lat,
                                "per_channel_f1": per_ch}
        head_rows.append(k)

    # paired comparisons: ours vs every other model on the same test ids
    ours = test[(test["model"] == OURS) & (test["variant"] == "meta")]
    if not ours.empty:
        for k in head_rows:
            if k == OURS:
                continue
            other = test[(test["model"] == k) & (test["variant"] == "meta")]
            if other.empty:
                other = test[test["model"] == k]
            metrics["paired"][k] = paired(ours, other, rng)

    # timing: temporal drift on email, voice early warning, metadata ablation, OOD urgency subset
    temporal, ablation, early, ood, realcalls = {}, {}, {}, {}, {}
    for k in head_rows:
        for v in ("meta", "text"):
            a = test[(test["model"] == k) & (test["variant"] == v) & (test["channel"] == "email")]
            b = ev[(ev["model"] == k) & (ev["variant"] == v) & (ev["split"] == "test_temporal")]
            if not a.empty and not b.empty:
                temporal[f"{k}/{v}"] = {"f1_test_email": _f1(a["label"].to_numpy(int), a["pred"].fillna(-1).to_numpy(int)),
                                        "f1_2008": quality(b, rng)["f1"], "f1_2008_ci": quality(b, rng).get("f1_ci")}
        gm = test[(test["model"] == k) & (test["variant"] == "meta")]
        gt = test[(test["model"] == k) & (test["variant"] == "text")]
        if not gm.empty and not gt.empty:
            ablation[k] = {"f1_meta": quality(gm, rng, ci=False)["f1"], "f1_text": quality(gt, rng, ci=False)["f1"]}
        pre = ev[(ev["model"] == k) & (ev["split"] == "voice_prefix") & (ev["variant"] == "meta")]
        if len(pre) >= 270:  # only complete runs: a partial prefix set gives misleading per-step rates
            curve = {}
            for frac, gp in pre.groupby("prefix_frac"):
                q = quality(gp, rng, ci=False)
                curve[f"{int(frac * 100)}%"] = {"recall": q["recall"], "fpr": q["fpr"]}
            full = test[(test["model"] == k) & (test["variant"] == "meta") & test["id"].isin(set(pre["base_id"]))]
            if not full.empty:
                q = quality(full, rng, ci=False)
                curve["100%"] = {"recall": q["recall"], "fpr": q["fpr"]}
            early[k] = curve
        go = ev[(ev["model"] == k) & (ev["split"] == "test_ood") & (ev["variant"] == "meta")]
        if not go.empty:
            urg = go[go["tactics"].fillna("").str.contains("urgency_pressure") | (go["label"] == 0)]
            ood[k] = {**quality(go, rng), "urgency_subset_acc": quality(urg, rng, ci=False)["acc"] if not urg.empty else None,
                      "hard_negative_fpr": quality(go[go["confusable_class"].fillna("None") != "None"], rng, ci=False)["fpr"]
                      if (go["confusable_class"].fillna("None") != "None").any() else None}
        gr = ev[(ev["model"] == k) & (ev["split"] == "test_realcalls") & (ev["variant"] == "meta")]
        if not gr.empty:
            realcalls[k] = quality(gr, rng, ci=False)["recall"]
    metrics.update(temporal=temporal, ablation=ablation, early_warning=early, ood=ood, realcalls_recall=realcalls)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    charts = make_charts(metrics, head_rows, label)
    write_summary(metrics, head_rows, label, charts, budgets)
    print(f"wrote {RESULTS / 'summary.md'}, metrics.json and {len(charts)} charts")
    for n in notes:
        print("note:", n)


def make_charts(metrics: dict, keys: list[str], label) -> list[str]:
    # charts show the core systems; router and harness variants stay in the tables
    keys = [k for k in keys if not k.startswith("cascade") and not k.endswith("+knn")]
    out_dir = RESULTS / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    made = []
    M = metrics["models"]

    # 1. F1 with 95% CI on the shared test set
    ks = [k for k in keys if "f1_ci" in M[k]["quality"]]
    if ks:
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(ks) + 1.4))
        ax.set_xlim(min(100 * M[k]["quality"]["f1_ci"][0] for k in ks) - 2, 101.5)
        for i, k in enumerate(reversed(ks)):
            q = M[k]["quality"]
            lo, hi = q["f1_ci"]
            ax.plot([100 * lo, 100 * hi], [i, i], color=COLORS.get(k, INK2), linewidth=2, solid_capstyle="round")
            ax.plot(100 * q["f1"], i, "o", color=COLORS.get(k, INK2), markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(100 * hi + 0.6, i, f"{100 * q['f1']:.1f}", va="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(ks)), [label(k) for k in reversed(ks)], color=INK)
        _style(ax, f"F1 on the shared held-out test set (n={M[ks[0]]['quality']['n']})", "F1 % (line = 95% bootstrap CI)")
        fig.tight_layout()
        fig.savefig(out_dir / "f1_ci.png", dpi=160)
        plt.close(fig)
        made.append("charts/f1_ci.png")

    # 2. cost vs quality (one log x-axis, one linear y-axis)
    ks = [k for k in keys if M[k]["efficiency"].get("usd_per_1m_msgs", 0) and not np.isnan(M[k]["efficiency"]["usd_per_1m_msgs"])]
    if ks:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for k in ks:
            x, y = M[k]["efficiency"]["usd_per_1m_msgs"], 100 * M[k]["quality"]["f1"]
            ax.plot(x, y, "o", color=COLORS.get(k, INK2), markersize=10, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(label(k), (x, y), textcoords="offset points", xytext=(8, 4), color=INK, fontsize=9)
        ax.set_xscale("log")
        xs = [M[k]["efficiency"]["usd_per_1m_msgs"] for k in ks]
        ax.set_xlim(min(xs) / 2, max(xs) * 4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:.2f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        _style(ax, "Quality vs. cost per 1M messages (up and left is better)", "$ per 1M classifications (log scale)", "F1 on test (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "cost_vs_f1.png", dpi=160)
        plt.close(fig)
        made.append("charts/cost_vs_f1.png")

    # 3. latency p50 (dot) to p95 (line end)
    ks = [k for k in keys if M[k]["latency"].get("p50_s") is not None and k != "rules"]
    if ks:
        fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(ks) + 1.4))
        for i, k in enumerate(reversed(ks)):
            L = M[k]["latency"]
            ax.plot([L["p50_s"], L["p95_s"]], [i, i], color=COLORS.get(k, INK2), linewidth=2, solid_capstyle="round")
            ax.plot(L["p50_s"], i, "o", color=COLORS.get(k, INK2), markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(L["p95_s"] * 1.04, i, f"p50 {L['p50_s']:.2f}s · p95 {L['p95_s']:.2f}s", va="center", color=INK, fontsize=9)
        ax.axvline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_ylim(-0.9, len(ks) - 0.5)
        ax.text(1.0, -0.75, " 1 s budget (SMS, calls)", color=INK2, fontsize=8, va="center")
        ax.set_xscale("log")
        ax.set_xlim(right=max(M[k]["latency"]["p95_s"] for k in ks) * 6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}s"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(range(len(ks)), [label(k) for k in reversed(ks)], color=INK)
        _style(ax, "Latency per classification (dot = p50, line end = p95)", "seconds (log scale)")
        fig.tight_layout()
        fig.savefig(out_dir / "latency.png", dpi=160)
        plt.close(fig)
        made.append("charts/latency.png")

    # 4. voice early warning: detection rate vs share of the call heard
    E = metrics.get("early_warning") or {}
    if E:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for k in [k for k in keys if k in E]:
            pts = sorted(((int(f[:-1]), v["recall"]) for f, v in E[k].items()))
            ax.plot([p for p, _ in pts], [100 * r for _, r in pts], "-o", color=COLORS.get(k, INK2), linewidth=2,
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label(k))
        ax.set_xticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"])
        ax.set_ylim(0, 105)
        ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
        _style(ax, "Scam calls caught vs. how much of the call was heard", "share of call transcript seen", "scam calls flagged (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "voice_early_warning.png", dpi=160)
        plt.close(fig)
        made.append("charts/voice_early_warning.png")

    # 5. temporal drift: in-distribution email F1 -> 2008 CEAS-08 F1 (dumbbell)
    T = {k: v for k, v in (metrics.get("temporal") or {}).items() if k.endswith("/meta")}
    if T:
        ks = [k for k in keys if f"{k}/meta" in T]
        fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(ks) + 1.4))
        for i, k in enumerate(reversed(ks)):
            t = T[f"{k}/meta"]
            a, b = 100 * t["f1_test_email"], 100 * t["f1_2008"]
            ax.plot([a, b], [i, i], color=COLORS.get(k, INK2), linewidth=2)
            ax.plot(a, i, "o", color=SURFACE, markeredgecolor=COLORS.get(k, INK2), markersize=9, markeredgewidth=2)
            ax.plot(b, i, "o", color=COLORS.get(k, INK2), markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(max(a, b) + 0.8, i, f"{a:.1f} → {b:.1f}", va="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(ks)), [label(k) for k in reversed(ks)], color=INK)
        _style(ax, "Temporal drift: email F1, pre-2008 test (hollow) → unseen 2008 corpus (filled)", "F1 (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "temporal_drift.png", dpi=160)
        plt.close(fig)
        made.append("charts/temporal_drift.png")
    return made


def write_summary(metrics: dict, keys: list[str], label, charts: list[str], budgets: dict) -> None:
    M, P = metrics["models"], metrics["paired"]
    L = [f"# Results\n\n_Auto-generated by `make report` at {metrics['generated_at']}. Every model was scored on the "
         f"same frozen, hash-checked test files; 95% CIs are percentile bootstraps ({N_BOOT} resamples)._\n"]
    L.append("## Headline: shared held-out test set (600 cases: email, SMS, URL, call transcripts)\n")
    L.append("| Model | Size (total / active) | Serving | Accuracy % [95% CI] | F1 % [95% CI] | Precision % | Recall % | FPR % | Parse fail % | In / out tokens | $ per 1M msgs | Latency p50 / p95 (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        x = M[k]
        q, e, lt = x["quality"], x["efficiency"], x["latency"]
        lat = f"{lt['p50_s']:.2f} / {lt['p95_s']:.2f}" if lt.get("p50_s") is not None else "n/a"
        L.append(f"| {x['label']} | {x['size']} | {x['serving']} | {_ci(q['acc'], q.get('acc_ci'))} | {_ci(q['f1'], q.get('f1_ci'))} "
                 f"| {100 * q['precision']:.1f} | {100 * q['recall']:.1f} | {100 * q['fpr']:.1f} | {100 * q['parse_fail']:.1f} "
                 f"| {e['in_tokens']:.0f} / {e['out_tokens']:.1f} | {_money(e.get('usd_per_1m_msgs'))} | {lat} |")
    if P:
        L.append(f"\n## Ours ({label(OURS)}) vs. each baseline: paired on identical cases\n")
        L.append("| Baseline | n | ΔF1 (pts) [95% CI] | Only ours right | Only baseline right | McNemar exact p |")
        L.append("|---|---|---|---|---|---|")
        for k, pr in P.items():
            lo, hi = pr["delta_f1_ci"]
            L.append(f"| {label(k)} | {pr['n']} | {100 * pr['delta_f1']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] "
                     f"| {pr['ours_only_correct']} | {pr['baseline_only_correct']} | {pr['mcnemar_p']:.3g} |")
    chans = sorted({c for k in keys for c in M[k]["per_channel_f1"]})
    L.append("\n## F1 % by channel (test)\n")
    L.append("| Model | " + " | ".join(chans) + " |")
    L.append("|---|" + "---|" * len(chans))
    for k in keys:
        L.append(f"| {M[k]['label']} | " + " | ".join(f"{100 * M[k]['per_channel_f1'].get(c, float('nan')):.1f}" for c in chans) + " |")

    L.append("\n## Timing\n")
    T = metrics.get("temporal") or {}
    if T:
        L.append("**Temporal drift (email):** trained/tested on pre-2008 corpora, then scored on the unseen 2008 CEAS corpus (n=200).\n")
        L.append("| Model / prompt | F1 % pre-2008 test | F1 % 2008 [95% CI] | Δ |")
        L.append("|---|---|---|---|")
        for k, t in T.items():
            mk, v = k.split("/")
            L.append(f"| {label(mk)} / {v} | {100 * t['f1_test_email']:.1f} | {_ci(t['f1_2008'], t.get('f1_2008_ci'))} "
                     f"| {100 * (t['f1_2008'] - t['f1_test_email']):+.1f} |")
    E = metrics.get("early_warning") or {}
    if E:
        L.append("\n**Early warning on calls:** scam calls flagged (recall) / benign calls falsely flagged (FPR) after hearing part of the call (100 test calls).\n")
        L.append("| Model | 25% | 50% | 75% | 100% |")
        L.append("|---|---|---|---|---|")
        for k, curve in E.items():
            cells = [f"{100 * curve[f]['recall']:.0f} / {100 * curve[f]['fpr']:.0f}" if f in curve else "n/a"
                     for f in ("25%", "50%", "75%", "100%")]
            L.append(f"| {label(k)} | " + " | ".join(cells) + " |")
    L.append(f"\n**Latency budgets** (share of requests answered within: " + ", ".join(f"{c} ≤ {b}s" for c, b in budgets.items()) + "):\n")
    L.append("| Model | p50 (s) | p95 (s) | Within budget % | Server TTFT p50 (s) | Measured on |")
    L.append("|---|---|---|---|---|---|")
    for k in keys:
        lt = M[k]["latency"]
        if lt.get("p50_s") is None or k == "rules":
            continue
        ttft = f"{lt['server_ttft_p50_s']:.3f}" if lt.get("server_ttft_p50_s") else "n/a"
        L.append(f"| {M[k]['label']} | {lt['p50_s']:.2f} | {lt['p95_s']:.2f} | {100 * lt['within_budget']:.0f} | {ttft} | {lt['basis']} |")

    A = metrics.get("ablation") or {}
    if A:
        L.append("\n## Metadata ablation (test F1 %)\n")
        L.append("| Model | With metadata | Text only | Δ |")
        L.append("|---|---|---|---|")
        for k, a in A.items():
            L.append(f"| {label(k)} | {100 * a['f1_meta']:.1f} | {100 * a['f1_text']:.1f} | {100 * (a['f1_meta'] - a['f1_text']):+.1f} |")
    O = metrics.get("ood") or {}
    if O:
        L.append("\n## Out-of-distribution: recent synthetic scams (ScamGuardBench, English)\n")
        L.append("| Model | n | Accuracy % [95% CI] | F1 % | Urgency-tactic subset acc % | Hard-negative FPR % |")
        L.append("|---|---|---|---|---|---|")
        for k, o in O.items():
            u = f"{100 * o['urgency_subset_acc']:.1f}" if o.get("urgency_subset_acc") is not None else "n/a"
            h = f"{100 * o['hard_negative_fpr']:.1f}" if o.get("hard_negative_fpr") is not None else "n/a"
            L.append(f"| {label(k)} | {o['n']} | {_ci(o['acc'], o.get('acc_ci'))} | {100 * o['f1']:.1f} | {u} | {h} |")
    R = metrics.get("realcalls_recall") or {}
    if R:
        L.append("\n**Real scam-baiting calls (n=20, all scams; recall only):** " +
                 ", ".join(f"{label(k)} {100 * v:.0f}%" for k, v in R.items()))
    if charts:
        L.append("\n## Charts\n")
        L += [f"![{c}]({c})" for c in charts]
    L.append("\n## Measurement notes\n")
    L.append("- Cost per 1M messages: serverless rows use Fireworks per-token prices from `config/models.yaml`; dedicated rows "
             "divide the GPU-hour price by measured throughput (`--mode throughput`), so they assume a fully used deployment.")
    L.append("- Latency: wall-clock from the client, including network. Serverless endpoints are shared; the fine-tuned model runs on a "
             "dedicated deployment, so the serving mode is listed on every row.")
    L.append("- Unparseable outputs count as wrong (a false alarm on benign items, a miss on malicious ones).")
    L.append("- `serverless-training` rows were sampled inside a Fireworks Serverless Training session (greedy decoding, no "
             "deployment): their cost uses published per-token sampling prices and their latency is indicative only.")
    for n in metrics.get("notes", []):
        L.append(f"- {n}")
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
