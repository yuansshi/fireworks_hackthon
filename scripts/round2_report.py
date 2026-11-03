"""Summarize post-deadline fine-tuning rounds against round 1, round 2 and the frontier baseline.

    uv run python scripts/round2_report.py                 # -> results/round2/summary.md, loss_curve.png
    uv run python scripts/round2_report.py --round round3  # -> results/round3/summary.md, loss_curve.png

Selection happened on dev data only (scripts/train_serverless.py round2); this report shows every epoch on
the frozen test sets so the choice is auditable, and paired tests use identical, hash-checked cases.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phish.evaluate import DATA, RESULTS, read_jsonl, sha256
from phish.report import INK, INK2, SURFACE, _style, paired, quality

TESTS = ["test", "test_temporal", "test_ood", "voice_prefix", "test_realcalls"]
ROUND1 = {"ft-nemotron-super": "Round 1: Nemotron Super + LoRA, ep1 (meta)",
          "nemotron-super-base": "Nemotron Super, untuned", "kimi-k3": "Kimi K3 (frontier)"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def frame(path) -> pd.DataFrame:
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_ok"] = df["error"].isna()
    return df.sort_values("_ok").drop_duplicates("id", keep="last")


def pct(x) -> str:
    return f"{100 * x:.1f}"


def main(round_dir: str) -> None:
    R2 = RESULTS / round_dir
    rng = np.random.default_rng(0)
    current = {s: sha256(DATA / f"{s}.jsonl") for s in TESTS}
    runs = sorted(d for d in R2.iterdir() if d.is_dir() and d.name != "smoke" and (d / "selection.json").exists())
    refs = dict(ROUND1)
    r1 = {k: frame(RESULTS / f"{k}__meta__test.jsonl") for k in ROUND1}
    best_r2 = RESULTS / "round2" / "text-r16" / "text-r16-ep1__text__test.jsonl"
    if round_dir != "round2" and best_r2.exists():  # later rounds also compare with round 2's best model
        refs["round2-best"] = "Round 2 best: text-only ep1"
        r1["round2-best"] = frame(best_r2)
    L = [f"# {round_dir.replace('round', 'Round ')} fine-tuning (post-deadline follow-up; not part of the hackathon submission)", "",
         "_Checkpoints were selected on dev data only: the 200-case validation split plus two fresh dev slices "
         "(200 later-corpus emails, 169 recent synthetic scams) that are disjoint from every frozen test file. "
         "Every epoch is still reported on the frozen tests so the choice can be audited._", ""]
    notes, selected = [], {}
    for run in runs:
        st = json.loads((run / "settings.json").read_text())
        sel = json.loads((run / "selection.json").read_text())
        best = sel.get("selected_epoch")
        variant = st["variant"]
        extra = f", +{st['extra_examples']} extra rows" if st.get("extra_examples") else ""
        L += [f"## Run `{run.name}`: {variant} prompt, LoRA rank {st['lora_rank']} (alpha {st['lora_alpha']}), "
              f"lr {st['learning_rate']} ({st['lr_schedule']}), batch {st['batch_size']}, {st['epochs']} epoch(s), "
              f"seed {st['seed']}{extra}", "",
              f"Training: {st['train_examples']} examples, {st['train_tokens_per_epoch']:,} tokens/epoch, "
              f"est. ${st['est_train_cost_usd']} · session `{st['session']}` · targets: MLP, attention, unembedding (SDK defaults)", "",
              "| Checkpoint (epoch) | Dev: val | Dev: later corpus | Dev: recent synthetic | Dev mean | Test acc % [95% CI] | Test F1 % | FPR % | Errors | Later corpus acc % | Recent synthetic acc % |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for ep, d in sorted(sel["dev"].items(), key=lambda kv: float(kv[0])):
            key = f"{run.name}-ep{ep}"
            t = {s: frame(run / f"{key}__{variant}__{s}.jsonl") for s in TESTS}
            for s, df in t.items():
                if not df.empty and df["test_sha256"].iloc[0] != current[s]:
                    notes.append(f"{key}/{s} was scored on a different {s}.jsonl")
            q = quality(t["test"], rng) if not t["test"].empty else None
            tq = quality(t["test_temporal"], rng, ci=False) if not t["test_temporal"].empty else None
            oq = quality(t["test_ood"], rng, ci=False) if not t["test_ood"].empty else None
            mark = " **(selected)**" if str(ep) == str(best) else ""
            if q:
                errs = int(round(q["n"] * (1 - q["acc"])))
                ci = f" [{pct(q['acc_ci'][0])}, {pct(q['acc_ci'][1])}]" if "acc_ci" in q else ""
                L.append(f"| {ep}{mark} | {pct(d['val'])} | {pct(d['dev_temporal'])} | {pct(d['dev_ood'])} | {pct(d['mean'])} "
                         f"| {pct(q['acc'])}{ci} | {pct(q['f1'])} | {pct(q['fpr'])} | {errs} "
                         f"| {pct(tq['acc']) if tq else 'n/a'} | {pct(oq['acc']) if oq else 'n/a'} |")
            else:
                L.append(f"| {ep}{mark} | {pct(d['val'])} | {pct(d['dev_temporal'])} | {pct(d['dev_ood'])} | {pct(d['mean'])} | pending | | | | | |")
            if str(ep) == str(best) and not t["test"].empty:
                selected[run.name] = (variant, t)
        L.append("")

    if selected:
        L += ["## Paired comparisons on the 600 frozen test cases (selected checkpoint)", "",
              "| Run (selected epoch) | vs | n | ΔF1 pts [95% CI] | Only run right | Only other right | McNemar p |",
              "|---|---|---|---|---|---|---|"]
        for name, (variant, t) in selected.items():
            for k, label in refs.items():
                if r1[k].empty:
                    continue
                pr = paired(t["test"], r1[k], rng)
                lo, hi = pr["delta_f1_ci"]
                L.append(f"| {name} | {label} | {pr['n']} | {100 * pr['delta_f1']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] "
                         f"| {pr['ours_only_correct']} | {pr['baseline_only_correct']} | {pr['mcnemar_p']:.3g} |")
        L.append("")

    # metadata ablation: text-only epoch 1 vs the meta epoch-1 checkpoints
    text_runs = [r for r in runs if json.loads((r / "settings.json").read_text())["variant"] == "text"]
    if text_runs and round_dir == "round2":
        tr = text_runs[0]
        L += ["## Metadata ablation (epoch 1, same recipe)", "",
              "| Test set | Text only (no metadata) | Meta, round 1 | Meta, round 2 run A |", "|---|---|---|---|"]
        meta_r2 = next((r for r in runs if r not in text_runs), None)
        for s in ["test", "test_temporal", "test_ood"]:
            a = frame(tr / f"{tr.name}-ep1__text__{s}.jsonl")
            b = frame(RESULTS / f"ft-nemotron-super__meta__{s}.jsonl")
            c = frame(meta_r2 / f"{meta_r2.name}-ep1__meta__{s}.jsonl") if meta_r2 else pd.DataFrame()
            cell = lambda df: pct(quality(df, rng, ci=False)["acc"]) if not df.empty else "n/a"
            L.append(f"| {s} | {cell(a)} | {cell(b)} | {cell(c)} |")
        L.append("")

    # loss curves
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, run in enumerate(runs):
        log = read_jsonl(run / "train_log.jsonl")
        if log:
            steps = [r["step"] for r in log]
            loss = pd.Series([r["loss_per_answer_token"] for r in log]).rolling(5, min_periods=1).mean()
            ax.plot(steps, loss, color=COLORS[i % len(COLORS)], linewidth=2, label=run.name)
    if runs:
        ax.legend(frameon=False, labelcolor=INK)
        _style(ax, "Training loss per answer token (5-step rolling mean)", "step (batch of 32)", "loss / answer token")
        fig.tight_layout()
        fig.savefig(R2 / "loss_curve.png", dpi=160, facecolor=SURFACE)
        L += ["![Loss curves](loss_curve.png)", ""]
    plt.close(fig)
    L += [f"- {n}" for n in notes] or []
    (R2 / "summary.md").write_text("\n".join(L) + "\n")
    print(f"wrote {R2 / 'summary.md'} ({len(runs)} runs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", dest="round_dir", default="round2")
    main(ap.parse_args().round_dir)
