# PhishGuard: a fine-tuned Nemotron that screens scams across email, SMS, URLs and calls

**Fireworks: Deploy with Microsoft & NVIDIA** (HackerSquad, Nov 2 2026). Sponsor track: **NVIDIA Best-Use**, using an NVIDIA Nemotron 3.5 Super model LoRA-fine-tuned on Fireworks Serverless Training.

## Who it is for, and why it matters

Security and fraud teams at banks, telcos and enterprises screen a flood of messages on every channel: emails, SMS, links, and phone calls that are increasingly scripted (and now deepfaked). A frontier model is accurate, but at millions of messages a day it is too slow and too expensive to run on everything, and it means sending customer data to a third party. **PhishGuard** is an open model (NVIDIA Nemotron 3.5 Super, 120B total / 12B active parameters, about 23x smaller than the 2.8T frontier baseline), fine-tuned with LoRA on Fireworks. It labels each item `malicious` or `benign` from the content plus cheap metadata (sender/link/domain facts and timing cues), fast enough to act before the click or before the call ends.

## What we measured (same held-out cases for every system)

| | |
|---|---|
| **Task** | Binary: malicious (phishing, scam, fraud, unsolicited bulk spam) vs. benign, across 4 channels |
| **Our system** | `nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` + LoRA (rank 16, 2 epochs, 2,400 balanced examples), trained with Fireworks **Serverless Training** and evaluated by sampling the adapter in the training session (no deployment) |
| **Frontier baselines** | Kimi K3 (2.8T) and DeepSeek V4.1 Flash (552B) on Fireworks serverless, with the same prompt and metadata; plus Kimi K3 *with reasoning* on a 200-case subset as a ceiling check |
| **Other baselines** | the untuned Nemotron 3.5 Super (same session, same decoding), Nemotron Lightning 30B-A3B on serverless, a retrieval-augmented few-shot harness and router (cheaper tiers), and a zero-cost rules baseline over the metadata |
| **Test sets** | 600 in-distribution (150 per channel, balanced) · 200 emails from an unseen 2008 corpus (temporal drift) · ~200 recent synthetic scams (ScamGuardBench, out-of-distribution) · 300 call prefixes (early warning) · 20 real scam-baiting calls |
| **Metrics** | accuracy, precision, recall, F1, FPR with 95% bootstrap CIs; paired McNemar tests vs. each baseline; input/output tokens; $ per 1M messages; p50/p95 latency and share within a per-channel latency budget |

## Results

_All numbers are measured, from [`results/summary.md`](results/summary.md) (regenerate with `make report`). Same 600 held-out cases for every system; 95% bootstrap CIs; paired McNemar tests._

| System | Size (total / active) | Accuracy % [95% CI] | F1 % | $ per 1M messages |
|---|---|---|---|---|
| **Nemotron 3.5 Super + LoRA (ours)** | 120B / 12B | **97.8** [96.7, 99.0] | **97.8** | **$233** |
| Kimi K3 (frontier baseline) | 2.8T | 96.2 [94.7, 97.5] | 96.1 | $1,600 |
| Kimi K3 + reasoning (200-case subset) | 2.8T | 93.5 [90.0, 96.5] | 93.6 | $2,751 |
| DeepSeek V4.1 Flash | 552B / 16B | 93.5 [91.5, 95.3] | 93.6 | $114 |
| Nemotron 3.5 Super, untuned (same decoding) | 120B / 12B | 90.7 [88.3, 92.8] | 91.2 | $233 |
| Rules on metadata (no model) | — | 74.3 [70.7, 77.5] | 74.6 | $0 |

- **Quality:** the fine-tuned Nemotron beats the frontier baseline on identical cases: **ΔF1 +1.7 pts [+0.2, +3.3], McNemar p = 0.041** (15 cases only ours got right vs. 5 only Kimi K3 got right). Fine-tuning adds **+6.6 F1 pts** over the same model untuned (p = 4e-9). Precision is 99.0% (1.0% false-positive rate vs. 2.7% for Kimi K3).
- **Cost and size:** **6.9x cheaper** per message than Kimi K3, from a model **23x smaller** (120B vs. 2.8T total parameters).
- **Timing, early warning on calls:** after hearing **25%** of a call, ours flags **96%** of scams with **2%** false alarms, and 100% / 0% from the halfway point. Kimi K3 reaches 98% / 14% at 25% and 100% / 20% at halfway: it catches scams as early, but with 7-10x more false alarms.
- **Timing, drift (the honest tradeoff):** on an email corpus from a later year that was never trained on (2008 CEAS), ours falls to 83.8 F1 vs. 91.5 for Kimi K3. On recent synthetic scams (ScamGuardBench), it is 86.0% vs. 94.0% accuracy. The fine-tune learned some corpus-specific patterns, so it should be refreshed with recent data, or paired with a frontier fallback for out-of-distribution traffic.

![F1 with 95% CI](results/charts/f1_ci.png)
![Quality vs cost](results/charts/cost_vs_f1.png)
![Early warning on calls](results/charts/voice_early_warning.png)
![Temporal drift](results/charts/temporal_drift.png)

## How it works

```
HF datasets ──► build_data.py ──► unified records {channel, text, meta, timestamp, label}
                 (dedupe, balanced, domain-grouped URL split, CEAS-08 held out as 2008 test)
                        │
          features.py: links/domains, sender mismatch, domain age, urgency & deadline cues, send time
                        │
           prompt.py: one template, meta vs text-only variants, JSON-schema label output
              ┌─────────┴──────────┐
  Fireworks Serverless Training   Fireworks serverless inference
  LoRA on Nemotron 3.5 Super      Kimi K3 · DeepSeek V4.1 Flash · Nemotron Lightning
              └─────────┬──────────┘
        evaluate.py (async, resumable, hash-stamped) ──► report.py (CIs, McNemar, cost, latency, charts)
```

**Metadata and timing.** Each prompt carries a short Metadata block computed offline: link count, registered domains, shorteners, IP or punycode hosts, suspicious TLDs, brand-in-path tricks, sender vs. link domain mismatch, and (for URLs) domain age plus page features. **Timing** shows up three ways: (1) urgency and deadline cues and send-time/off-hours as features, (2) a temporal-drift test on a corpus from a later year that was never trained on, and (3) latency budgets per channel (URL ≤ 0.5 s, SMS/call ≤ 1 s, email ≤ 2 s), plus an early-warning curve that shows how much of a call must be heard before the scam is flagged.

## Data

| Channel | Dataset (Hugging Face) | License | Notes |
|---|---|---|---|
| Email | `puyang2025/seven-phishing-email-datasets` | inherits sources | 7 corpora, all with both labels; positives are mostly spam. CEAS-08 (2008) is held out entirely as the temporal test; `receiver` and `dataset_name` are never shown to models |
| SMS | `ucirvine/sms_spam` | UCI SMS Spam Collection | spam vs. ham |
| URL / website | `pirocheto/phishing-url` | CC-BY-4.0 | split grouped by registered domain; `page_rank`, `google_index`, `web_traffic`, `statistical_report` excluded as near-label leaks |
| Calls | `BothBosu/scam-dialogue` | Apache-2.0 | synthetic (Llama-3-70B) transcripts, a proxy for real calls; plus 20 real calls from `BothBosu/youtube-scam-conversations` |
| OOD | `flowxai/scamguardbench` (English only) | Apache-2.0 per card | recent synthetic scams with urgency tactics and hard legitimate look-alikes |

Raw data is never committed. `make data` rebuilds the exact splits (seed 42); `data/manifest.json` records row counts, SHA256 hashes and dataset revisions.

## Reproduce

```bash
uv sync                      # Python 3.12
cp .env.example .env         # add FIREWORKS_API_KEY (and FIREWORKS_ACCOUNT_ID for fine-tuning)
make data                    # frozen splits + fine-tuning JSONL (~20 s)
make rules                   # zero-cost baseline
make preflight               # one test call per model
make -j3 baselines           # serverless baselines (~$5)
uv run python scripts/train_serverless.py train --epochs 2      # LoRA SFT on Nemotron 3.5 Super (~$4.5)
uv run python scripts/train_serverless.py eval --ckpt base --model-key nemotron-super-base
uv run python scripts/train_serverless.py eval --ckpt ep2  --model-key ft-nemotron-super
make report                  # results/summary.md, metrics.json, charts
```

## Tradeoffs and limitations

- In-distribution scores are optimistic: train and test share corpora. The 2008 corpus and ScamGuardBench sets are the honest generalization checks and are reported separately.
- Email and SMS positives are mostly bulk spam, not targeted phishing; our label definition says so explicitly and every model sees the same definition.
- Call transcripts are synthetic and templated. The 20 real calls are a small spot check only.
- Latency: the fine-tuned model is sampled inside a Fireworks training session, not a production endpoint, so its latency is indicative only; baselines run on shared serverless endpoints. The serving mode is listed on every row.
- Cost per 1M messages for the fine-tuned model uses Fireworks' published per-token sampling prices for Nemotron 3.5 Super.
- Managed fine-tuning was unavailable to our account during the event (dataset uploads disabled), so we trained through the Serverless Training API with the identical frozen data.
- Out of scope: deepfake audio and video detection (no media fine-tuning on Fireworks in this window, and serverless audio was deprecated), and "fake API" scams (no public dataset).

## Team, disclosures, sponsor track

- **Team:** _[FILL: names]_
- **Built during the event:** all code in this repository. No pre-existing project code or trained models.
- **AI assistance:** Claude Code helped write the data pipeline, evaluation harness and report generator. The team reviewed and ran everything and owns the results.
- **NVIDIA Best-Use track:** the submitted system *is* an NVIDIA Nemotron model (Nemotron 3.5 Super, 120B-A12B), LoRA-fine-tuned on Fireworks Serverless Training. Nemotron Lightning 30B-A3B also powers the cheapest tier of our router.
- **Datasets:** credited above under their licenses.
