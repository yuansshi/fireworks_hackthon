# Phishing Detection: Fine-tuned Nemotron vs. Frontier Baselines (Fireworks Deploy)

> **Historical plan (written ~1 PM, Nov 2).** What was actually built differs: Nemotron 3.5 Super (120B-A12B) fine-tuned with LoRA on Fireworks Serverless Training, evaluated at the **epoch-1** checkpoint, with no deployment. See the README for the executed system and results.


> Design doc + execution plan for **Fireworks: Deploy with Microsoft & NVIDIA** (HackerSquad, Nov 2 2026, TERRA SF). Sections marked **[FILL]** are for the team to complete as we go.

## Context

**What the challenge asks for.** "Build an open-model system for a meaningful task and demonstrate a measurable improvement against a frontier-model baseline."
- The improvement can be in quality, cost or latency. Any cost or latency win has to keep task quality useful.
- Prizes: $10k / $5k / $3k, plus an NVIDIA Best-Use award (DGX Spark). The project must use Fireworks to be eligible.

**How it is scored.**
| Criterion | Weight | What judges look for |
|---|---|---|
| Benchmark | 40% | Same held-out cases for both systems, 100+ varied examples, uncertainty estimates or statistical tests |
| Business value | 25% | A clear user and a compelling reason to adopt |
| Technical execution | 20% | Engineering quality and meaningful experimentation |
| Presentation | 10% | A convincing demo |
| Documentation | 5% | A reproducible README |

**Hard deadlines (PT).**
| Time | Deadline |
|---|---|
| ~3:00 PM | Team's v1 results (the "~2 h" target) |
| 3:30 PM | Start submitting |
| 4:00 PM | Video uploaded |
| 4:30 PM | Submissions close and development stops |

**What we must submit.**
- A public GitHub repo with setup and eval instructions.
- The Git commit hash of the version to be judged.
- An overview: task, user, approach, business value.
- Evaluation evidence: baseline, dataset source and size, metric, method, before/after results, limitations.
- A demo video under 120 s (aim for under 60 s).
- The team list.
- Sponsor-track info.
- Disclosures of AI-assistant use and of any pre-existing work.

**Our constraints.** $50 Fireworks credit, Fireworks models only, data from Hugging Face, 2 people.

**Decisions made with you.**
- **Channels:** email, SMS, URL/website, voice-call transcripts. Deepfake media is out of v1; a call enters only as a transcript + metadata.
- **"Timing" means three things:** detection latency vs. a per-channel budget, temporal drift, and urgency/time metadata.
- **Story:** a small fine-tuned model beats frontier models.

**Fireworks facts that shape the plan** (current docs, 2026-11-02):
- LoRA adapters can't be served on serverless. They need an on-demand deployment (H100/H200 about $8/GPU-hr, billed per second, spin-up takes minutes).
- LoRA SFT costs $0.50 per 1M training tokens up to 16B parameters, and $3 per 1M for 16–80B.
- **Without a payment method:** 0 training GPUs, and a 10 requests/min cap across the whole account.

**Current state.** Project folder is empty (only `agent.md`), not a git repo, no `firectl`, no API key in the shell. `uv` is installed; we'll use Python 3.12.

## How we win (rubric → design)

- **Benchmark (40%).**
  - Fine-tuned **NVIDIA Nemotron 3.5 Lightning 30B-A3B (LoRA)** vs. 2 frontier models on serverless, given a fair, strong prompt (no sandbagging).
  - Everything runs on the same frozen ~1,000 held-out cases: 600 in-distribution, 200 temporal, ~200 recent OOD.
  - Report 95% bootstrap CIs and paired McNemar tests.
  - The headline claim is "equal or better F1 at much lower cost and latency", backed by evidence.
- **Business value (25%).**
  - User: the security or fraud team at a bank, telco or enterprise that screens email, SMS, URLs and call transcripts.
  - Pitch: cheap enough to run on every message, fast enough to act before the click or before the call ends, and an open model they can self-host for privacy.
  - Show $ per 1M messages vs. the frontier models.
- **Technical execution (20%).** Metadata ablation, temporal-drift holdout, an early-warning curve on calls, a multi-channel LoRA, and a reproducible harness.
- **NVIDIA award.**
  - Nemotron is the fine-tuned model; NVIDIA's talk was literally "Nemotron LoRA on Fireworks".
  - Nemotron's 3B active parameters drive the cost and latency win.
- **Presentation and documentation (15%).** A 60–90 s demo of the same messages side by side, plus a README with one-command reproduction.

## Experiment matrix

Serverless model ids all start with `accounts/fireworks/models/`. Prices are $ per 1M tokens, input/output.

| Role | Model | Size (total/active) | Served via | Variants |
|---|---|---|---|---|
| **Ours** | FT-meta: LoRA r16 on `nemotron-lightning-3p5-30b-a3b`, trained on the meta prompt | 30B / 3B | on-demand deployment (1 GPU) | meta |
| Ablation (optional) | FT-text: same base, trained text-only | 30B / 3B | same deployment if multi-LoRA works | text-only |
| Before (own base) | `nemotron-lightning-3p5-30b-a3b`, $0.05/$0.20 | 30B / 3B | serverless; also bare on our deployment for a same-hardware latency comparison | meta + text-only |
| **Frontier baseline 1** | `kimi-k3`, $3/$15, `reasoning_effort:"none"` | 2.81T MoE | serverless | meta + text-only |
| Frontier ceiling check | `kimi-k3` with default reasoning, on a 200-case stratified subset | 2.81T | serverless | meta |
| Frontier baseline 2 (cost-efficient) | `deepseek-v4p1-flash`, $0.30/$1.20, reasoning `"none"` | 552B / 8–16B | serverless | meta + text-only |
| Zero-cost baselines | majority class; keyword + metadata rules | — | local | — |
| Stretch | `glm-5p3` or `qwen3p8-max` as a third frontier; `rune` decision model (input-only billing) | — | serverless | meta |

Notes:
- **Why reasoning off:** it is the realistic setting for screening every message, and it keeps cost sane (Kimi K3 with thinking on would cost about $20+ for one run). To avoid any charge of a weakened baseline, the reasoning-on ceiling run is reported next to it.
- Nemotron and gpt-oss leave serverless on Nov 7, which doesn't affect today.

**Test sets.** Every model gets the same files, frozen and hashed before any run:
1. `test`: in-distribution, balanced, 150 per channel (600).
2. `test_temporal`: 200 emails from 2008 (CEAS-08). Training uses only pre-2008 emails.
3. `test_ood`: about 200 recent synthetic English items from ScamGuardBench, with hard negatives and urgency tactics.
4. `voice_prefix`: 100 test call transcripts cut at 25/50/75% of turns (300). The 100% point is already in `test`.

## Data (columns and splits checked through the HF datasets-server on 2026-11-02)

**Per channel:** 600 train + 50 val + 150 test, balanced 50/50. Totals: 2,400 train, 200 val, 600 test.

| Channel | Dataset | What we use | Metadata / timing | Caveats |
|---|---|---|---|---|
| Email | `puyang2025/seven-phishing-email-datasets` (162k rows) | **In-distribution pool:** rows dated before 2008 plus Enron/Ling rows with null dates. **`test_temporal`:** 2008 rows (CEAS-08), balanced 200. Drop fake dates (2038/2099/2100) | sender, subject, date (hour, weekday, off-hours), `urls` flag; URL features derived from the body | Positives are mostly spam, not pure phishing, so the label definition below covers spam. The year split is also a corpus split. **Mask `receiver` and never feed `dataset_name`** (both leak the source) |
| SMS | `ucirvine/sms_spam` (5,574; 747 spam) | 300 spam / 300 ham for train; the rest go to val/test | derived: link, shortener, money/OTP/prize cues | Spam, not strictly smishing; license listed as unknown on HF, so cite UCI |
| URL / website | `pirocheto/phishing-url` (7,658 / 3,772; CC-BY-4.0) | Its train/test splits, **regrouped so the two share no registered domain** | domain_age, registration length, dns_record, whois flag, plus page-content features (login form, iframe, external favicon, …), which cover "website" without downloading HTML | **Exclude `page_rank`, `google_index`, `web_traffic`:** they nearly give away the label and aren't available at click time for brand-new domains |
| Voice | `BothBosu/scam-dialogue` (1,280 / 320; Apache-2.0; 8 scam types) | train → train/val; test → `test` + `voice_prefix` | derived: OTP / gift card / remote access / authority cues | Generated by Llama-3-70B and templated, so the report labels it a synthetic proxy. Real-world spot check: `BothBosu/youtube-scam-conversations` (20 real scam calls, recall only) |
| Recent OOD | `flowxai/scamguardbench` (English rows only, ~200) → **`test_ood`** | `scam_likely` → malicious, `no_indicators` → benign, `suspicious` dropped | `tactics` includes `urgency_pressure`, so it gives an urgency-cue breakdown; includes hard `legit_confusable` negatives | Synthetic; the card says Apache-2.0 but the viewer shows no license, so cite and verify |

- **Rule:** never train on one of ealvaradob, zefang-liu or cybersectony and test on another. They share email sources, so the test would not be held out.
- **Hygiene:**
  - Normalize text, then hash-dedupe within and across splits.
  - Balance labels within each source.
  - Record each dataset's id + revision in `data/manifest.json`.
  - Keep raw data out of git (licenses, PII). Judges rebuild it with `make data`.
- **Stretch:** ealvaradob `webs` (465 MB of raw HTML) for a true HTML channel; `kirco567/ScamWeb-Dataset` (2025–26, but 29.7 GB, so only url + title via DuckDB column projection).
- **Out of scope, with reasons in the README:**
  - Deepfake audio/video: HF sets are media-only, and Deepfake-Eval-2024's terms forbid training.
  - Fireworks serverless audio was deprecated in June 2026.
  - No "fake API" dataset exists.

## Metadata and timing features

All features are computed locally from fields the dataset already has, so they are free and fast. No live WHOIS or URL fetching: a lookup today on a 2008 domain would give wrong answers and leak information.

| Channel | Metadata (when the dataset has it, or derived) |
|---|---|
| All | URL count, registered domains (`tldextract` with its bundled suffix list, offline), shortener flag, IP-literal host, punycode, suspicious TLD, brand keyword in the path or subdomain but not the domain |
| Email | sender domain, display-name ≠ address domain (parsed from `sender`), link domain ≠ sender domain, subject, attachment mention |
| SMS | has link, shortener, money/OTP/prize/gift-card mentions |
| URL / website | length, subdomain depth, `@` / `//` tricks, HTTPS, plus curated pirocheto features (domain_age, registration length, dns_record, login form, iframe, external favicon, …). Raw HTML only in the stretch goal |
| Voice transcript | requests for OTP / gift card / remote access / payment, authority impersonation (bank, IRS, tech support) |
| **Timing** | urgency lexicon hits ("within 24 hours", "suspended", "final notice"), explicit deadlines (regex), send hour/weekday and off-hours flag when a timestamp exists, domain age |

Leakage guard: never feed a column that identifies the source dataset, and drop any feature that exists for only one class.

## Prompt and label format

- One template for every model and channel, in two variants: **meta** (Channel + Metadata block + Content) and **text-only** (Channel + Content). Content is truncated to ~512 tokens.
- The system prompt gives a short, explicit definition of phishing/scam and asks for JSON. Every model gets the same strong prompt, so the baseline isn't sandbagged.
- Output: `{"label":"malicious"|"benign"}` through `response_format` `json_schema`. That is about 6–8 output tokens, and fine-tuned outputs match base outputs exactly.
  - One written definition is shared by the prompt, the README and every model: **malicious = phishing, scam, fraud, or unsolicited bulk spam that tries to get the recipient to click, pay, reply, call back, install, or disclose information.**
  - The definition matches the dataset labels, so frontier models aren't penalized for a mismatch.
- Decoding: temperature 0, `max_tokens` 16. Reasoning `"none"` on Kimi/DeepSeek, and Nemotron's thinking turned off.
- Preflight checks that `usage.completion_tokens_details.reasoning_tokens` = 0.
- Fine-tuning rows: `{"messages":[system, user, assistant=<label JSON>]}` with the identical template, and no thinking traces.

## Fine-tuning and serving (Track B)

```bash
brew tap fw-ai/firectl && brew install firectl && firectl signin
firectl quota update monthly-spend-usd --value 50          # spend cap (verify flag)
firectl dataset create phish-meta-train data/ft_meta_train.jsonl
firectl dataset create phish-meta-val   data/ft_meta_val.jsonl
firectl sftj create --base-model accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b \
  --dataset phish-meta-train --evaluation-dataset phish-meta-val \
  --output-model phish-nemotron-meta --epochs 2 --lora-rank 16
firectl sftj get <JOB_ID>
firectl deployment create accounts/<ACCT>/models/phish-nemotron-meta \
  --deployment-shape <shape> --scale-to-zero-window 5m --wait   # or a multi-LoRA base deployment (--enable-addons, BF16 shape)
firectl deployment delete <DEP_ID> --wait                       # right after evals
```

- **Training data:** about 2,400 rows (600 per channel, balanced), plus 200 for validation, 2 epochs.
- **Always pass `--lora-rank`.** Through REST or the SDK it defaults to 0, which means full-parameter training at 2× the price.
- **Smoke job at about 1:30 PM:** 200 rows, 1 epoch. It proves the training quota, the JSONL format and the queue time before the real launch.
- **Prefer multi-LoRA.** One deployment then hosts FT-meta, FT-text and the bare base. If Nemotron has no BF16 addon shape, use live-merge for FT-meta only and drop the FT-text ablation.
- Create the deployment when FT-meta is about ⅔ done, so spin-up overlaps training.

## Eval harness and metrics

- **Client.** Async OpenAI SDK pointed at `https://api.fireworks.ai/inference/v1`.
  - Fireworks-only parameters go through `extra_body`: `reasoning_effort`, `perf_metrics_in_response: true`.
  - `.with_raw_response` logs `fireworks-server-time-to-first-token` and `fireworks-server-processing-time`.
  - Concurrency 8–16, honors `Retry-After` on 429, resumable JSONL output.
- **Same cases everywhere.** Each results file stores the SHA256 of its test file, and `report.py` refuses to compare runs whose hashes don't match.
- **Metrics per model × channel × split.**
  - Size (total/active params).
  - Accuracy, precision, recall, F1, FPR, and parse-failure rate.
  - Mean input/output tokens, reasoning tokens included.
  - $ per 1k and per 1M messages.
  - p50/p95 latency, server TTFT, throughput, and % within the latency budget.
  - Temporal Δ, metadata Δ, and the voice early-warning curve (recall and FPR after 25/50/75/100% of the call).
- **Statistics.** 95% bootstrap CIs (1,000 resamples) on accuracy and F1; paired McNemar exact test of FT vs. each frontier model; paired-bootstrap CI on ΔF1.
- **Latency budgets [FILL/confirm]:** URL pre-click ≤ 0.5 s, SMS ≤ 1 s, voice chunk ≤ 1 s, email ≤ 2 s.
  - Latency is measured on a fixed 100-case subset at concurrency 1 after warm-up; throughput at concurrency 16.
  - Every row says whether it ran on shared serverless or a dedicated deployment. Base vs. FT is also measured on the same deployment.

## Budget ($50 credit; plan ≈ $33, keep ≈ $17 in reserve, hard stop at $45 spent)

| Item | Basis | Est. |
|---|---|---|
| Smoke FT | 200 rows × 1 epoch, about 80k tokens × $3/1M | $0.25 |
| FT-meta (Nemotron LoRA) | 2,400 rows × ~380 tokens × 2 epochs ≈ 1.8M tokens × $3/1M | $5.5 |
| FT-text (optional ablation) | same | $5.5 |
| Deployment for FT eval | 1 GPU (H100/H200 at $8/h) × ~1.25 h including spin-up | $10 |
| Kimi K3 evals (reasoning off) | ~2,100 requests × ~500 tokens × $3/1M, plus tiny output | $3.5 |
| Kimi K3 reasoning-on ceiling | 200 cases, ~1–1.5k thinking tokens × $15/1M | $3–5 |
| DeepSeek V4.1 Flash + Nemotron base evals | ~2,100 requests each | $0.5 |
| Dev calls and reruns | | $3 |

Guards:
- Prices live in `config/models.yaml`. The harness prints a projected cost before each run and aborts if it's over the per-run cap.
- Reconcile against the billing page at 2:15 and 2:50.
- Over $35 at the 2:15 checkpoint → drop FT-text and the ceiling run.

## Timeline (PT, assuming the plan is approved by ~1:05 PM)

**Who does what:**
- **Claude (this session)** writes the code for both tracks in critical-path order: scaffold → `build_data` → `client`/`evaluate` → fine-tune scripts → `report` → `demo`.
- **A and B** run the code, watch the jobs, make the go/no-go calls, record the video and submit.
- **You handle anything with credentials yourselves:** `firectl signin`, pasting `FIREWORKS_API_KEY` into `.env`, redeeming the promo.
- **Anything outward-facing** (creating the public GitHub repo, pushing) happens only after your OK.

| Time | Track A: data and report | Track B: Fireworks and eval |
|---|---|---|
| 1:05–1:20 | `git init`, uv project (Py 3.12), `.gitignore`/`.env.example`; start the HF downloads | **Preflight:** redeem $50 promo; confirm card on file (6,000 RPM, training quota > 0); install firectl and sign in; set spend cap; one test call per model id to check that reasoning is off, `json_schema` works and the headers come back |
| 1:20–1:45 | `build_data.py`: unified schema, features, dedupe, splits, files. Hand 200 SMS/email rows to B by 1:30 | `client.py` + `evaluate.py`; **1:30 smoke FT job**; dry-run on 20 rows across all models |
| **1:45** | **Freeze data**: manifest with counts and SHA256s | Upload FT data; **launch FT-meta (+ FT-text)** |
| 1:45–2:25 | `report.py` (metrics, CIs, McNemar, charts); rules baseline; README skeleton; business-value math ($ per 1M messages) | Baseline runs: Nemotron base, DeepSeek V4.1 Flash, Kimi K3 (meta on every test set, text-only on test + temporal); Kimi K3 reasoning-on ceiling |
| **2:15** | **Checkpoint:** baseline table v0, the fallback submission; check spend | Check FT status; create the deployment when FT-meta is ~⅔ done |
| 2:25–2:50 | Error analysis (10 FP/FN), draft findings | Smoke-test FT on the deployment; eval FT-meta (+ FT-text) on all sets; latency runs; **delete the deployment** |
| **2:50–3:05** | **v1 results:** `results/summary.md` + charts | Confirm teardown; reconcile cost |
| 3:05–3:30 | Final README | `demo.py`; record a 60–90 s video |
| 3:30–4:00 | Make the repo public, check it in an incognito window, submit the form + commit hash | Upload the video (**by 4:00**) |
| 4:00–4:30 | Buffer. No new results after submitting; development stops at 4:30 | |

**Go / no-go:**
- **1:35, smoke FT can't start (quota or card):** escalate to Fireworks staff on site right away.
  - First fallback: Muse Glimmer 30B LoRA SFT, which is free and needs no card.
  - Second fallback: the no-fine-tune story the rules allow — metadata-augmented Nemotron base vs. Kimi K3, claiming cost and latency with useful quality.
- **2:30, FT-meta not finished:** keep the baseline table as v1. Evaluate FT if it lands by 3:15; the eval takes about 10 minutes.

## Submission package

- **README.md:**
  - Problem and user.
  - Approach, with a diagram.
  - Models and configs.
  - Datasets: source, size, license.
  - Method: splits, metrics, statistics.
  - Results table with 95% CIs and McNemar p-values; cost and latency.
  - Tradeoffs and limitations: synthetic voice data, dataset age, in-distribution vs. temporal results, serverless vs. dedicated latency, reasoning-off baseline next to the ceiling.
  - Reproduce: `uv sync` → `.env` → `make data eval report`.
  - Team.
  - **Sponsor track:** NVIDIA Nemotron LoRA on Fireworks.
  - **Disclosures:** AI assistance (Claude Code), no pre-existing code, datasets credited.
- **`results/`:** `summary.md`, `metrics.json`, PNG charts, and per-sample predictions stored as ids + labels only (no raw text, so no PII or license problems).
- **Video, 60–90 s:**
  - 10 s problem.
  - 25 s live demo: the same SMS, email and call prefix through FT-Nemotron and Kimi K3, showing verdict, latency and $.
  - 20 s results chart.
  - 10 s business value.
- **Form:** commit hash, team list, sponsor track.

## Files to create

```
pyproject.toml, Makefile     uv, Python 3.12: openai, datasets, pandas, scikit-learn, scipy, tldextract, tenacity, tqdm, matplotlib, pyyaml, python-dotenv
.env.example, .gitignore     FIREWORKS_API_KEY (+ HF_TOKEN optional); ignore .env, data/, raw results text
config/models.yaml           model ids, total/active params, $/1M in/out, serving mode, reasoning setting
src/phish/data.py            HF loaders -> {id, channel, text, meta, timestamp, label, source, split}; dedupe; balanced sampling; temporal holdout
src/phish/features.py        metadata + timing features
src/phish/prompt.py          template (meta / text-only), JSON schema, tolerant parser
src/phish/client.py          Fireworks client: timing, headers, usage, cost, retries
src/phish/evaluate.py        async runner -> results/<model>__<variant>__<split>.jsonl (resumable, test-hash stamped)
src/phish/rules.py           zero-cost keyword/metadata baseline
src/phish/report.py          metrics, CIs, McNemar, summary.md, charts
scripts/build_data.py        data/{train,val,test,test_temporal,test_ood,voice_prefix}.jsonl + ft_{meta,text}_{train,val}.jsonl + manifest
scripts/finetune.sh, deploy.sh, teardown.sh
scripts/demo.py              side-by-side live demo for the video
README.md, docs/DESIGN.md    this doc, kept current; also fill in agent.md Setup/Run/Test
```

## Risks and fallbacks

| Risk | Mitigation / fallback |
|---|---|
| No card → 10 RPM and no training GPUs | First item in preflight; staff on site can help; Muse Glimmer free SFT; fall back to the no-fine-tune story |
| Fine-tune lands late | Smoke job at 1:30, real launch by 1:45; the 2:15 baseline table is the fallback |
| Nemotron has no multi-LoRA shape | Live-merge FT-meta only; drop the FT-text ablation |
| Adaptive TPM throttling on big models | Concurrency 8–16, honor `Retry-After`, resumable runs, start the frontier runs first |
| Thinking-token blow-up | Reasoning `"none"`, assert `reasoning_tokens` = 0, cap `max_tokens` |
| Dataset artifacts / leakage / old data | Dedupe within and across splits, balance labels within each source, temporal holdout, per-source accuracy in the appendix |
| Synthetic voice transcripts | Labeled in the report as a synthetic proxy |
| Repo can't be accessed at judging | Make it public by 3:45 and check it signed out; no keys in the history (`.env` ignored from the very first commit) |

## Verification

1. **Data:** `build_data.py --check` prints counts per channel × label × split, and checks:
   - zero normalized-text hash overlap across train/val/test/temporal/ood;
   - no URL domain shared between URL train and test;
   - `receiver`, `dataset_name`, `page_rank`, `google_index` and `web_traffic` never appear in any prompt;
   - the fine-tune JSONL schema is valid;
   - token-length stats.
2. **Harness smoke test:** 20 cases × every model → 100% parse rate, with tokens, server TTFT, latency and cost recorded, and reasoning tokens = 0 where intended.
3. **Sanity:** majority and rules baselines computed; every LLM beats majority; Kimi K3 rerun on 100 cases agrees ≥ 98% at temperature 0.
4. **Fine-tune:** job completed, validation loss going down, a 5-case smoke test on the deployment returns valid JSON.
5. **Stats:** CIs and McNemar computed only on hash-matched runs; `summary.md` regenerates from `results/` in one command.
6. **Cost:** harness total within ~10% of the billing page; `firectl deployment list` shows nothing running at the end.
7. **Submission:** repo link opens signed out; video plays; the commit hash matches `git rev-parse HEAD`.

## Open items **[FILL]**

- Latency budget per channel; confirm or adjust the defaults.
- **"Fake API" scope:** malicious API endpoints, a typosquatted SDK/package, or OAuth-consent phishing? No HF dataset exists, so v1 folds it into URL/domain features and the README lists it as future work.
- **Business-value persona:** bank, telco, or enterprise SOC? It sets the $ per 1M messages framing.
- Who records the video, and who owns the HackerSquad form and team roster.
