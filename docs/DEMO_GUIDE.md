# Demo video + slides: guide

The deck and the narrated video are **generated from measured results** and are ready to upload:

| Asset | Path | Notes |
|---|---|---|
| Video, 1:50 | `docs/video/phishguard_2min.mp4` | 1920×1080 H.264 + AAC, narrated (macOS `say`, voice Samantha); under the 2:00 limit; animated demo |
| Slides | `docs/slides.pdf` and `docs/slides/slide_01.png` … `slide_10.png` | 10 slides, 16:9, labeled by demo section; the same deck works for the live final |
| Narration | `docs/VIDEO_SCRIPT.md` | Per-slide script with timestamps |
| Generator | `scripts/make_media.py` | `uv run python scripts/make_media.py` rebuilds all of the above (needs Chrome, `say`, ffmpeg) |

Every number on screen comes from `results/summary.md`: 600 identical held-out cases, 95% bootstrap CIs, paired McNemar tests.

## 0. Required demo structure (organizers) and where it is covered

| Required part | Slides | What we show |
|---|---|---|
| **1. The Problem**: what we solve and why it matters | 1–2 | Scams hit email, SMS, links and calls; teams must screen everything before the click; frontier screening costs $1,600 per 1M messages |
| **2. Tech Stack**: what we used, what it enabled, what would have been impossible without it | 3–4 | Fireworks Serverless Training (LoRA on a 120B model in ~11 min for ~$2.20: no GPUs, no upload, no deployment), Fireworks serverless inference (frontier baselines through one API), NVIDIA Nemotron 3.5 Super (open, self-hostable), Hugging Face data and the eval harness |
| **3. Live Demo + Code**: the product in action, then the code | 5–7 | Animated terminal run on held-out messages (ours vs Kimi K3, $ per line), then the training loop and the features/scoring code |
| **Key: before / after fine-tuning numbers** | 8 (and 1, 9, 10) | Same model, same prompt: **90.7% → 97.8%** accuracy, errors **56 → 13** (−77%), false positives **15.0% → 1.0%**, call false alarms **34% → 2%**, every channel up; one honest regression (later unseen corpus 87.0% → 85.5%) |

## 1. What the judges score (from the event slides)

*"Evidence comes first."* The deck spends most of its time on the 40% criterion.

| Weight | Criterion | What judges look for | Where we cover it |
|---|---|---|---|
| 40% | Benchmark evaluation | Reliable improvement over the baseline | Slides 3, 4, 6: frozen 600-case test set; 97.8% vs 96.2% for Kimi K3; McNemar p = 0.041; ΔF1 +1.7 [+0.2, +3.3] |
| 25% | Business value | Clear user, valuable problem and benefit | Slides 2, 9: fraud teams screening every message; $233 vs $1,600 per 1M messages; self-hostable open model |
| 20% | Technical execution | A working, well-executed approach | Slides 5, 8: LoRA on Nemotron 3.5 Super via Serverless Training; early-warning and drift analysis |
| 10% | Presentation & demo | Clear explanation, convincing result | Slide 7 demo; one message per slide; under 2:00 |
| 5% | Documentation | Readable code, reproducible evaluation | README "Reproduce"; `make data` / `make report`; this guide |

**"Make the comparison credible" (event slide), checked against our work:**
- [x] **Keep the test fixed:** the same 600 held-out cases (plus 200 temporal, 200 out-of-distribution and 300 call-prefix cases) and the same scoring for every system. Files are hash-checked, and `report.py` refuses mismatched runs.
- [x] **Make runs repeatable:** seeded data build (`data/manifest.json`), prompts in `src/phish/prompt.py`, settings in `config/models.yaml`, every output saved in `results/*.jsonl`.
- [x] **Explain what improved:** both scores and the gain (slide 6), and the cost win reported together with quality.

**"Five steps to a strong submission" (event slide):**
1. Create your evaluation: four channels, malicious/benign, 600 held-out cases. Done.
2. Establish the frontier baseline: Kimi K3 (2.8T), plus DeepSeek V4.1 Flash and a reasoning-on ceiling. Done.
3. Engineer your system: LoRA fine-tune of NVIDIA Nemotron 3.5 Super on Fireworks. Done.
4. Rerun the same evaluation: same cases, same prompt, same scoring. Done.
5. Show the before & after: untuned 90.7%, then fine-tuned 97.8%, against 96.2% for Kimi K3. Done.

## 2. Lessons from the NVIDIA reference talk ("Nemotron LoRA on Fireworks: Training, Evaluation, and Tradeoffs")

The NVIDIA engineer's BANKING77 deck is effectively the template the judges expect: Nemotron Lightning 30B-A3B, 320 held-out questions, base vs. base + 16 examples vs. LoRA, and the punchline *"LoRA: +22 correct vs base at the same prompt size"*. Our deck follows the same arc:

| Their slide | Our slide |
|---|---|
| One narrow task, one untouched test set (big numbers) | Slide 3: 2,400 train · 600 held-out · 4 channels |
| Three ways to solve the same task | Slide 4: Kimi K3, DeepSeek, Nemotron untuned, Nemotron + LoRA |
| The lightweight training run (rank, epoch, time, $) | Slide 5: rank 16 · 1 epoch · ~11 min · ~$2.20 |
| Held-out accuracy and prompt size, one-line punchline | Slide 6: errors out of 600 (zero-based bars), accuracy, $ per 1M msgs; "+43 correct vs untuned", "43% fewer errors than Kimi K3" |
| Inspect gains, regressions and total cost | Slide 8: early-warning gain **and** the drift regression; total project spend ≈ $8 |

## 3. Slide by slide

| # | Section | Slide | One message |
|---|---|---|---|
| 1 | Problem | Title | Fine-tuning takes Nemotron from 90.7% to 97.8%, beating Kimi K3 (96.2%) at ~7× lower cost |
| 2 | Problem | Problem and user | Every channel, before the click; frontier screening is too expensive to run on everything |
| 3 | Tech stack | What the stack made possible | Each tool, what it enabled, and what was impossible without it |
| 4 | Tech stack | The evaluation | 2,400 train / 600 held-out; four systems, same prompt, same scoring |
| 5 | Live demo | Animated terminal run | Held-out messages: our verdicts (recorded) vs live Kimi K3, $ per line; both miss one URL |
| 6 | Code | Training loop | LoRA client, loss masked to the answer tokens, forward/backward, optimizer step, checkpoint |
| 7 | Code | Features and scoring | Metadata + timing cues into the prompt, JSON label out, paired McNemar on the same cases |
| 8 | **Results** | **Before vs after fine-tuning** | 90.7% → 97.8%; errors 56 → 13; FPR 15% → 1%; call false alarms 34% → 2%; per-channel gains; one regression |
| 9 | Results | vs the frontier | 13 vs 23 errors (p = 0.041), ~7× cheaper; tradeoff on newer unseen data |
| 10 | Close | Why adopt | +7.1 points from fine-tuning, ~7× cheaper, 23× smaller, self-hostable |

## 4. Video timeline (from `docs/VIDEO_SCRIPT.md`)

| Time | Section | Slide |
|---|---|---|
| 0:00–0:07 | Problem | 1 Title |
| 0:07–0:21 | Problem | 2 Problem and user |
| 0:21–0:38 | Tech stack | 3 What the stack made possible |
| 0:38–0:46 | Tech stack | 4 The evaluation |
| 0:46–0:56 | Live demo | 5 Animated terminal run |
| 0:56–1:06 | Code | 6 Training loop |
| 1:06–1:15 | Code | 7 Features and scoring |
| 1:15–1:29 | **Results** | **8 Before vs after fine-tuning** |
| 1:29–1:43 | Results | 9 vs the frontier |
| 1:43–1:50 | Close | 10 Why adopt |

Prefer a human voice? Read `docs/VIDEO_SCRIPT.md` over the slides with QuickTime (File → New Screen Recording, microphone on), keep it under 2:00, and export 1080p MP4.

## 5. Live final (if selected)

- Present the same 10 slides from `docs/slides.pdf` in the required order: Problem (1–2), Tech Stack (3–4), Live Demo + Code (5–7), then before/after (8).
- **Live demo option:**
  ```bash
  uv run python scripts/demo.py --models ft-nemotron-super kimi-k3                # ours (recorded) vs Kimi K3 (live)
  uv run python scripts/demo.py --models ft-nemotron-super kimi-k3 --channel sms --text "USPS: package on hold, pay fee at ..."
  uv run python scripts/demo.py --router                                          # cheap Nemotron router, live
  ```
  The fine-tuned adapter was sampled inside its training session (no deployment), so its verdicts are **replayed from the evaluation run**; say so.
- **Q&A backup answers:**
  - *Is the baseline weakened?* Same prompt with metadata for everyone. Kimi K3 with reasoning on was also run (93.5% on a 200-case subset), lower than with reasoning off.
  - *Why Serverless Training?* Managed fine-tuning uploads were disabled for our account. Serverless Training takes the data inline and samples the adapter in-session, with no deployment.
  - *Generalization?* The fine-tune is weaker on a later unseen email corpus (83.8 vs 91.5 F1) and on recent synthetic scams (86.0% vs 94.0%). The fix is periodic refreshes, plus a frontier fallback for out-of-distribution traffic.
  - *Latency?* Not claimed for the fine-tune: it ran in a training session, not on an endpoint. Kimi K3 serverless p50 is 1.19 s.
  - *Leakage guards?* Hash dedupe across splits, URL split grouped by domain, source-identifying fields never shown to models, and CEAS-08 held out entirely.

## 6. Checklist before uploading (video by 4:00 PM)

- [x] Video ≤ 2:00 (1:50), audio clear, numbers match `results/summary.md`
- [x] Follows Problem → Tech Stack → Live Demo + Code, and shows the before/after fine-tuning numbers
- [x] Mentions the tradeoff (drift) and the NVIDIA Nemotron usage
- [x] `docs/slides.pdf` committed; README links to the video and slides
- [ ] Team names in the README Team section
- [ ] Upload `docs/video/phishguard_2min.mp4` to the HackerSquad form (or an unlisted link), and test it in a private window
- [ ] Commit hash taken **after** the last commit (`git rev-parse HEAD`)
