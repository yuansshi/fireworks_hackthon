# Demo video + slides: guide

The deck and the narrated video are **generated from measured results** and are ready to upload:

| Asset | Path | Notes |
|---|---|---|
| Video, 1:46 | `docs/video/phishguard_2min.mp4` | 1920×1080 H.264 + AAC, narrated (macOS `say`, voice Samantha); under the 2:00 limit |
| Slides | `docs/slides.pdf` and `docs/slides/slide_01.png` … `slide_09.png` | 9 slides, 16:9; the same deck works for the live final |
| Narration | `docs/VIDEO_SCRIPT.md` | Per-slide script with timestamps |
| Generator | `scripts/make_media.py` | `uv run python scripts/make_media.py` rebuilds all of the above (needs Chrome, `say`, ffmpeg) |

Every number on screen comes from `results/summary.md`: 600 identical held-out cases, 95% bootstrap CIs, paired McNemar tests.

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

| # | Slide | One message | Rubric |
|---|---|---|---|
| 1 | Title | A fine-tuned NVIDIA Nemotron beats a frontier model at scam screening: 97.8% vs 96.2%, ~7× cheaper | All |
| 2 | Problem and user | Fraud teams must screen every channel before the click; frontier screening costs $1,600 per 1M messages | Business value |
| 3 | One evaluation, frozen | 2,400 train / 600 held-out / 4 channels, plus drift and early-warning sets | Benchmark |
| 4 | Four systems, same cases | Same prompt and scoring for every system; ours is the only change | Benchmark |
| 5 | Lightweight training run | Rank-16 LoRA, 1 epoch, ~11 min, ~$2.20, no upload or deployment | Technical |
| 6 | Accuracy and cost | 13 vs 23 errors (Kimi K3), p = 0.041, +43 correct vs untuned | Benchmark |
| 7 | Demo | Held-out items: our recorded verdicts vs live Kimi K3, with $ per line; both miss one URL | Presentation |
| 8 | Timing and tradeoff | 96% of scam calls caught after 25% of the call with 2% false alarms (Kimi K3: 14%); frontier wins on drift | Technical, honesty |
| 9 | Why adopt | ~7× cheaper, 23× smaller, self-hostable; repo for reproduction | Business value |

## 4. Video timeline (from `docs/VIDEO_SCRIPT.md`)

| Time | Slide |
|---|---|
| 0:00–0:08 | 1 Title |
| 0:08–0:23 | 2 Problem and user |
| 0:23–0:33 | 3 One evaluation |
| 0:33–0:42 | 4 Four systems |
| 0:42–0:50 | 5 Training run |
| 0:50–1:05 | 6 Accuracy and cost |
| 1:05–1:13 | 7 Demo |
| 1:13–1:33 | 8 Timing and tradeoff |
| 1:33–1:46 | 9 Why adopt |

Prefer a human voice? Read `docs/VIDEO_SCRIPT.md` over the slides with QuickTime (File → New Screen Recording, microphone on), keep it under 2:00, and export 1080p MP4.

## 5. Live final (if selected)

- Present the same 9 slides from `docs/slides.pdf`, about 15 s each.
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

- [x] Video ≤ 2:00 (1:46), audio clear, numbers match `results/summary.md`
- [x] Shows problem, approach, demo and measured result vs. the frontier baseline
- [x] Mentions the tradeoff (drift) and the NVIDIA Nemotron usage
- [x] `docs/slides.pdf` committed; README links to the video and slides
- [ ] Team names in the README Team section
- [ ] Upload `docs/video/phishguard_2min.mp4` to the HackerSquad form (or an unlisted link), and test it in a private window
- [ ] Commit hash taken **after** the last commit (`git rev-parse HEAD`)
