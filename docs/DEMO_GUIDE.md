# Demo video + slides: outline and how to make them

Requirements from the rules:
- **Video under 120 s** (aim for 90–110 s); must explain the problem, show the work, and show the measured result.
- **Finalists present live,** so the same slides double as the pitch deck.
- Numbers on screen must come from `results/summary.md`, measured, not estimated. Show tradeoffs honestly.

Owners: Kris builds the slides and records the narration; John drives the terminal demo. Start by 3:05 PM; upload by 4:00 PM.

---

## 1. Video outline (2:00 max)

| Time | Shot | What to say (gist) | Source |
|---|---|---|---|
| 0:00–0:12 | Slide 1: title | "Scammers hit every channel: email, SMS, links, phone calls. Screening every message with a frontier model is too slow and too expensive." | — |
| 0:12–0:25 | Slide 2: user + problem | "Fraud teams at banks and telcos need a verdict before the click or before the call ends, at millions of messages a day." | — |
| 0:25–0:40 | Slide 3: approach | "We fine-tuned NVIDIA Nemotron 30B-A3B with LoRA on Fireworks, using metadata and timing cues, and benchmarked it against Kimi K3 and DeepSeek V4.1 Flash on the same held-out cases." | `docs/DESIGN.md` |
| 0:40–1:10 | **Live terminal demo** | Run the same SMS, email and call prefix through our model and Kimi K3; point at verdict, latency, cost | `make demo` |
| 1:10–1:35 | Slide 4: results | Headline: F1 with 95% CI, $ per 1M messages, p95 latency, McNemar p-value | `results/charts/f1_ci.png`, `cost_vs_f1.png`, `latency.png` |
| 1:35–1:48 | Slide 5: timing | Temporal drift (2008 holdout) and the call early-warning curve: "caught after X% of the call" | `temporal_drift.png`, `voice_early_warning.png` |
| 1:48–2:00 | Slide 6: why adopt + limits | "N× cheaper, M× faster at equal quality; self-hostable open model." One line on limitations (synthetic call data, older email corpora) | `results/summary.md` |

Rule of thumb: about 2.5 spoken words per second, so ~300 words of script in total. Write it in `docs/VIDEO_SCRIPT.md`.

## 2. Slide outline (6 content slides + 1 backup)

1. **Title:** project name, one-line claim, team (John, Kris), logos used: Fireworks, NVIDIA Nemotron.
2. **Problem and user:** multi-channel scams, who screens them, why cost and latency matter (before-the-click, live calls).
3. **Approach:** one diagram, *HF datasets → metadata + timing features → Nemotron LoRA on Fireworks → eval harness*, plus the baselines.
4. **Results:** one table from `results/summary.md` (model, size, F1 ± CI, $/1M msgs, p95 latency) and `f1_ci.png` or `cost_vs_f1.png`.
5. **Timing:** `temporal_drift.png` + `voice_early_warning.png`; the per-channel latency budget hit rate.
6. **Why adopt + tradeoffs:** business value in $; honest limitations; NVIDIA sponsor-track line; repo URL.
7. **Backup (Q&A only):** method details: splits, dedupe and leakage guards, McNemar and bootstrap, reasoning-off vs. reasoning-on ceiling.

Style: one message per slide, large numbers, no more than ~20 words of text per slide, charts straight from `results/charts/`.

## 3. How to produce them

### Slides (about 20 min)
- Use **Google Slides** or **Keynote** (both export PDF).
- Insert PNGs from `results/charts/` after the final `make report` (~2:50). Do not hand-edit numbers; copy from `results/summary.md`.
- Export a PDF to `docs/slides.pdf` and commit it (judges can see it in the repo).

### Terminal demo (about 5 min to rehearse)
```bash
uv run python scripts/demo.py --models ft-meta kimi-k3        # same as `make demo`
uv run python scripts/demo.py --router                        # fallback if the FT deployment is down
uv run python scripts/demo.py --models ft-meta kimi-k3 --channel sms --text "USPS: package on hold, pay fee at ..."
```
- Make the terminal font large (≥ 20 pt), dark theme, window around 1280×720.
- **Note:** `ft-meta` only works while the fine-tuned deployment exists. Record the demo **before** John deletes the deployment, or use `--router` / base models.

### Recording (about 15 min)
- **Mac:** QuickTime Player → File → New Screen Recording (or `Cmd+Shift+5`), with the microphone on. Record slides (presenter mode) and terminal in one take, or in two clips.
- **Cleaner option:** OBS Studio with two scenes (slides, terminal) and one mic.
- Edit and trim in iMovie or QuickTime (Edit → Trim). Keep it **under 2:00**; check the final length before uploading.
- Export 1080p MP4. Upload to the HackerSquad form (or YouTube/Loom unlisted, if the form takes a link); test the link in incognito.

### Checklist before uploading (by 4:00 PM)
- [ ] Video ≤ 2:00, audio clear, numbers match `results/summary.md`
- [ ] Shows problem, demo and measured result vs. the frontier baseline
- [ ] Mentions tradeoffs/limitations and the NVIDIA Nemotron usage
- [ ] `docs/slides.pdf` committed; README links to the video
- [ ] Commit hash for the submission taken **after** the last commit (`git rev-parse HEAD`)
