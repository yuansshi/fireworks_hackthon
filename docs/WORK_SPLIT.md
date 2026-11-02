# Work split: John and Kris

Goal: both of us work in parallel until the one merge point at about 2:50 PM. Submissions open at 3:30 PM and close at 4:30 PM PT.

## John: Fireworks pipeline (critical path)

Owns everything that runs on Fireworks: jobs, deployments, spend. Works on the laptop that has `.env` and the `firectl` login.

| When | Task | Done when |
|---|---|---|
| Now | Confirm the FT-meta job is running (`firectl sftj list`). If it isn't, launch it: `uv run python scripts/finetune_api.py`. Launch FT-text only if spend allows | Job shows RUNNING/COMPLETED |
| Now → 2:15 | Finish the baselines: Nemotron base `meta` on test_temporal / test_ood / voice_prefix / test_realcalls, `text` on test + test_temporal; Kimi K3 reasoning-on ceiling (200 cases) | Files appear in `results/` |
| ~2:15 | Check spend on the billing page. Over $35 → drop FT-text and the ceiling run | Spend noted in `#status` below |
| FT ~⅔ done | Create the deployment for the fine-tuned model | Deployment READY |
| FT done | Smoke test 5 cases → eval FT on every test set → latency runs → **delete the deployment** | `results/ft-*` files exist; `firectl deployment list` is empty |
| ~2:50 | Commit and push `results/`, then tell Kris "FT results are in" | Pushed |
| 3:30–4:00 | Upload the demo video (**by 4:00**) | Video processed |

## Kris: evidence and story

Owns everything judges read. Needs only the repo; optionally run `uv sync` and `uv run python scripts/build_data.py` to inspect the data.

| When | Task | Done when |
|---|---|---|
| Now | Read `results/summary.md`. Error analysis: about 10 false positives/negatives per model (Kimi K3, DeepSeek, rules). Write them to `docs/ERRORS.md` | File pushed |
| Now → 2:45 | Write the README sections that don't need FT numbers: problem and user, business value ($ per 1M messages), datasets and licenses, method and statistics, limitations, disclosures (AI assistance: Claude Code), NVIDIA sponsor-track paragraph | README pushed |
| Now → 2:45 | Settle the `[FILL]` items in `docs/DESIGN.md`: latency budgets, persona, "fake API" note | Doc updated |
| Now → 2:45 | Demo prep: pick 3 striking examples (SMS, email, call prefix); write a 60–90 s video script in `docs/VIDEO_SCRIPT.md` | Script pushed |
| Now → 2:45 | HackerSquad: team roster (John + Kris), project draft; make the GitHub repo **public** and test it in an incognito window | Draft saved |
| ~2:50 | After John's push: `git pull` → `make report` → paste the final table and charts into the README | README has FT vs. frontier with CIs |
| 3:05–3:30 | Record the video with John (Kris presents, John drives the demo, or swap) | Video file ready |
| 3:30–4:00 | Submit the form plus the commit hash (`git rev-parse HEAD`) | Shows "submitted" |

## Dependencies

- **FT job ← frozen training data.** Ready; not blocked.
- **Kris's error analysis and README ← existing baseline results.** Ready now.
- **Final table, the FT-vs-frontier claim and the video ← FT results from John.** This is the only merge point, at about 2:50.
- **Fallback:** if FT isn't done by about 2:30, Kris finishes the README around the baseline table. FT goes in only if it lands by about 3:15.

## Git rules (to avoid conflicts)

- `git pull --rebase` before every push; commit small and often.
- **John** commits `results/`, `config/` and `scripts/`. **Kris** commits `README.md` and `docs/`.
- Never commit `.env` (it's git-ignored). Never change eval results after 4:30.

## Status

- John:
- Kris:
