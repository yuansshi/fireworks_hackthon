# Agent Guide

## Project
Hackathon project (Fireworks: Deploy, Nov 2 2026): fine-tuned NVIDIA Nemotron LoRA vs. frontier
baselines for multi-channel phishing/scam detection (email, SMS, URL, call transcripts).
Design doc: `docs/DESIGN.md`. Results: `results/summary.md`.

## Setup
- Install: `uv sync` (Python 3.12), then `cp .env.example .env` and add `FIREWORKS_API_KEY`.
- Run: `make data`, `make rules`, `make preflight`, `make -j3 baselines`, `scripts/finetune.sh meta`, `make ft-eval`, `make report`.
- Test: `make check` (data leakage/format checks); `make preflight` (one call per model).

## Structure
- `src/phish/`: data loaders, features, prompt, Fireworks client, eval runner, rules baseline, report.
- `scripts/`: build_data, preflight, finetune/deploy/teardown, demo.
- `config/models.yaml`: model ids, sizes, prices, request settings (single source for cost math).
- `data/` (gitignored, rebuilt by `make data`), `results/` (committed: predictions as ids + labels, summary, charts).

## Conventions
- Keep changes small and focused.
- Match existing code style.
- Never commit `.env` or raw data. Never change eval results after the 4:30 PM PT deadline.

## Notes
- The Hugging Face Xet CDN is blocked on the corporate network; `HF_HUB_DISABLE_XET=1` is set in `phish.data`.
- LoRA adapters need an on-demand deployment (no serverless LoRA). Delete deployments right after evals.
