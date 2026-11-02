# One-command reproduction. Serverless targets need FIREWORKS_API_KEY in .env.
PY := uv run python
ALL := test test_temporal test_ood voice_prefix test_realcalls

.PHONY: setup data check rules preflight baselines base-nemotron base-deepseek base-kimi ceiling latency ft-eval report demo

setup:
	uv sync

data:            ## build frozen splits + fine-tuning JSONL + manifest (~20 s)
	$(PY) scripts/build_data.py

check:
	$(PY) scripts/build_data.py --check-only

rules:           ## zero-cost metadata baseline
	$(PY) -m phish.rules

preflight:       ## one test call per serverless model
	$(PY) scripts/preflight.py

baselines: base-nemotron base-deepseek base-kimi   ## run with `make -j3 baselines` to go in parallel

base-nemotron:
	$(PY) -m phish.evaluate --model nemotron-base --variant meta --split $(ALL)
	$(PY) -m phish.evaluate --model nemotron-base --variant text --split test test_temporal

base-deepseek:
	$(PY) -m phish.evaluate --model deepseek-v4p1-flash --variant meta --split $(ALL)
	$(PY) -m phish.evaluate --model deepseek-v4p1-flash --variant text --split test test_temporal

base-kimi:
	$(PY) -m phish.evaluate --model kimi-k3 --variant meta --split $(ALL)
	$(PY) -m phish.evaluate --model kimi-k3 --variant text --split test test_temporal

ceiling:         ## Kimi K3 with reasoning on a 200-case stratified subset (~$3-6)
	$(PY) -m phish.evaluate --model kimi-k3-reasoning --variant meta --split test --per-cell 25 --max-cost 8

latency:         ## concurrency-1 latency on the same 104 cases for the serverless models
	for m in nemotron-base deepseek-v4p1-flash kimi-k3; do $(PY) -m phish.evaluate --model $$m --mode latency; done

ft-eval:         ## after scripts/deploy.sh and FT_META_MODEL in .env
	$(PY) -m phish.evaluate --model ft-meta --variant meta --split $(ALL)
	$(PY) -m phish.evaluate --model ft-meta --mode latency
	$(PY) -m phish.evaluate --model ft-meta --mode throughput

report:          ## results/summary.md, metrics.json, charts
	$(PY) -m phish.report

demo:
	$(PY) scripts/demo.py --models ft-meta kimi-k3
