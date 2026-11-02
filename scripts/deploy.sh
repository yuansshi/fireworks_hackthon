#!/usr/bin/env bash
# Serve fine-tuned adapters for evaluation. GPU billing runs while replicas are up:
# run scripts/teardown.sh as soon as the FT evals finish.
#   scripts/deploy.sh merged <output-model-name>   # live-merge: one adapter on its own deployment
#   scripts/deploy.sh multi                        # base deployment with addons (multi-LoRA); then load adapters
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a

ACCT="${FIREWORKS_ACCOUNT_ID:?set FIREWORKS_ACCOUNT_ID in .env}"
SHAPE="${DEPLOYMENT_SHAPE:?set DEPLOYMENT_SHAPE to a shape that fits the 30B base (see Fireworks deployment docs)}"
BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b}"
mode="${1:?usage: deploy.sh merged <model-name> | multi}"

case "$mode" in
  merged)
    firectl deployment create "accounts/${ACCT}/models/${2:?model name}" \
      --deployment-shape "$SHAPE" --scale-to-zero-window 5m --wait ;;
  multi)
    firectl deployment create "$BASE_MODEL" --deployment-shape "$SHAPE" --enable-addons \
      --scale-to-zero-window 5m --wait
    echo "Now load each adapter onto this deployment (see docs.fireworks.ai/fine-tuning/deploying-loras)." ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
firectl deployment list
echo
echo "Put the queryable model strings in .env, e.g."
echo "  FT_META_MODEL=accounts/${ACCT}/models/<ft-model>#accounts/${ACCT}/deployments/<deployment-id>"
