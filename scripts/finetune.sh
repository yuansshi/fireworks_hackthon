#!/usr/bin/env bash
# Upload the frozen fine-tuning data and launch a LoRA SFT job on Nemotron.
#   scripts/finetune.sh smoke   # 200-row plumbing check, 1 epoch (~$0.25)
#   scripts/finetune.sh meta    # FT-meta, the primary model (~$5)
#   scripts/finetune.sh text    # FT-text, the metadata ablation (~$5)
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a

BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b}"
EPOCHS="${EPOCHS:-2}"
LORA_RANK="${LORA_RANK:-16}"   # always explicit: REST/SDK default 0 means full-parameter at 2x price
STAMP="$(date +%m%d-%H%M)"
which="${1:?usage: finetune.sh smoke|meta|text}"

case "$which" in
  smoke) train=data/ft_smoke_train.jsonl; val="";                       EPOCHS=1 ;;
  meta)  train=data/ft_meta_train.jsonl;  val=data/ft_meta_val.jsonl ;;
  text)  train=data/ft_text_train.jsonl;  val=data/ft_text_val.jsonl ;;
  *) echo "unknown job: $which" >&2; exit 2 ;;
esac
name="phish-${which}-${STAMP}"

# firectl moved to noun-verb commands (`firectl dataset create`); fall back to the old verb-noun form.
if firectl sftj --help >/dev/null 2>&1; then
  ds_create() { firectl dataset create "$@"; }
  job_create() { firectl sftj create "$@"; }
else
  ds_create() { firectl create dataset "$@"; }
  job_create() { firectl create sftj "$@"; }
fi

ds_create "${name}-train" "$train"
args=(--base-model "$BASE_MODEL" --dataset "${name}-train" --output-model "$name"
      --epochs "$EPOCHS" --lora-rank "$LORA_RANK")
if [[ -n "$val" ]]; then
  ds_create "${name}-val" "$val"
  args+=(--evaluation-dataset "${name}-val")
fi
job_create "${args[@]}"
echo
echo "Output model: accounts/${FIREWORKS_ACCOUNT_ID:-<acct>}/models/${name}"
echo "Monitor:      firectl sftj list   |   firectl sftj get <JOB_ID>"
