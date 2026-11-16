#!/bin/bash
# Joint contrastive LID + sentiment - all modes for one model, then the evaluation table.
# Usage: bash run.sh [MODEL]            default xlm-roberta-base
#        EXTRA='--dtype bfloat16 --device_map auto --max_memory 0:7GiB,1:20GiB --gradient_checkpointing --eval_batch_size 32' bash run.sh Qwen/Qwen2.5-7B-Instruct
#        bash run.sh ai4bharat/IndicBERT-v3-1B        (gated repo: needs a Hugging Face token with access)
set -e
export TOKENIZERS_PARALLELISM=false
MODEL=${1:-xlm-roberta-base}; TAG=$(basename "$MODEL"); EXTRA=${EXTRA:-}
mkdir -p logs results
[ -e dataset/SentiMix/test.csv ] || { echo "dataset/SentiMix missing - run the shared get_sentimix.py"; exit 1; }

run() {  # skip a mode whose metrics already exist (resumable)
  local mode=$1; shift
  [ -f "results/${TAG}_${mode}_seed42_metrics.json" ] && { echo "[skip] $TAG $mode"; return; }
  python3 main.py --model_name "$MODEL" --mode "$mode" $EXTRA "$@" 2>&1 | tee "logs/${TAG}_${mode}.log"
}
run sentiment                      # single-task baseline (attention pooling, no LID)
run joint                          # both tasks, cross-attention pooling, no contrastive term
run joint_contrastive              # proposed: + SupCon on LID tokens + SupCon on tweets
run contrastive_lid                # ablation: contrastive on LID only
run contrastive_sent               # ablation: contrastive on sentiment only

python3 evaluate.py --split test --per_class
python3 evaluate.py --split dev
