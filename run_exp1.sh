#!/bin/bash
# Experiment 1 - LID + Sentiment on SentiMix Hi-En.
# Usage: bash run_exp1.sh [MODEL]     (default xlm-roberta-base; needs .env, see .env.example)
# Large decoder (Qwen2.5-7B-Instruct) runs need: --dtype bfloat16 --device_map auto --gradient_checkpointing
set -e
export TOKENIZERS_PARALLELISM=false
MODEL=${1:-xlm-roberta-base}
TAG=$(basename "$MODEL")
EXTRA=${EXTRA:-}                      # e.g. EXTRA='--dtype bfloat16 --device_map auto --gradient_checkpointing --eval_batch_size 32'
mkdir -p logs results/exp1/lid
[ -f dataset/SentiMix/test.csv ] || python3 get_sentimix.py

# 1. LID taggers (task 1): fine-tuned encoders vs. lexicon baseline -> pick the best, then FREEZE it
python3 lid_tagger.py --model_name lexicon                                      2>&1 | tee logs/exp1_lid_lexicon.log
python3 lid_tagger.py --model_name xlm-roberta-base                             2>&1 | tee logs/exp1_lid_xlmr_full.log
python3 lid_tagger.py --model_name xlm-roberta-base --unfreeze_last 2           2>&1 | tee logs/exp1_lid_xlmr_last2.log
python3 lid_tagger.py --model_name bert-base-multilingual-cased                 2>&1 | tee logs/exp1_lid_mbert_full.log
python3 lid_eval.py --split dev  --pred-glob "results/exp1/lid/*/dev_pred_tags.csv"
python3 lid_eval.py --split test --pred-glob "results/exp1/lid/*/test_pred_tags.csv"
BEST=$(python3 -c "import pandas as pd; print(pd.read_csv('results/exp1/lid/lid_evaluation_dev.csv').sort_values('macro_f1', ascending=False).iloc[0]['tagger'])")
PRED=results/exp1/lid/$BEST; echo "$PRED" > results/exp1/lid/best_tagger.txt   # best tagger by dev macro-F1, frozen from here on

# 2. sentiment (task 2): baseline vs. LID fusion, gold tags vs. frozen-tagger tags
python3 sentiment_lid.py --model_name "$MODEL" --lid_fusion none $EXTRA                                   2>&1 | tee "logs/exp1_${TAG}_none.log"
for SRC in gold predicted; do
  python3 sentiment_lid.py --model_name "$MODEL" --lid_fusion concat --lid_inject_layer 0 --lid_source $SRC --pred_tags_dir $PRED $EXTRA 2>&1 | tee "logs/exp1_${TAG}_concat0_${SRC}.log"
  python3 sentiment_lid.py --model_name "$MODEL" --lid_fusion add    --lid_inject_layer 0 --lid_source $SRC --pred_tags_dir $PRED $EXTRA 2>&1 | tee "logs/exp1_${TAG}_add0_${SRC}.log"
done
# fusion at the first *trainable* block (cheaper: no backward pass through the frozen blocks)
LAST2=$(python3 -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('$MODEL').num_hidden_layers - 2)")
python3 sentiment_lid.py --model_name "$MODEL" --lid_fusion concat --lid_inject_layer $LAST2 --lid_source gold $EXTRA 2>&1 | tee "logs/exp1_${TAG}_concat${LAST2}_gold.log"

# 3. evaluation: paper-style table + the shared evaluator
python3 eval_sentimix.py --ground results/exp1/ground.csv --pred-glob "results/exp1/*.csv"
python3 evaluate_all.py  --ground results/exp1/ground.csv --pred-glob "results/exp1/*.csv" --out results/exp1/evaluation_results.csv
