#!/bin/bash
# Experiment 2 (Switch Point + Sentiment) - full pipeline for one sentiment model.
# Usage: bash run.sh [MODEL]            default xlm-roberta-base
#        EXTRA='--dtype bfloat16 --device_map auto --max_memory 0:7GiB,1:20GiB --gradient_checkpointing --eval_batch_size 32' bash run.sh Qwen/Qwen2.5-7B-Instruct
set -e
export TOKENIZERS_PARALLELISM=false
MODEL=${1:-xlm-roberta-base}; TAG=$(basename "$MODEL"); EXTRA=${EXTRA:-}
mkdir -p logs results/exp2
[ -f dataset/SentiMix/test.csv ] || python3 get_sentimix.py            # shared data pipeline (cp .env.example .env first)

# Task 1 - token-level LID tagger (frozen afterwards) -> switch points from predicted LID tags
LID=results/exp2/lid/xlm-roberta-base_full
[ -f $LID/test_pred_tags.csv ] || python3 lib/lid_tagger.py --model_name xlm-roberta-base --save_model 2>&1 | tee logs/lid_tagger.log

# Approach 5, part 1 - switch-point predictors after "Beyond Detection" (window-based BERT+RNN, transformer)
SW=results/exp2/switch
[ -f $SW/window_lstm_w5/test_pred_switch.csv ]   || python3 lib/switch_predictor.py --paradigm window --window 5          2>&1 | tee logs/switch_window_w5.log
[ -f $SW/window_lstm_wflex/test_pred_switch.csv ] || python3 lib/switch_predictor.py --paradigm window --window 0          2>&1 | tee logs/switch_window_flex.log
[ -f $SW/transformer_xlm-roberta-base_last2/test_pred_switch.csv ] || python3 lib/switch_predictor.py --paradigm transformer 2>&1 | tee logs/switch_transformer.log
python3 lib/report.py --switch_quality
BEST=$(python3 -c "import json,glob; m={f.split('/')[-2]: json.load(open(f))['dev']['flag_vs_gold_switch']['f1'] for f in glob.glob('results/exp2/switch/*/switch_metrics.json')}; print(max(m, key=m.get))")
PRED=$SW/$BEST; echo "$PRED" > $SW/best_predictor.txt           # best predictor by dev switch-F1 of its per-word flags

# Approaches 1-5 for the sentiment model
run() { python3 main.py --model_name "$MODEL" $EXTRA "$@" 2>&1 | tee "logs/${TAG}_$(echo "$*" | tr ' /' '__').log"; }
run --approach no_switch                                                           # 1
run --approach binary   --switch_source gold                                       # 2 (gold switch points)
run --approach binary   --switch_source lid --switch_dir $LID                      # 2 (switch points from the frozen LID tagger)
run --approach distance --switch_source gold                                       # 3
run --approach distance --switch_source lid --switch_dir $LID                      # 3
run --approach contrastive --contrastive_weight 0.1                                # 4
run --approach beyond_detection --pooling binary   --switch_dir $PRED              # 5
run --approach beyond_detection --pooling distance --switch_dir $PRED              # 5

# Evaluation: README tables + the shared evaluator (macro P/R/F1)
python3 lib/report.py; python3 lib/report.py --by_switches
python3 evaluate_all.py --ground results/exp2/ground.csv --pred-glob "results/exp2/*.csv" --out results/exp2/evaluation_results.csv
