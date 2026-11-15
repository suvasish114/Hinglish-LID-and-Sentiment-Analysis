#!/bin/bash
# Experiment 0 - SentiMix Hi-En sentiment baselines.
# Usage: bash run.sh            (needs .env, see .env.example; dataset is built by the shared get_sentimix.py)
# For SLURM clusters prepend the usual #SBATCH header and `source venv/bin/activate`.
set -e
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results/exp0
[ -f dataset/SentiMix/test.csv ] || python3 get_sentimix.py

# 1. the paper's baseline (Section 5): mBERT, full fine-tuning, max len 56, 3 epochs, AdamW 2e-5
python3 baseline.py --paper                                        2>&1 | tee logs/exp0_mbert_paper.log

# 2. project baselines: last 2 transformer blocks trainable, seed 42, batch 32, 3 epochs, lr 2e-5
python3 baseline.py --model_name bert-base-multilingual-cased      2>&1 | tee logs/exp0_mbert_last2.log
python3 baseline.py --model_name xlm-roberta-base                  2>&1 | tee logs/exp0_xlmr_last2.log
python3 baseline.py --model_name Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --device_map auto \
        --max_memory "0:7GiB,1:20GiB" --gradient_checkpointing --eval_batch_size 32 \
                                                                   2>&1 | tee logs/exp0_qwen7b_last2.log
# ai4bharat/IndicBERT-v3-1B is a gated checkpoint: run `hf auth login` and accept its terms first
# python3 baseline.py --model_name ai4bharat/IndicBERT-v3-1B --trust_remote_code --dtype bfloat16 --pooling mean 2>&1 | tee logs/exp0_indicbert_last2.log

# 3. evaluation: paper-style table (P/R/F1 per class, weighted F1, macro F1, micro F1) + shared evaluator
python3 eval_sentimix.py --ground results/exp0/ground.csv --pred-glob "results/exp0/*.csv"
python3 evaluate_all.py  --ground results/exp0/ground.csv --pred-glob "results/exp0/*.csv" --out results/exp0/evaluation_results.csv
