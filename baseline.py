# PROGRAM GATEWAY - Experiment 0 (SentiMix Hi-En baselines)
#
# 1. Paper baseline (Patwa et al., 2020, Section 5): bert-base-multilingual-cased fine-tuned end-to-end,
#    max sequence length 56, 3 epochs, AdamW with lr 2e-5. Reported in the paper: weighted F1 = 65.4.
#       python3 baseline.py --paper
# 2. Project baselines: the same classifier with the models / hyper-parameters fixed for all experiments
#    (only the last 2 transformer blocks are trained, seed 42, batch size 32):
#       python3 baseline.py --model_name xlm-roberta-base
#       python3 baseline.py --model_name Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --device_map auto --gradient_checkpointing
#       python3 baseline.py --model_name ai4bharat/IndicBERT-v3-1B --trust_remote_code --pooling mean   (gated model: `hf auth login` first)
# Every run writes results/exp0/<run>.csv (id, predictions) + ground.csv for the shared evaluate_all.py,
# and eval_sentimix.py prints the paper-style table (per-class P/R/F1, weighted F1, macro F1, micro F1).
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.trainer import add_common_args, train_and_evaluate  # noqa: E402

PAPER = dict(model_name="bert-base-multilingual-cased", unfreeze_last=-1, max_length=56, epochs=3, lr=2e-5)


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 0: SentiMix Hi-En sentiment baselines")
    add_common_args(p)
    p.add_argument("--paper", action="store_true", help="exact Section-5 baseline configuration of the SentiMix paper")
    p.set_defaults(results_dir="results/exp0")
    args = p.parse_args()
    if args.paper:
        for k, v in PAPER.items():
            setattr(args, k, v)
        args.run_name = args.run_name or f"mBERT_paper-baseline_full_seed{args.seed}"
    return args


if __name__ == "__main__":
    train_and_evaluate(parse_args())
