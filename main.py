# PROGRAM GATEWAY - Experiment 2 (Switch Point + Sentiment)
#
# Research question: does giving the sentiment model the language switch-point information improve the score?
# Five approaches, selected with --approach (details and equations in README / lib/model.py):
#   1 no_switch         attention pooling a_i = softmax(w^T h_i + b), no switch information          (baseline)
#   2 binary            switch-biased attention pooling with a learned embedding of the binary switch flag
#   3 distance          same, with an embedding of the bucketed distance to the nearest switch (0,1,2,3+)
#   4 contrastive       approach-1 pooling + LASER3-CO contrastive distillation loss on language views (lib/model.py)
#   5 beyond_detection  approach-2/3 pooling with switch points PREDICTED by lib/switch_predictor.py
#                       (window-based BERT+RNN or transformer token classifier, after "Beyond Detection")
# Switch points for 2/3 come from --switch_source gold (SentiMix LID tags) or lid (frozen LID tagger, lib/lid_tagger.py).
# Project hyper-parameters: last 2 transformer blocks trained, seed 42, batch 32 (temperature does not apply: a
# classification head, no generation). Models: Qwen/Qwen2.5-7B-Instruct, FacebookAI/xlm-roberta-base.
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.train import SWITCH_SOURCES, train_and_evaluate  # noqa: E402

APPROACHES = {"no_switch": 1, "binary": 2, "distance": 3, "contrastive": 4, "beyond_detection": 5}


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2: switch-point aware sentiment classification on SentiMix Hi-En")
    p.add_argument("--approach", default="no_switch", choices=list(APPROACHES))
    p.add_argument("--pooling", default=None, choices=["binary", "distance"],
                   help="approach 5 only: which switch-aware pooling to feed the predicted switches to (default binary)")
    p.add_argument("--switch_source", default="gold", choices=SWITCH_SOURCES, help="approaches 2/3: gold or lid")
    p.add_argument("--switch_dir", default=None, help="dir with {split}_pred_tags.csv (lid) or {split}_pred_switch.csv (predictor)")
    p.add_argument("--switch_dim", type=int, default=16, help="size of e^sw (8-16 recommended)")
    # approach 4
    p.add_argument("--contrastive_weight", type=float, default=0.1, help="weight of the contrastive loss added to CE")
    p.add_argument("--contrastive_tau", type=float, default=0.05)
    p.add_argument("--contrastive_queue", type=int, default=4096)
    p.add_argument("--contrastive_filter", type=float, default=None, help="sigma of LASER3-CO-Filter (e.g. 0.9); off by default")
    # model / optimisation (project defaults)
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--unfreeze_last", type=int, default=2, help="transformer blocks to train (-1 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=None, help="lr of pooling + classifier (default: --lr)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--select_metric", default="weighted_f1", choices=["weighted_f1", "macro_f1", "accuracy"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--device_map", default="none", help="'auto' to shard a 7B model over all GPUs")
    p.add_argument("--max_memory", default=None, help="e.g. '0:7GiB,1:20GiB'")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None, help="smoke tests")
    p.add_argument("--max_eval_samples", type=int, default=None, help="smoke tests")
    p.add_argument("--results_dir", default="results/exp2")
    p.add_argument("--run_name", default=None)
    args = p.parse_args()

    args.approach_id = APPROACHES[args.approach]
    if args.approach in ("no_switch", "contrastive"):
        args.pooling, args.switch_source = "none", "gold"
    elif args.approach in ("binary", "distance"):
        args.pooling = args.approach
        if args.switch_source == "predictor":
            p.error("approaches 2/3 take --switch_source gold|lid; predicted switch points are approach 5")
    else:                                                     # beyond_detection
        args.pooling = args.pooling or "binary"
        args.switch_source = "predictor"
    if args.switch_source != "gold" and not args.switch_dir:
        p.error(f"--switch_dir is required for --switch_source {args.switch_source}")
    return args


if __name__ == "__main__":
    train_and_evaluate(parse_args())
