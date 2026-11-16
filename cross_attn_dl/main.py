# PROGRAM GATEWAY - Joint contrastive learning of LID + sentiment on SentiMix Hi-En
#
# One shared encoder trained on both tasks at once (word-level LID {h,e,o} and tweet-level sentiment), with a
# supervised contrastive loss on each task and a cross-attention pooling that reads the LID posterior (lib/model.py).
# Modes (--mode), from the single-task baseline to the full model:
#   sentiment          sentiment only, attention pooling                                   (baseline, no LID)
#   joint              LID CE + sentiment CE, cross-attention pooling over LID-aware states (no contrastive)
#   joint_contrastive  joint + SupCon on the LID tokens + SupCon on the tweets             (the proposed model)
#   contrastive_lid    joint + SupCon on the LID tokens only          } ablations
#   contrastive_sent   joint + SupCon on the tweets only              }
# Project recipe: last 2 transformer blocks trained, seed 42, batch 32, 3 epochs, max length 128.
# Models: FacebookAI/xlm-roberta-base, Qwen/Qwen2.5-7B-Instruct, ai4bharat/IndicBERT-v3-1B.
# Metrics: macro precision / recall / F1 (and per class) for both tasks; nothing else.
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.train import train_and_evaluate  # noqa: E402

MODES = {  # mode -> (use_lid, fusion, contrastive)
    "sentiment": (False, "attn", "none"),
    "joint": (True, "cross_attn", "none"),
    "joint_contrastive": (True, "cross_attn", "both"),
    "contrastive_lid": (True, "cross_attn", "lid"),
    "contrastive_sent": (True, "cross_attn", "sent"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Joint contrastive LID + sentiment on SentiMix Hi-En")
    p.add_argument("--mode", default="joint_contrastive", choices=list(MODES))
    p.add_argument("--fusion", default=None, choices=["cross_attn", "attn"], help="override the pooling of the mode")
    # loss weights
    p.add_argument("--lid_weight", type=float, default=1.0, help="alpha: weight of the LID cross-entropy")
    p.add_argument("--con_weight_lid", type=float, default=0.1, help="lambda_lid: weight of the token-level SupCon")
    p.add_argument("--con_weight_sent", type=float, default=0.1, help="lambda_sent: weight of the tweet-level SupCon")
    p.add_argument("--tau", type=float, default=0.1, help="SupCon temperature")
    p.add_argument("--proj_dim", type=int, default=128, help="output size of the projection heads g_l, g_s")
    p.add_argument("--n_heads", type=int, default=8, help="heads of the cross-attention pooling")
    p.add_argument("--max_con_tokens", type=int, default=2048, help="cap on the tokens entering the LID SupCon per batch")
    # model / optimisation (project defaults)
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--unfreeze_last", type=int, default=2, help="transformer blocks to train (-1 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=None, help="lr of the heads (default: --lr)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--device_map", default="none", help="'auto' to shard a 7B model over all GPUs")
    p.add_argument("--max_memory", default=None, help="e.g. '0:7GiB,1:20GiB'")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None, help="smoke tests")
    p.add_argument("--max_eval_samples", type=int, default=None, help="smoke tests")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--run_name", default=None)
    p.add_argument("--save_model", action="store_true", help="write <checkpoint_dir>/<run>.pt (best dev epoch) for joint_contrastive.py")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    args = p.parse_args()
    args.use_lid, fusion, args.contrastive = MODES[args.mode]
    args.fusion = args.fusion or fusion
    return args


if __name__ == "__main__":
    train_and_evaluate(parse_args())
