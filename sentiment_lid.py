# PROGRAM GATEWAY - Experiment 1 (LID + Sentiment)
#
# Research question: do word-level LID tags improve sentence-level sentiment classification?
#   --lid_fusion none              baseline (Experiment 0 model, no LID information)
#   --lid_fusion concat|add        LID-tag embedding fused into the hidden states entering block
#                                  --lid_inject_layer (0 = input embeddings)
#   --lid_source gold|predicted    gold SentiMix tags, or tags from the frozen tagger of lid_tagger.py
#                                  (--pred_tags_dir must hold {train,dev,test}_pred_tags.csv)
# The LID tagger is never updated: it only provides discrete tags, so no gradient reaches it.
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.sentiment_model import LID_FUSIONS      # noqa: E402
from lib.trainer import add_common_args, train_and_evaluate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1: LID-aware sentiment classification on SentiMix Hi-En")
    add_common_args(p)
    p.add_argument("--lid_fusion", default="concat", choices=LID_FUSIONS)
    p.add_argument("--lid_dim", type=int, default=16, help="size of the LID tag embedding")
    p.add_argument("--lid_inject_layer", type=int, default=0, help="transformer block whose input receives the LID signal")
    p.add_argument("--lid_source", default="gold", choices=["gold", "predicted"])
    p.add_argument("--pred_tags_dir", default="results/exp1/lid/xlm-roberta-base_full",
                   help="directory with {train,dev,test}_pred_tags.csv written by lid_tagger.py")
    p.set_defaults(results_dir="results/exp1")
    return p.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
