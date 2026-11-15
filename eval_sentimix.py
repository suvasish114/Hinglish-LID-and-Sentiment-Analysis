"""Experiment 0 - evaluation in the format of the SentiMix paper (Patwa et al., 2020, Table 3).

For every prediction CSV (`id, predictions`) matching --pred-glob it reports, against the ground-truth
CSV (`id, sentiment`): per-class precision / recall / F1 for Positive, Neutral and Negative, the
support-weighted F1 (the official SemEval-2020 Task 9 metric), macro P/R/F1 (the metric used by the
shared `evaluate_all.py`) and micro-F1 (= accuracy). The paper's own mBERT baseline row is printed
for reference.

    python3 eval_sentimix.py --ground results/exp0/ground.csv --pred-glob "results/exp0/*.csv"
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

from lib.metrics import CLASSES, paper_header, paper_row, sentiment_metrics

PAPER_BASELINE = {  # Table 3, rank 45, "Baseline" (bert-base-multilingual-cased, max len 56, 3 epochs, AdamW 2e-5)
    "positive_precision": .728, "positive_recall": .688, "positive_f1": .707,
    "neutral_precision": .562, "neutral_recall": .602, "neutral_f1": .581,
    "negative_precision": .691, "negative_recall": .674, "negative_f1": .683,
    "weighted_f1": .654, "macro_f1": float("nan"), "micro_f1": float("nan"),
}
PAPER_BEST = {**{k: float("nan") for k in PAPER_BASELINE}, "positive_precision": .843, "positive_recall": .760,
              "positive_f1": .799, "neutral_precision": .652, "neutral_recall": .731, "neutral_f1": .689,
              "negative_precision": .785, "negative_recall": .754, "negative_f1": .769, "weighted_f1": .750}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ground", type=Path, default=Path("results/exp0/ground.csv"))
    p.add_argument("--pred-glob", default="results/exp0/*.csv")
    p.add_argument("--pred-col", default="predictions")
    p.add_argument("--label-col", default="sentiment")
    p.add_argument("--out", type=Path, default=None, help="CSV with all metrics (default: <ground dir>/evaluation_paper_style.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    ground = pd.read_csv(args.ground, dtype=str)
    ground[args.label_col] = ground[args.label_col].str.strip().str.lower()
    out_path = args.out or args.ground.parent / "evaluation_paper_style.csv"
    skip = {args.ground.name, out_path.name}
    rows = []
    for f in sorted(map(Path, glob.glob(args.pred_glob))):
        if f.name in skip or "evaluation" in f.name.lower() or "pred_tags" in f.name:
            continue
        pred = pd.read_csv(f, dtype=str)
        if "id" not in pred.columns or args.pred_col not in pred.columns:
            print(f"  - {f.name}: SKIPPED (needs columns id,{args.pred_col})")
            continue
        merged = ground.merge(pred[["id", args.pred_col]], on="id", how="inner")
        if merged.empty:
            print(f"  - {f.name}: SKIPPED (no overlapping ids)")
            continue
        m = sentiment_metrics(merged[args.label_col], merged[args.pred_col].str.strip().str.lower())
        m["missing"] = len(ground) - len(merged)
        rows.append({"system": f.stem, **m})

    print("=" * 140)
    print("SENTIMIX HI-EN TEST - PAPER-STYLE EVALUATION (scores in %; wF1 = weighted F1 = official metric, mF1 = macro, µF1 = micro = accuracy)")
    print("=" * 140)
    print(paper_header())
    print("-" * 140)
    for r in sorted(rows, key=lambda r: -r["weighted_f1"]):
        print(paper_row(r["system"], r))
    print("-" * 140)
    print(paper_row("paper: mBERT baseline (rank 45)", PAPER_BASELINE))
    print(paper_row("paper: best system KK2018 (rank 1)", PAPER_BEST))
    if rows:
        cols = ["system", "n", "missing"] + [f"{c}_{m}" for c in CLASSES for m in ["precision", "recall", "f1"]] + \
               ["weighted_precision", "weighted_recall", "weighted_f1", "macro_precision", "macro_recall", "macro_f1",
                "micro_f1", "accuracy"]
        pd.DataFrame(rows)[cols].sort_values("weighted_f1", ascending=False).to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
