"""Experiment 1 - word-level LID evaluation (token accuracy, per-tag P/R/F1, macro-F1).

Scores one or more prediction CSVs (`id, pred_tags` where pred_tags is a python list of word tags in
{h, e, o} or {hin, eng, o}) against the gold SentiMix tags of a split. Used to pick the best LID
tagger (fine-tuned encoders, the lexicon baseline, the Microsoft LID tool, LLM prompting, ...).

    python3 lid_eval.py --split test --pred-glob "results/exp1/lid/*/test_pred_tags.csv"
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from lib.sentimix_data import LID_TAGS, load_predicted_tags, load_split


def lid_scores(pred: dict, df) -> dict:
    """pred: id -> list[str] word tags. Words missing from a prediction count as wrong (`?`)."""
    y_true, y_pred = [], []
    for _, r in df.iterrows():
        p = list(pred.get(r["id"], []))
        p = (p + ["?"] * len(r["tags"]))[:len(r["tags"])]
        y_true += r["tags"]
        y_pred += p
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    out = {"n_words": int(len(y_true)), "token_accuracy": float((y_true == y_pred).mean())}
    f1s = []
    for c in LID_TAGS:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[f"precision_{c}"], out[f"recall_{c}"], out[f"f1_{c}"] = p, r, f
        f1s.append(f)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--pred-glob", default="results/exp1/lid/*/test_pred_tags.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    gold = load_split(args.split)
    rows = []
    for f in sorted(map(Path, glob.glob(args.pred_glob))):
        m = lid_scores(load_predicted_tags(f), gold)
        rows.append({"tagger": f.parent.name if f.name.endswith("_pred_tags.csv") else f.stem, **m})
        print(f"  - {rows[-1]['tagger']}: acc={m['token_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"F1(h/e/o)={m['f1_h']:.3f}/{m['f1_e']:.3f}/{m['f1_o']:.3f}")
    if not rows:
        raise SystemExit(f"no prediction files match {args.pred_glob!r}")
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out = args.out or Path(f"results/exp1/lid/lid_evaluation_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
