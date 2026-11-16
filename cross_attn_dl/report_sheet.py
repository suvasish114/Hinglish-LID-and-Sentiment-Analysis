"""Rows for the group's result sheet (one per run): Model Name, Parameters, Adaptation, Approach,
LID accuracy / precision / recall / F1, sentiment accuracy / precision / recall / F1, Comments.
Precision / recall / F1 are macro-averaged, values as fractions (0.8884), recomputed from the saved predictions.

    python3 report_sheet.py [--model xlm-roberta-base] [--results_dir results]    -> results/sheet_rows.tsv (paste into the sheet)
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from lib.data import LABELS, LID_TAGS, load_split

PARAMS = {"xlm-roberta-base": "0.3B", "Qwen2.5-7B-Instruct": "7.6B", "IndicBERT-v3-1B": "1B"}
APPROACH = {
    "sentiment": "Single-task baseline: attention pooling, sentiment only",
    "joint": "Joint LID + sentiment (multi-task, cross-attention pooling over LID-aware token states), no contrastive loss",
    "joint_contrastive": "Joint contrastive learning: LID + sentiment with SupCon on LID tokens and on tweets, cross-attention fusion",
    "contrastive_lid": "Joint LID + sentiment with SupCon on LID tokens only (ablation)",
    "contrastive_sent": "Joint LID + sentiment with SupCon on tweets only (ablation)",
}
COMMENT = "last 2 transformer blocks + heads trained, seed 42, batch 32, 3 epochs, best-dev checkpoint, macro P/R/F1"


def prf(y_true, y_pred, classes):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    return [accuracy_score(y_true, y_pred), p, r, f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--model", default=None, help="only runs of this model tag (e.g. xlm-roberta-base)")
    args = ap.parse_args()
    rd = Path(args.results_dir)
    test = load_split("test")
    gold_sent = dict(zip(test["id"], test["sentiment"]))
    gold_tags = dict(zip(test["id"], test["tags"]))

    rows = []
    for mf in sorted(rd.glob("*_metrics.json")):
        m = json.load(open(mf))
        run, a = m["run_name"], m["args"]
        tag = a["model_name"].split("/")[-1]
        if args.model and tag != args.model:
            continue
        pred = pd.read_csv(rd / f"{run}.csv", dtype=str)
        sent = prf([gold_sent[i] for i in pred["id"]], pred["predictions"].tolist(), LABELS)
        lid = [None] * 4
        if (rd / f"{run}_lid.csv").exists():
            lp = pd.read_csv(rd / f"{run}_lid.csv", dtype=str)
            yt, yp = [], []
            for i, s in zip(lp["id"], lp["pred_tags"]):
                yt += gold_tags[i]
                yp += ast.literal_eval(s)
            lid = prf(yt, yp, LID_TAGS)
        fmt = lambda v: "" if v is None else f"{v:.4f}"
        rows.append([tag, PARAMS.get(tag, ""), "Yes", APPROACH[a["mode"]], *map(fmt, lid), *map(fmt, sent), COMMENT])

    header = ["Model Name", "Parameters", "Adaptation", "Approach", "LID Accuracy", "LID Precision", "LID Recall", "LID F1",
              "SA Accuracy", "SA Precision", "SA Recall", "SA F1", "Comments"]
    order = list(APPROACH)
    rows.sort(key=lambda r: (r[0], order.index(next(k for k, v in APPROACH.items() if v == r[3]))))
    out = rd / "sheet_rows.tsv"
    out.write_text("\t".join(header) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
    print("\t".join(header))
    for r in rows:
        print("\t".join(r))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
