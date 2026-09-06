"""Batch-evaluate every prediction CSV against the ground-truth file.

Metrics are computed in-process with pandas (no subprocess, no scikit-learn):
the ground truth is read once and each prediction file is joined to it on `id`.
"""

from pathlib import Path
import argparse
import glob

import pandas as pd

RESERVED_GROUND_COLS = {"id", "sentences"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground", type=Path, default=Path("ground.csv"),
                        help="Ground-truth CSV (needs an `id` column and a label column).")
    parser.add_argument("--out", type=Path, default=Path("evaluation_results.csv"),
                        help="Where to write the model-comparison table.")
    parser.add_argument("--pred-glob", default="*.csv",
                        help="Glob (relative to CWD) used to discover prediction files.")
    parser.add_argument("--label-col", default=None,
                        help="Ground-truth label column. Auto-detected when omitted.")
    parser.add_argument("--pred-col", default="predictions",
                        help="Prediction column inside each prediction CSV.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Compare labels verbatim instead of casefold()+strip().")
    return parser.parse_args()


def detect_label_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.strip().lower() not in RESERVED_GROUND_COLS]
    if len(candidates) != 1:
        raise SystemExit(
            f"ERROR: cannot auto-detect the label column from {list(df.columns)}. "
            f"Pass --label-col explicitly."
        )
    return candidates[0]


def macro_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Macro-averaged precision / recall / F1 over the classes present in y_true."""
    classes = sorted(y_true.unique())
    precisions, recalls, f1s = [], [], []
    for cls in classes:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    n = len(classes)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_precision": sum(precisions) / n if n else 0.0,
        "macro_recall": sum(recalls) / n if n else 0.0,
        "macro_f1": sum(f1s) / n if n else 0.0,
    }


def evaluate(ground: pd.DataFrame, label_col: str, pred_path: Path,
             pred_col: str, normalize: bool) -> dict:
    pred = pd.read_csv(pred_path)
    if "id" not in pred.columns or pred_col not in pred.columns:
        raise ValueError(f"needs columns 'id' and '{pred_col}', found {list(pred.columns)}")

    merged = ground.merge(
        pred[["id", pred_col]].rename(columns={pred_col: "_pred"}),
        on="id", how="inner",
    )
    if merged.empty:
        raise ValueError("no overlapping `id` values with the ground file")

    y_true = merged[label_col].astype(str)
    y_pred = merged["_pred"].astype(str)
    if normalize:
        y_true = y_true.str.strip().str.casefold()
        y_pred = y_pred.str.strip().str.casefold()

    return {
        "model": pred_path.stem,
        "n": len(merged),
        "missing": len(ground) - len(merged),
        **macro_metrics(y_true, y_pred),
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("COMI-LINGUA LID - BATCH EVALUATION")
    print("=" * 80)

    if not args.ground.exists():
        raise SystemExit(f"ERROR: {args.ground} not found.")

    ground = pd.read_csv(args.ground)                 # read once, reuse for every file
    if "id" not in ground.columns:
        raise SystemExit(f"ERROR: {args.ground} has no `id` column.")
    label_col = args.label_col or detect_label_col(ground)

    skip = {args.ground.name, args.out.name}
    pred_files = sorted(
        p for p in map(Path, glob.glob(args.pred_glob))
        if p.is_file() and p.name not in skip and "evaluation" not in p.name.lower()
    )
    if not pred_files:
        raise SystemExit(f"ERROR: no prediction files match {args.pred_glob!r}.")

    print(f"\nGround file : {args.ground}  (label column: {label_col!r})")
    print(f"Prediction files : {len(pred_files)}")

    results = []
    for pred_file in pred_files:
        try:
            row = evaluate(ground, label_col, pred_file, args.pred_col, not args.no_normalize)
        except (ValueError, pd.errors.ParserError) as exc:
            print(f"  - {pred_file.name}: SKIPPED ({exc})")
            continue
        print(f"  - {pred_file.name}: f1={row['macro_f1']:.4f}  n={row['n']}"
              + (f"  missing={row['missing']}" if row["missing"] else ""))
        results.append(row)

    if not results:
        raise SystemExit("\nNo evaluation results were generated.")

    results_df = (
        pd.DataFrame(results)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    results_df.to_csv(args.out, index=False)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + "=" * 80)
    print(f"Results saved to: {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
