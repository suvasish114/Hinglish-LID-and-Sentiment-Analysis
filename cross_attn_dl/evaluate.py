"""Collect the <run>_metrics.json files of a results directory into one table (macro P / R / F1 of both tasks,
test split by default) and write it as CSV + markdown.

    python3 evaluate.py                                   # results/ -> results/evaluation_results.{csv,md}
    python3 evaluate.py --split dev --per_class
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--per_class", action="store_true", help="also list the per-class F1 of both tasks")
    args = p.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(args.results_dir) / "*_metrics.json"))):
        m = json.load(open(f))
        a, s = m["args"], m.get(args.split)
        if not s:                                  # e.g. prompting baselines are scored on the test split only
            continue
        row = {"model": a["model_name"].split("/")[-1], "mode": a["mode"], "fusion": a["fusion"], "contrastive": a["contrastive"],
               "seed": a["seed"], "run": m["run_name"]}
        for task in ["sentiment", "lid"]:
            t = s.get(task)
            for k in ["precision", "recall", "f1"]:
                row[f"{task}_{k}"] = round(100 * t[f"macro_{k}"], 2) if t else None
            if args.per_class and t:
                for c, v in t["per_class"].items():
                    row[f"{task}_{c}_f1"] = round(100 * v["f1"], 2)
        row["minutes"] = round(m["train_minutes"], 1)
        rows.append(row)
    if not rows:
        print(f"no *_metrics.json in {args.results_dir}")
        return
    df = pd.DataFrame(rows)
    out = Path(args.results_dir) / f"evaluation_results_{args.split}"
    df.to_csv(out.with_suffix(".csv"), index=False)
    md = to_markdown(df.fillna("-"))
    out.with_suffix(".md").write_text(md + "\n")
    print(f"[{args.split}] macro precision / recall / F1 (x100)\n{md}\n-> {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
