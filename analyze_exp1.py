"""Experiment 1 - does the LID signal help more on heavily code-mixed tweets?

Buckets the test tweets by their Code-Mixing Index (CMI, Gambäck & Das 2016; also reported in the SentiMix
paper): CMI = 100 * (1 - max_l N_l / (N - U)), where N_l counts the tokens of language l (h / e) and U the
language-independent tokens (o). CMI 0 = monolingual tweet. For every prediction CSV matching --pred-glob it
prints the weighted F1 per bucket, so a baseline run and its LID-fusion variants can be compared side by side.

    python3 analyze_exp1.py --pred-glob "results/exp1/xlm-roberta-base_last2*seed42.csv"
"""

import argparse
import glob
from collections import Counter
from pathlib import Path

import pandas as pd

from lib.metrics import sentiment_metrics
from lib.sentimix_data import load_split

BUCKETS = [(-1, 0, "CMI = 0 (monolingual)"), (0, 20, "0 < CMI <= 20"), (20, 40, "20 < CMI <= 40"), (40, 101, "CMI > 40")]


def cmi(tags) -> float:
    c = Counter(tags)
    n, u = len(tags), c.get("o", 0)
    if n - u == 0:
        return 0.0
    return 100.0 * (1.0 - max(c.get("h", 0), c.get("e", 0)) / (n - u))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-glob", default="results/exp1/*seed42.csv")
    ap.add_argument("--out", type=Path, default=Path("results/exp1/analysis_cmi.csv"))
    args = ap.parse_args()
    test = load_split("test")
    test["cmi"] = test["tags"].apply(cmi)
    test["bucket"] = pd.cut(test["cmi"], bins=[b[0] for b in BUCKETS] + [BUCKETS[-1][1]], labels=[b[2] for b in BUCKETS])
    sizes = test["bucket"].value_counts().reindex([b[2] for b in BUCKETS])
    print(f"test tweets by CMI bucket: {sizes.to_dict()}   (mean CMI {test['cmi'].mean():.1f})")
    rows = []
    for f in sorted(map(Path, glob.glob(args.pred_glob))):
        if f.name == "ground.csv" or "evaluation" in f.name or f.stat().st_size == 0:
            continue
        pred = pd.read_csv(f, dtype=str)
        if "predictions" not in pred.columns:
            continue
        m = test.merge(pred, on="id")
        row = {"run": f.stem, "all": 100 * sentiment_metrics(m["sentiment"], m["predictions"])["weighted_f1"]}
        for b in BUCKETS:
            sub = m[m["bucket"] == b[2]]
            row[b[2]] = 100 * sentiment_metrics(sub["sentiment"], sub["predictions"])["weighted_f1"] if len(sub) else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("all", ascending=False)
    print("\nweighted F1 (%) per CMI bucket\n" + df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
