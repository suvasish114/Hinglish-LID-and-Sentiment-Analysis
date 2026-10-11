"""Markdown summary of every `<run>_metrics.json` in a results directory (the tables in the README).

    python3 summarize_results.py --results_dir results/exp0
    python3 summarize_results.py --results_dir results/exp1 --lid      # adds the LID-fusion columns
    python3 summarize_results.py --results_dir results/exp1 --lid --aggregate   # mean +- std over seeds per configuration
"""

import argparse
import glob
import json
import re
from pathlib import Path


def fmt(x):
    return f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/exp0")
    ap.add_argument("--lid", action="store_true", help="show LID fusion / source columns (Experiment 1)")
    ap.add_argument("--aggregate", action="store_true", help="group runs that differ only by seed: mean +- std")
    args = ap.parse_args()
    rows, raw = [], []
    for f in sorted(glob.glob(str(Path(args.results_dir) / "*_metrics.json"))):
        m = json.load(open(f))
        a, t, d = m["args"], m["test"], m["dev"]
        trainable = re.search(r"trainable=([\d.]+M)", m.get("model", ""))
        row = {"run": m["run_name"], "model": a["model_name"].split("/")[-1],
               "trained blocks": "all" if a["unfreeze_last"] < 0 else f"last {a['unfreeze_last']}",
               "trainable": trainable.group(1) if trainable else "?"}
        if args.lid:
            fusion = a.get("lid_fusion", "none")
            row["LID fusion"] = "none (baseline)" if fusion == "none" else f"{fusion} @ block {a.get('lid_inject_layer', 0)}"
            row["LID tags"] = "-" if fusion == "none" else a.get("lid_source", "gold")
        row.update({"Pos F1": fmt(t["positive_f1"]), "Neu F1": fmt(t["neutral_f1"]), "Neg F1": fmt(t["negative_f1"]),
                    "weighted F1": fmt(t["weighted_f1"]), "macro F1": fmt(t["macro_f1"]), "acc (micro F1)": fmt(t["accuracy"]),
                    "dev wF1": fmt(d["weighted_f1"]), "min": f"{m['train_minutes']:.0f}"})
        rows.append(row)
        raw.append((re.sub(r"_seed\d+$", "", m["run_name"]), a["seed"], t, d, row))
    if not rows:
        raise SystemExit(f"no *_metrics.json in {args.results_dir}")
    if args.aggregate:
        import numpy as np
        groups = {}
        for cfg, seed, t, d, row in raw:
            groups.setdefault(cfg, []).append((seed, t, d, row))
        rows = []
        for cfg, items in groups.items():
            row = {k: v for k, v in items[0][3].items() if k not in ("Pos F1", "Neu F1", "Neg F1", "weighted F1", "macro F1", "acc (micro F1)", "dev wF1", "min")}
            row["run"] = cfg
            row["seeds"] = ",".join(str(i[0]) for i in sorted(items, key=lambda i: i[0]))
            for col, key in [("Pos F1", "positive_f1"), ("Neu F1", "neutral_f1"), ("Neg F1", "negative_f1"),
                             ("weighted F1", "weighted_f1"), ("macro F1", "macro_f1"), ("acc (micro F1)", "accuracy")]:
                v = np.array([100 * i[1][key] for i in items])
                row[col] = f"{v.mean():.1f} +- {v.std(ddof=0):.1f}" if len(v) > 1 else f"{v.mean():.1f}"
            v = np.array([100 * i[2]["weighted_f1"] for i in items])
            row["dev wF1"] = f"{v.mean():.1f} +- {v.std(ddof=0):.1f}" if len(v) > 1 else f"{v.mean():.1f}"
            rows.append(row)
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in sorted(rows, key=lambda r: -float(str(r["weighted F1"]).split()[0])):
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
