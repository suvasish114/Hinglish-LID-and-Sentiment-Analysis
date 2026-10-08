"""Experiment 2 - result tables for the README.

    python3 lib/report.py                       # sentiment runs: one row per configuration, mean +- std over seeds
    python3 lib/report.py --by_switches         # seed-42 runs: weighted F1 by number of switch points in the tweet
    python3 lib/report.py --switch_quality      # how good are the switch points of each source (lid taggers, predictors)?
"""

import argparse
import ast
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import load_predicted_tags, load_split          # noqa: E402
from lib.switch import switch_points                          # noqa: E402

APPROACH_NAMES = {"no_switch": "1 no switch (attention pooling)", "binary": "2 binary switch embedding",
                  "distance": "3 distance-to-switch embedding", "contrastive": "4 contrastive distillation (LASER3-CO)",
                  "beyond_detection": "5 Beyond-Detection predicted switches"}


def fmt(vals):
    v = 100 * np.asarray(vals, dtype=float)
    return f"{v.mean():.1f} +- {v.std():.1f}" if len(v) > 1 else f"{v.mean():.1f}"


def load_runs(results_dir):
    runs = []
    for f in sorted(glob.glob(str(Path(results_dir) / "*_metrics.json"))):
        m = json.load(open(f))
        a = m["args"]
        runs.append({"config": re.sub(r"_seed\d+$", "", m["run_name"]), "seed": a["seed"], "model": a["model_name"].split("/")[-1],
                     "approach": APPROACH_NAMES.get(a["approach"], a["approach"]),
                     "switch source": "-" if a["pooling"] == "none" else a["switch_source"] + (f" ({Path(a['switch_dir']).name})" if a.get("switch_dir") else ""),
                     "pooling": a["pooling"], "test": m["test"], "dev": m["dev"], "minutes": m["train_minutes"]})
    return runs


def table(results_dir):
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no *_metrics.json in {results_dir}")
    groups = {}
    for r in runs:
        groups.setdefault(r["config"], []).append(r)
    rows = []
    for cfg, rs in groups.items():
        r0 = rs[0]
        rows.append({"model": r0["model"], "approach": r0["approach"], "pooling": r0["pooling"], "switch source": r0["switch source"],
                     "seeds": ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"])),
                     "weighted F1": fmt([r["test"]["weighted_f1"] for r in rs]), "macro F1": fmt([r["test"]["macro_f1"] for r in rs]),
                     "acc": fmt([r["test"]["accuracy"] for r in rs]), "dev wF1": fmt([r["dev"]["weighted_f1"] for r in rs]),
                     "min/run": f"{np.mean([r['minutes'] for r in rs]):.0f}", "_sort": (r0["model"], r0["approach"], r0["pooling"], r0["switch source"])})
    rows.sort(key=lambda r: r["_sort"])
    cols = [c for c in rows[0] if c != "_sort"]
    print("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def by_switches(results_dir, seed=42):
    runs = [r for r in load_runs(results_dir) if r["seed"] == seed]
    buckets = ["switches_1-2", "switches_3-5", "switches_6+"]
    cols = ["model", "approach", "switch source", "all"] + [f"{b.split('_')[1]} switches (n={runs[0]['test'][b]['n']})" for b in buckets if b in runs[0]["test"]]
    print("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|")
    for r in sorted(runs, key=lambda r: (r["model"], r["approach"], r["switch source"])):
        vals = [f"{100 * r['test']['weighted_f1']:.1f}"] + [f"{100 * r['test'][b]['weighted_f1']:.1f}" for b in buckets if b in r["test"]]
        print(f"| {r['model']} | {r['approach']} | {r['switch source']} | " + " | ".join(vals) + " |")


def switch_quality(results_dir, split="test"):
    gold = load_split(split)
    gold_sw = {r["id"]: switch_points(r["tags"]) for _, r in gold.iterrows()}
    rows = []
    for f in sorted(glob.glob(str(Path(results_dir) / "lid" / "*" / f"{split}_pred_tags.csv"))):
        tags = load_predicted_tags(f)
        rows.append(("lid tagger: " + Path(f).parent.name, {i: switch_points(t) for i, t in tags.items()}))
    for f in sorted(glob.glob(str(Path(results_dir) / "switch" / "*" / f"{split}_pred_switch.csv"))):
        df = pd.read_csv(f, dtype=str)
        rows.append(("switch predictor: " + Path(f).parent.name, {r["id"]: [int(v) for v in ast.literal_eval(r["pred_switch"])] for _, r in df.iterrows()}))
    print(f"| switch-point source | precision | recall | F1 (switch class) | predicted switches / gold switches |\n|---|---|---|---|---|")
    for name, pred in rows:
        tp = fp = fn = 0
        for i, g in gold_sw.items():
            p = (list(pred.get(i, [])) + [0] * len(g))[:len(g)]
            tp += sum(1 for a, b in zip(p, g) if a == 1 and b == 1)
            fp += sum(1 for a, b in zip(p, g) if a == 1 and b == 0)
            fn += sum(1 for a, b in zip(p, g) if a == 0 and b == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"| {name} | {100 * prec:.1f} | {100 * rec:.1f} | {100 * f1:.1f} | {tp + fp} / {tp + fn} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/exp2")
    ap.add_argument("--by_switches", action="store_true")
    ap.add_argument("--switch_quality", action="store_true")
    a = ap.parse_args()
    if a.switch_quality:
        switch_quality(a.results_dir)
    elif a.by_switches:
        by_switches(a.results_dir)
    else:
        table(a.results_dir)
