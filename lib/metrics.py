"""Sentiment metrics in the format of the SentiMix paper (Patwa et al., 2020, Table 3).

The official SemEval-2020 Task 9 metric is the F1 score averaged over the three classes weighted by
support ("weighted F1"). We additionally report macro P/R/F1 (used by the shared `evaluate_all.py`)
and micro-F1, which for single-label classification equals accuracy.
"""

from sklearn.metrics import precision_recall_fscore_support

CLASSES = ["positive", "neutral", "negative"]           # order of the paper's table


def sentiment_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    out = {"n": len(y_true)}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    for c, pc, rc, fc, sc in zip(CLASSES, p, r, f, s):
        out[f"{c}_precision"], out[f"{c}_recall"], out[f"{c}_f1"], out[f"{c}_support"] = float(pc), float(rc), float(fc), int(sc)
    for avg in ["weighted", "macro", "micro"]:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, average=avg, zero_division=0)
        out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = float(p), float(r), float(f)
    out["accuracy"] = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    return out


def paper_row(name: str, m: dict) -> str:
    """One row in the layout of Table 3 of the paper (scores in %)."""
    cells = [name[:34].ljust(34)]
    for c in CLASSES:
        cells += [f"{100 * m[f'{c}_precision']:5.1f}", f"{100 * m[f'{c}_recall']:5.1f}", f"{100 * m[f'{c}_f1']:5.1f}"]
    cells += [f"{100 * m['weighted_f1']:5.1f}", f"{100 * m['macro_f1']:5.1f}", f"{100 * m['micro_f1']:5.1f}"]
    return " | ".join(cells)


def paper_header() -> str:
    head = ["system".ljust(34)]
    for c in CLASSES:
        head += [f"{c[:3].title()}-P", f"{c[:3].title()}-R", f"{c[:3].title()}-F1"]
    head += ["wF1 ", "mF1 ", "µF1 "]
    return " | ".join(head)
