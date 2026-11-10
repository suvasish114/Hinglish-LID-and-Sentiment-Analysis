"""Evaluate sentiment predictions matched by ID and save a confusion matrix.

Usage: python eval.py ground.csv predicted.csv
"""

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def _read_sentiments(path):
    """Read string IDs (preserving leading zeros) and reject ambiguous rows."""
    sentiments = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "sentiment"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns are 'id' and 'sentiment'.")
        for row_number, row in enumerate(reader, start=2):
            identifier = (row.get("id") or "").strip()
            sentiment = (row.get("sentiment") or "").strip()
            if not identifier or not sentiment:
                raise ValueError(f"{path}: empty id or sentiment at row {row_number}.")
            if identifier in sentiments:
                raise ValueError(f"{path}: duplicate id {identifier!r}.")
            sentiments[identifier] = sentiment
    return sentiments


def evaluate_sentiment(ground_csv, predicted_csv, output_path=None, average="macro"):
    """Return precision, recall, F1 and micro-F1, and save a PNG confusion matrix.

    Both files must contain unique, nonempty 'id' and 'sentiment' columns
    and the same set of IDs. Row order and extra columns do not matter.
    Labels are case-sensitive; surrounding whitespace is removed.
    Macro averaging weights each sentiment equally; 'micro' and 'weighted'
    are also supported. Undefined metrics are returned as zero.
    Micro-F1 is always reported independently of the selected averaging mode.
    The default plot location is confusion_matrix.png beside this script.
    """
    if average not in {"macro", "micro", "weighted"}:
        raise ValueError("average must be 'macro', 'micro', or 'weighted'.")
    ground = _read_sentiments(ground_csv)
    predicted = _read_sentiments(predicted_csv)
    missing = ground.keys() - predicted.keys()
    extra = predicted.keys() - ground.keys()
    if missing or extra:
        raise ValueError(
            f"ID sets differ: {len(missing)} missing predictions, "
            f"{len(extra)} unexpected prediction IDs. "
            f"Missing examples: {sorted(missing)[:5]}; "
            f"unexpected examples: {sorted(extra)[:5]}."
        )
    if not ground:
        raise ValueError("The CSV files contain no data rows.")

    y_true = list(ground.values())
    y_pred = [predicted[identifier] for identifier in ground]
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0
    )
    micro_f1 = f1_score(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Use a noninteractive canvas so the function also works on servers.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sklearn.metrics import ConfusionMatrixDisplay

    destination = (
        Path(output_path) if output_path is not None
        else Path(__file__).resolve().parent / "confusion_matrix.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = max(6, len(labels) * 0.8)
    figure = Figure(figsize=(size, size))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axes, cmap="Blues", values_format="d", xticks_rotation=45
    )
    axes.set_title(f"Sentiment confusion matrix ({len(y_true)} samples)")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "micro_f1": float(micro_f1),
        "average": average,
        "samples": len(y_true),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_path": str(destination.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_csv", nargs="?", default="ground.csv")
    parser.add_argument("predicted_csv", nargs="?", default="predicted.csv")
    parser.add_argument("--output", help="Confusion matrix image path")
    parser.add_argument("--average", choices=["macro", "micro", "weighted"], default="macro")
    args = parser.parse_args()
    try:
        result = evaluate_sentiment(
            args.ground_csv, args.predicted_csv, args.output, args.average
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
