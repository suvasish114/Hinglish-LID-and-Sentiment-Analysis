import pandas as pd
import sys, csv, sys, ast
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)
sys.path.insert(0, "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/tools/LID_tool")
sys.path.insert(0, "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/tools/mallet-2.0.8")
import matplotlib.pyplot as plt
from getLanguage import langIdentify

# config
classifier = "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/tools/LID_tool/classifiers/HiEn.classifier"
INPUT_CSV = Path("/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/comi_lingua_dataset/test.csv")
OUTPUT_CSV = Path("/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/results/microsoft_lid.csv")
max_id = 1000

# helper functions
def normalize_label(label): # sanity clean
    return str(label).strip().lower()

def unwrap_prediction(prediction_result):
    if (isinstance(prediction_result, list) and len(prediction_result) == 1 and isinstance(prediction_result[0], list)):
        return prediction_result[0]
    return prediction_result

def ground_truth_to_pairs(ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = ast.literal_eval(ground_truth)
    return [(item["key"], normalize_label(item["value"])) for item in ground_truth]

def prediction_to_pairs(prediction_result):
    prediction_result = unwrap_prediction(prediction_result)
    return [(key, normalize_label(label)) for key, label in prediction_result]

def main(): # main function
    all_true_labels, all_predicted_labels, unmatched_tokens, skipped_row = [], [], [], []
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as ipfile, open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as opfile:
        reader = csv.DictReader(ipfile)
        writer = csv.DictWriter(opfile, fieldnames=["id", "predictions"])
        writer.writeheader() # write header
        for row in tqdm(reader, desc="Running LID"):
            sample_id = row["id"]
            if int(sample_id) >= max_id:
                break
            sentence = row["Sentences"]
            ground = row["Predicted tags"]
            ground_pairs = ground_truth_to_pairs(ground)
            try:
                output_tags = langIdentify(sentence, classifier)
            except:
                print(f"Error occured at index: {sample_id}")
                skipped_row.append(int(sample_id))
                continue
            predicted_pairs = prediction_to_pairs(output_tags)
            
            # Save model output in the CSV.
            writer.writerow({"id": sample_id, "predictions": repr(predicted_pairs)})

            # Match tokens using both the token text and its position.
            # This correctly handles repeated words such as "JDU".
            for index, (ground_token, true_label) in enumerate(ground_pairs):
                if index >= len(predicted_pairs):
                    unmatched_tokens.append(
                        f"id={sample_id}, index={index}, token={ground_token}: missing prediction"
                    )
                    continue

                predicted_token, predicted_label = predicted_pairs[index]

                if ground_token != predicted_token:
                    unmatched_tokens.append(
                        f"id={sample_id}, index={index}: "
                        f"ground={ground_token!r}, prediction={predicted_token!r}"
                    )
                    continue

                all_true_labels.append(true_label)
                all_predicted_labels.append(predicted_label)

            # Detect any extra predictions.
            if len(predicted_pairs) > len(ground_pairs):
                for index in range(len(ground_pairs), len(predicted_pairs)):
                    unmatched_tokens.append(
                        f"id={sample_id}, index={index}, "
                        f"extra prediction={predicted_pairs[index]!r}"
                    )

    # Calculate scores after processing all CSV rows.
    if not all_true_labels:
        raise ValueError("No matching tokens were found for evaluation.")

    labels = sorted(set(all_true_labels) | set(all_predicted_labels))

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels,
        all_predicted_labels,
        average="weighted",
        zero_division=0
    )

    accuracy = sum(actual == predicted for actual, predicted in zip(all_true_labels, all_predicted_labels)) / len(all_true_labels)

    print(f"Matched tokens: {len(all_true_labels)}")
    print(f"Unmatched tokens: {len(unmatched_tokens)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")

    print("\nPer-language results:")
    print(classification_report(
        all_true_labels,
        all_predicted_labels,
        labels=labels,
        zero_division=0
    ))

    # Confusion matrix: rows = ground truth, columns = prediction.
    matrix = confusion_matrix(
        all_true_labels,
        all_predicted_labels,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Language Identification Confusion Matrix")
    plt.tight_layout()
    plt.savefig("/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/plot/confusion_matrix.png", dpi=200)
    plt.close()

    # Optional: inspect token mismatches.
    if unmatched_tokens:
        print("\nFirst 10 unmatched tokens:")
        print("\n".join(unmatched_tokens[:10]))
    
    if len(skipped_row) >= 1:
        with open("skipped_rows.txt", "w") as file:
            for row_id in skipped_row:
                file.write(f"{row_id}, ")

if __name__ == "__main__": # driving code
    main()