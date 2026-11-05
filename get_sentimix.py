# this script downloads dataset from HuggingFace and processes it in CSV format

import os, ast
import csv
import unicodedata
import pandas as pd
from huggingface_hub import snapshot_download

MODEL_PATH="model_checkpoints/"
HG_DATACARD="RTT1/SentiMix"
DATA_HOME="dataset/"
BATCH_SIZE = 8
EPOCH = 10

folder = os.path.join(DATA_HOME, "SentiMix")
tag_mapping = {"o": "o", "hin": "h", "eng": "e"}
if DATA_HOME is None:
    raise ValueError("DATA_HOME environment variable is not defined.")
os.makedirs(DATA_HOME, exist_ok=True)

def convert_tags(value):
    if pd.isna(value):
        return value
    tags = ast.literal_eval(value)
    return [tag_mapping.get(tag, tag) for tag in tags]

def load_dataset(datacard):
    dataset_dir = os.path.join(DATA_HOME, datacard.rsplit("/", 1)[-1])
    # A directory alone does not prove a previous download completed.
    print("[+] Loading dataset from HuggingFace...")
    try:
        snapshot_download(
            repo_id=datacard, repo_type="dataset", local_dir=dataset_dir
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download {datacard}: {exc}") from exc
    return dataset_dir


def process_dataset(dataset_dir, split_name="train", is_roman=False):
    files = sorted(
        name for name in os.listdir(dataset_dir)
        if name.startswith(f"{split_name}_")
        and os.path.isfile(os.path.join(dataset_dir, name))
        and not name.lower().endswith(".csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No source files matching {split_name}_* in {dataset_dir}. "
            f"Available files: {sorted(os.listdir(dataset_dir))}"
        )
    rows = []
    for name in files:
        current = None
        with open(os.path.join(dataset_dir, name), encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                line = line.rstrip("\r\n")
                if not line.strip():
                    if current is not None and current["sentence"]:
                        rows.append(current)
                    current = None
                    continue
                fields = line.split("\t")
                if fields[0].strip().lower() == "meta":
                    if current is not None and current["sentence"]:
                        rows.append(current)
                    if len(fields) < 2:
                        raise ValueError(f"{name}:{line_number}: missing sample ID")
                    current = {
                        "id": fields[1].strip(), "sentence": [], "tags": [],
                        "sentiment": fields[2].strip().lower() if len(fields) > 2 else "neutral",
                    }
                else:
                    if current is None or len(fields) < 2:
                        raise ValueError(f"{name}:{line_number}: malformed token row")
                    current["sentence"].append(fields[0].strip().lower())
                    tag = fields[1].strip().lower()
                    current["tags"].append("e" if tag == "emt" else tag_mapping.get(tag, tag))
        if current is not None and current["sentence"]:
            rows.append(current)
    selected = [row for row in rows if not is_roman or is_roman_text(" ".join(row["sentence"]))]
    if not selected:
        raise ValueError(f"No samples retained for {split_name}; inspect the source format.")
    output_file = os.path.join(dataset_dir, f"{split_name}.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "sentence", "tags", "sentiment"])
        writer.writeheader()
        writer.writerows(selected)
    print(f"[SUCCESS] Saved {len(selected)} / {len(rows)} samples to {output_file}")


def is_roman_text(sentence: str) -> bool:
    if not isinstance(sentence, str):
        return False
    for char in sentence:
        if char.isalpha():
            try:
                char_name = unicodedata.name(char)
            except ValueError:
                continue
            if "LATIN" not in char_name:
                return False
    return True

def replace_emt_tags(csv_path):
    df = pd.read_csv(csv_path)
    def replace_tags(value):
        if pd.isna(value):
            return value

        tags = ast.literal_eval(value)
        if not isinstance(tags, list):
            raise ValueError(f"Expected a list of tags, got: {value!r}")
        return repr(["e" if tag == "emt" else tag for tag in tags])
    df["tags"] = df["tags"].apply(replace_tags)
    df.to_csv(csv_path, index=False)

def main():
    dataset_dir = load_dataset(HG_DATACARD)
    labels_path = os.path.join(dataset_dir, "test_labels_hinglish.txt")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing test labels: {labels_path}")
    for split in ("train", "dev", "Hindi"):
        process_dataset(dataset_dir, split, is_roman=True)
    conn = pd.read_csv(os.path.join(dataset_dir, "Hindi.csv"), dtype={"id": "string"})
    test = pd.read_csv(labels_path, dtype={"Uid": "string"})
    test = test.rename(columns={"Uid": "id", "Sentiment": "sentiment"})
    if not {"id", "sentiment"}.issubset(test.columns):
        raise ValueError(f"Unexpected test-label columns: {list(test.columns)}")
    lookup = conn.drop_duplicates(subset="id", keep="first").set_index("id")[["sentence", "tags"]]
    pos = test.columns.get_loc("id") + 1
    for offset, col in enumerate(["sentence", "tags"]):
        test.insert(pos + offset, col, test["id"].map(lookup[col]))
    test = test.dropna(subset=["sentence", "tags"])
    if test.empty:
        raise ValueError("No test-label IDs matched retained Hindi samples.")
    test.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)
    print(f"[SUCCESS] Saved {len(test)} test samples")
    # Keep source files and Hindi.csv so reruns remain possible.
    for filename in os.listdir(folder): # remove all files does not ends with .csv
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and not filename.lower().endswith(".csv"):
            os.remove(file_path)
    os.remove(os.path.join(folder, "Hindi.csv"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[FAILED] {exc}") from exc
