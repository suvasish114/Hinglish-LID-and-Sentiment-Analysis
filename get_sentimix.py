# this script downloads dataset from HuggingFace and processes it in CSV format

import os, ast
import csv
import unicodedata
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
DATA_HOME = os.getenv("DATA_HOME")
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

def load_dataset(datacard_env):
    try:
        datacard = os.getenv(datacard_env)
        if datacard is None:
            raise ValueError(f"Environment variable '{datacard_env}' is not defined.")
        dataset_name = datacard.split("/")[-1]
        dataset_dir = os.path.join(DATA_HOME, dataset_name)
        if os.path.isdir(dataset_dir):
            print("[-] Dataset already exists... [SKIPPING DOWNLOAD]")
            return datacard
        print("[+] Loading dataset from HuggingFace...", end=" ")
        snapshot_download(repo_id=datacard, repo_type="dataset", local_dir=dataset_dir)

        print("[DONE]")
        print(f"\t[+] Dataset stored at {dataset_dir}/")
        return datacard

    except Exception as e:
        print("[FAILED]")
        print(f"\t[-] Failed to download dataset.")
        # print(f"\t[-] Error: {e}")
        return None

def process_dataset(datacard_env, split_name="train", is_roman=False):
    datacard = load_dataset(datacard_env)
    if datacard is None:
        print("[-] Datacard not found... [ABORTING]")
        return
    dataset_name = datacard.split("/")[-1]
    dataset_dir = os.path.join(DATA_HOME, dataset_name)
    all_dataset, clean_dataset = [], []
    for file in os.listdir(dataset_dir):
        if file.startswith(f"{split_name}_"):
            file_path = os.path.join(dataset_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                meta = 0
                senti = "neutral"
                sent, tags = [], []
                for line in f.read().split("\n"):
                    try:
                        if line.lower().strip().startswith("meta"):
                            tags, sent = [], []
                            meta = line.split("\t")[1]
                            senti = (line.split("\t")[2].strip().lower())
                        elif line.strip() != "":
                            sent.append(line.split("\t")[0].strip().lower())
                            tags.append(line.split("\t")[1].strip().lower())
                        else:
                            all_dataset.append({
                                "id": meta,
                                "sentence": sent,
                                "tags": tags,
                                "sentiment": senti
                            })

                    except Exception as e:
                        # print(f"[-] Error: {e}")
                        continue
    output_file = os.path.join(dataset_dir, f"{split_name}.csv")

    if is_roman:
        for row in all_dataset:
            sentence = " ".join(row["sentence"])
            if is_roman_text(sentence):
                clean_dataset.append(row)
        dataset_to_write = clean_dataset
    else:
        dataset_to_write = all_dataset

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "id",
                "sentence",
                "tags",
                "sentiment"
            ])
        writer.writeheader()
        writer.writerows(dataset_to_write)
    print(f"[SUCCESS] Dataset saved to {output_file}")

    if is_roman:
        print(f"\t[+] Total samples   : {len(all_dataset)}")
        print(f"\t[+] Roman samples   : {len(clean_dataset)}")
        print(f"\t[-] Removed samples : " f"{len(all_dataset) - len(clean_dataset)}")

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
    process_dataset("HG_DATACARD", "train", is_roman=True)
    process_dataset("HG_DATACARD", "dev", is_roman=True)
    process_dataset("HG_DATACARD", "Hindi", is_roman=True)
    folder = os.path.join(DATA_HOME, "SentiMix")
    conn = pd.read_csv(os.path.join(DATA_HOME, "SentiMix", "Hindi.csv"), dtype={"id": "string"})
    test = pd.read_csv(os.path.join(DATA_HOME, "SentiMix", "test_labels_hinglish.txt"), dtype={"Uid": "string"})
    test = test.rename(columns={"Uid": "id", "Sentiment": "sentiment"}) # remane columns
    lookup = conn.drop_duplicates(subset="id", keep="first").set_index("id")[["sentence", "tags"]]
    pos = test.columns.get_loc("id") + 1
    for offset, col in enumerate(["sentence", "tags"]):
        test.insert(pos + offset, col, test["id"].map(lookup[col]))
    test = test.dropna(subset=["sentence", "tags"])
    test.to_csv(os.path.join(DATA_HOME, "SentiMix", "test.csv"), index=False)
    folder = os.path.join(DATA_HOME, "sentimix")
    for filename in os.listdir(folder): # remove all files does not ends with .csv
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and not filename.lower().endswith(".csv"):
            os.remove(file_path)
    os.remove(os.path.join(folder, "Hindi.csv"))
    for filename in os.listdir(folder):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path, dtype=str)
            if "tags" in df.columns:
                df["tags"] = df["tags"].apply(convert_tags)
                df.to_csv(file_path, index=False)
                print(f"Updated: {filename}")
    replace_emt_tags(os.path.join(DATA_HOME, "SentiMix", "train.csv")) # replace some OOD tags from train
    replace_emt_tags(os.path.join(DATA_HOME, "SentiMix", "dev.csv")) # replace some OOD tags from dev
    replace_emt_tags(os.path.join(DATA_HOME, "SentiMix", "test.csv")) # replace some OOD tags from test
    
if __name__ == "__main__":
    main()