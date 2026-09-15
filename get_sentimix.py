# this script downloads dataset from HuggingFace and processes it in CSV format

import os
import csv
import unicodedata
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
DATA_HOME = os.getenv("DATA_HOME")
if DATA_HOME is None:
    raise ValueError("DATA_HOME environment variable is not defined.")

os.makedirs(DATA_HOME, exist_ok=True)

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
        print(f"\t[-] Error: {e}")
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
                        print(f"[-] Error: {e}")
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

if __name__ == "__main__":
    process_dataset("HG_DATACARD", "train", is_roman=True)