# Download model checkpoints from huggingface hub.
# ⚠️ This script will download all the files which is memory and time consuming
# ⚠️ Some of the models require additional authentication from the huggingface hub. Please check the model repo for more details.

from huggingface_hub import snapshot_download
from pathlib import Path

all_model_ids = { # base model and their repo ids on huggingface (to download)
    "mistralai"             : "mistralai/Mistral-7B-Instruct-v0.3", # 7B
    "cohereLabs_command"    : "CohereLabs/c4ai-command-a-03-2025", # 111B
    "llama_8b"              : "meta-llama/Llama-3.1-8B-Instruct", # 8B
    "cohereLabs_aya"        : "CohereLabs/aya-expanse-8b", # 8B
    "qwen"                  : "Qwen/Qwen2.5-7B-Instruct", # 7B
    "llama_70b"             : "meta-llama/Llama-3.3-70B-Instruct" # 70B
}

def download_model(model_id):
    model_name = model_id.split("/")[-1] # extract the model name from the repo id
    model_path = Path.home().joinpath('models', model_name) # path to store the checkpoints
    model_path.mkdir(parents=True, exist_ok=True) # create if path does not exist
    snapshot_download(repo_id=model_id, local_dir=model_path) # download chekpoints from remote

if __name__ == "__main__":
    print("[c] Downloading all model checkpoints may reuire a lot of time and memory. Please make sure you have enough disk space and memory before proceeding.")
    choice = input("[i] Do you want to download all the models? (y/n): ")
    if choice.lower() == 'y': # download all models 
        for model_id in all_model_ids.values(): # for bluk download all models from all_model_ids
            print(f"[+] downloading model: {model_id}")
            download_model(model_id)
    else: # for individual checkpoints download
        model_id = input("[i] Enter the model repo id (e.g., mistralai/Mistral-7B-Instruct-v0.3): ")
        print(f"[+] downloading model: {model_id}")
        download_model(model_id)
