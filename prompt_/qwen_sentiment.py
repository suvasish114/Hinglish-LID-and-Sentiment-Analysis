# Zero-shot / one-shot sentiment prompting with Qwen2.5-7B-Instruct.
#
# Qwen/Qwen2.5-7B-Instruct is a causal (decoder-only) chat model.
#
# The model reads the same prompts as the encoder baselines and writes its
# answer; the first of positive / negative / neutral in the reply is the
# prediction (greedy decoding, so runs are repeatable).
#
# Output (in results/):
#   qwen_zero_shot.csv   columns: id, sentiment
#   qwen_one_shot.csv    columns: id, sentiment
#   raw/qwen_<setting>_raw.csv   the model's full replies, for checking the parsing
#
# Needs a GPU (~16 GB in bf16). Run from anywhere:
#   python3 prompt_/qwen_sentiment.py [--n_samples N]

# imports
import argparse
import ast
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# config (paths are relative to the repo root, not to where you run from)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CARD = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct"
ZERO_SHOT_PROMPT = REPO_ROOT / "prompts" / "zero_shot_sentiment_prompt.txt"
ONE_SHOT_PROMPT = REPO_ROOT / "prompts" / "one_shot_sentiment_prompt.txt"
RESULTS_DIR = REPO_ROOT / "results"
TAG = "qwen"                               # output file prefix

LABELS = ["positive", "negative", "neutral"]
FALLBACK_LABEL = "neutral"                  # used when the reply names no label
MAX_NEW_TOKENS = 10                         # the answer is a single word
SAVE_EVERY = 250                            # partial save every N sentences


# data
def find_test_csv() -> Path:
    # accept dataset/SentiMix or dataset/Sentimix (Linux paths are case-sensitive)
    for folder in ("SentiMix", "Sentimix"):
        path = REPO_ROOT / "dataset" / folder / "test.csv"
        if path.exists():
            return path
    raise FileNotFoundError("test.csv not found in {}/dataset/SentiMix".format(REPO_ROOT))


def fix_mojibake(token: str) -> str:
    # Repair tokens like 'â€¦' or 'ðÿ˜…' (UTF-8 bytes that were read as cp1252).
    # The 'ÿ' case is an emoji whose 'Ÿ' got lower-cased by the dataset's
    # preprocessing, so it is restored before decoding. Tokens that cannot be
    # repaired are returned unchanged.
    if token.isascii():
        return token
    try:
        return token.replace("ÿ", "Ÿ").encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return token


def detokenize(tokens: list) -> str:
    # Join the token list back into a normal-looking sentence.
    # Re-attaches @mentions, #hashtags, URLs split into 'https // t . co / x',
    # underscores inside handles, and punctuation, then repairs broken emoji.
    toks = [fix_mojibake(str(t)) for t in tokens]
    out, i, n = [], 0, len(toks)
    attach_next = False                 # glue the next token to the previous one
    no_space_before = {".", ",", "!", "?", ";", ":", ")", "]", "%", "…"}
    glue_after = {"@", "#", "(", "["}

    while i < n:
        tok = toks[i]

        # URL: https // t . co / abc  ->  https://t.co/abc
        if tok in ("http", "https") and i + 1 < n and toks[i + 1] == "//":
            url, i, expect_word = tok + "://", i + 2, True
            while i < n:
                if toks[i] in {".", "/", "-"}:
                    url += toks[i]
                    expect_word = True
                elif expect_word:
                    url += toks[i]
                    expect_word = False
                else:
                    break
                i += 1
            out.append((url, attach_next))
            attach_next = False
            continue

        glue = attach_next or tok in no_space_before or tok == "_"
        if out and out[-1][0].endswith("_"):
            glue = True                 # keep @user_name together
        out.append((tok, glue))
        attach_next = tok in glue_after or tok == "_"
        i += 1

    text = ""
    for piece, glue in out:
        text += piece if (glue or not text) else " " + piece
    return text.strip()


def clean_sentence(sentence) -> str:
    try:
        words = ast.literal_eval(sentence)
        if isinstance(words, list):
            return detokenize(words)
    except (ValueError, SyntaxError, TypeError):
        pass
    return str(sentence).strip()


def load_test_samples(csv_path: Path, n=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("id", "sentence"):
        if col not in df.columns:
            raise ValueError("Missing column: {}".format(col))
    df = df.dropna(subset=["id", "sentence"]).copy()
    if n:
        df = df.head(n).copy()
    df["text"] = df["sentence"].apply(clean_sentence)
    return df.reset_index(drop=True)


# model
def has_weights(folder: Path) -> bool:
    return any(folder.glob("*.safetensors")) or any(folder.glob("*.bin"))


def ensure_model(model_dir: Path = MODEL_DIR) -> Path:
    # Download once into models/, then reuse. config.json alone is not enough:
    # an interrupted download leaves it behind without the weights.
    if not ((model_dir / "config.json").exists() and has_weights(model_dir)):
        print("[+] downloading {} into '{}' ...".format(MODEL_CARD, model_dir))
        snapshot_download(MODEL_CARD, local_dir=str(model_dir),
                          ignore_patterns=["*.gguf", "*.onnx", "onnx/*", "original/*"])
        if not has_weights(model_dir):
            raise RuntimeError("download finished but {} has no weights".format(model_dir))
    print("[+] loading model from", model_dir)
    return model_dir


def get_model(model_dir: Path = MODEL_DIR):
    local = ensure_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(local), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(local), local_files_only=True, torch_dtype="auto", device_map="auto")
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_reply(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                            temperature=None, top_p=None, top_k=None,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]          # drop the prompt
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# prompting
def build_prompt(template: str, sentence: str) -> str:
    # The whole prompt is kept (a generative model can follow the instructions
    # and the trailing 'Output:'), only {text} is filled in.
    return template.replace("{text}", sentence)


LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b")


def parse_label(reply: str):
    # First label word in the reply, or None if the reply names none.
    match = LABEL_RE.search(reply.lower())
    return match.group(1) if match else None


# running
def run_setting(name: str, template: str, df: pd.DataFrame, generate_fn) -> None:
    out_csv = RESULTS_DIR / "{}_{}.csv".format(TAG, name)
    raw_csv = RESULTS_DIR / "raw" / "{}_{}_raw.csv".format(TAG, name)
    preds, replies, unparsed = [], [], 0
    for i, sentence in enumerate(tqdm(df["text"], desc=name)):
        reply = generate_fn(build_prompt(template, sentence))
        label = parse_label(reply)
        if label is None:
            unparsed += 1
            label = FALLBACK_LABEL
        preds.append(label)
        replies.append(reply)
        if (i + 1) % SAVE_EVERY == 0:                    # partial save, just in case
            pd.DataFrame({"id": df["id"][:len(preds)], "sentiment": preds}).to_csv(out_csv, index=False)
    pd.DataFrame({"id": df["id"], "sentiment": preds}).to_csv(out_csv, index=False)
    pd.DataFrame({"id": df["id"], "reply": replies}).to_csv(raw_csv, index=False)
    print("wrote", out_csv)
    if unparsed:
        print("[!] {} of {} replies named no label; set to '{}' (see {})".format(
            unparsed, len(preds), FALLBACK_LABEL, raw_csv))


# driving code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct zero/one-shot sentiment prompting")
    parser.add_argument("--test_csv", type=Path, default=None, help="default: dataset/SentiMix/test.csv")
    parser.add_argument("--n_samples", type=int, default=None, help="first N sentences only (default: all)")
    args = parser.parse_args()

    (RESULTS_DIR / "raw").mkdir(parents=True, exist_ok=True)
    df = load_test_samples(args.test_csv or find_test_csv(), args.n_samples)
    print("sentences:", len(df))

    model, tokenizer = get_model(MODEL_DIR)
    print("device:", model.device)

    def generate(prompt):
        return generate_reply(model, tokenizer, prompt)

    run_setting("zero_shot", ZERO_SHOT_PROMPT.read_text(encoding="utf-8"), df, generate)
    run_setting("one_shot", ONE_SHOT_PROMPT.read_text(encoding="utf-8"), df, generate)
