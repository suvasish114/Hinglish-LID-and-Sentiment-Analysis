# Zero-shot / one-shot sentiment prompting with xlm-roberta-base (baseline).
#
# XLM-R is an encoder: it cannot write an answer. Each prompt therefore ends in
# 'Output: <mask>' and the model's scores for the label words at the <mask>
# decide the sentiment.
#
# Output (in results/):
#   xlmr_zero_shot.csv   columns: id, sentiment
#   xlmr_one_shot.csv    columns: id, sentiment
#
# Run from anywhere:  python3 prompt_/roberta_base_tsne.py [--n_samples N]

# imports
import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForMaskedLM

# config (paths are relative to the repo root, not to where you run from)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CARD = "FacebookAI/xlm-roberta-base"
MODEL_DIR = REPO_ROOT / "models" / "xlm-roberta-base-mlm"
ZERO_SHOT_PROMPT = REPO_ROOT / "prompts" / "zero_shot_sentiment_prompt.txt"
ONE_SHOT_PROMPT = REPO_ROOT / "prompts" / "one_shot_sentiment_prompt.txt"
RESULTS_DIR = REPO_ROOT / "results"
TAG = "xlmr"                                # output file prefix

LABELS = ["positive", "negative", "neutral"]
MAX_SENTENCE_TOKENS = 100                   # keeps prompt + sentence a sensible length
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


def download_model(model_dir: Path) -> None:
    # The repo also holds TensorFlow / Flax / ONNX copies of the same weights,
    # which PyTorch never reads, so they are skipped. The .bin duplicate is
    # skipped too unless the repo has no .safetensors file.
    skip = ["*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.ot"]
    snapshot_download(MODEL_CARD, local_dir=str(model_dir), ignore_patterns=skip + ["*.bin"])
    if not any(model_dir.glob("*.safetensors")):
        snapshot_download(MODEL_CARD, local_dir=str(model_dir), ignore_patterns=skip)


def get_model(model_dir: Path = MODEL_DIR):
    # Load xlm-roberta-base WITH its masked-LM head (download once, then reuse).
    # config.json alone is not enough: an interrupted download leaves it behind
    # without the weights, and that must count as "missing".
    if not ((model_dir / "config.json").exists() and has_weights(model_dir)):
        print("[+] downloading {} into '{}' ...".format(MODEL_CARD, model_dir))
        download_model(model_dir)
        if not has_weights(model_dir):
            raise RuntimeError("download finished but {} has no weights".format(model_dir))
    print("[+] loading model from", model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model, info = AutoModelForMaskedLM.from_pretrained(
        str(model_dir), local_files_only=True, output_loading_info=True)
    if "lm_head.dense.weight" in info["missing_keys"]:
        raise RuntimeError("The LM head was not in the checkpoint, so it would be "
                           "random (was this folder saved with AutoModel?). "
                           "Delete {} and run again.".format(model_dir))
    model.eval()
    return model, tokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def label_id_groups(tokenizer) -> list:
    # For each label, the vocabulary ids that stand for it (same method as the
    # IndicBERT file, so both models are scored the same way).
    # Whole-word tokens ("positive" is one piece) are preferred; only if a label
    # has none do we fall back to the first sub-word. An id claimed by two labels
    # cannot tell them apart, so it is dropped.
    whole, first = [], []
    for label in LABELS:
        w, f = set(), set()
        for variant in (" " + label, label, " " + label.capitalize(), label.capitalize()):
            pieces = [t for t in tokenizer.encode(variant, add_special_tokens=False)
                      if tokenizer.convert_ids_to_tokens(t) not in ("▁", "")]
            if len(pieces) == 1:
                w.add(pieces[0])
            elif pieces:
                f.add(pieces[0])
        whole.append(w)
        first.append(f)
    groups = [sorted(w if w else f) for w, f in zip(whole, first)]

    counts = Counter(i for g in groups for i in g)
    groups = [[i for i in g if counts[i] == 1] for g in groups]
    if any(not g for g in groups):
        raise ValueError("a label has no usable token; pick other label words: {}".format(groups))
    print("label tokens:", {l: tokenizer.convert_ids_to_tokens(g) for l, g in zip(LABELS, groups)})
    return groups


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


# prompting
def build_prompt(template: str, sentence: str, tokenizer) -> str:
    # Fill {text}, then turn the final 'Output:' into a slot for the model:
    # everything after the last 'Output:' is dropped and <mask> is put where
    # the label goes. In the one-shot prompt the example's 'Output: positive'
    # comes earlier, so it is kept.
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    if len(ids) > MAX_SENTENCE_TOKENS:          # decode only when cut, to keep the text as-is
        sentence = tokenizer.decode(ids[:MAX_SENTENCE_TOKENS])
    rendered = template.replace("{text}", sentence)
    if "Output:" not in rendered:
        raise ValueError("prompt needs an 'Output:' line")
    return rendered.rsplit("Output:", 1)[0] + "Output: " + tokenizer.mask_token


@torch.inference_mode()
def predict_label(model, tokenizer, prompt: str, label_groups: list, device) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    mask_pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero().flatten()
    if len(mask_pos) != 1:
        raise ValueError("expected exactly one <mask>, found {} (prompt too long?)".format(len(mask_pos)))
    enc = {k: v.to(device) for k, v in enc.items()}
    # standard MLM: the hidden state AT the mask predicts the masked token
    logits = model(**enc).logits[0, int(mask_pos[0])].float().cpu().numpy()
    scores = [logsumexp(logits[ids]) for ids in label_groups]
    return LABELS[int(np.argmax(scores))]


# running
def run_setting(name: str, template: str, df: pd.DataFrame, predict_fn) -> None:
    out_csv = RESULTS_DIR / "{}_{}.csv".format(TAG, name)
    preds = []
    for i, sentence in enumerate(tqdm(df["text"], desc=name)):
        preds.append(predict_fn(template, sentence))
        if (i + 1) % SAVE_EVERY == 0:                    # partial save, just in case
            pd.DataFrame({"id": df["id"][:len(preds)], "sentiment": preds}).to_csv(out_csv, index=False)
    pd.DataFrame({"id": df["id"], "sentiment": preds}).to_csv(out_csv, index=False)
    print("wrote", out_csv)


# driving code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XLM-R zero/one-shot sentiment prompting")
    parser.add_argument("--test_csv", type=Path, default=None, help="default: dataset/SentiMix/test.csv")
    parser.add_argument("--n_samples", type=int, default=None, help="first N sentences only (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_test_samples(args.test_csv or find_test_csv(), args.n_samples)
    print("sentences:", len(df))

    model, tokenizer = get_model(MODEL_DIR)
    device = pick_device()
    print("device:", device)
    model.to(device)
    label_groups = label_id_groups(tokenizer)

    def predict(template, sentence):
        prompt = build_prompt(template, sentence, tokenizer)
        return predict_label(model, tokenizer, prompt, label_groups, device)

    run_setting("zero_shot", ZERO_SHOT_PROMPT.read_text(encoding="utf-8"), df, predict)
    run_setting("one_shot", ONE_SHOT_PROMPT.read_text(encoding="utf-8"), df, predict)
