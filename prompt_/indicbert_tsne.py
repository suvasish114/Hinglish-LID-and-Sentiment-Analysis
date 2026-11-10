# Zero-shot / one-shot sentiment prompting with IndicBERT-v3-1B (baseline).
#
# IndicBERT-v3 is a bidirectional encoder trained with masked NEXT-token
# prediction (MNTP): the hidden state at position i-1 predicts the token that
# was masked at position i. It cannot write an answer, so generate() is not
# used. Each prompt ends in 'Output: <mask>' and the scores of the label words
# at that slot decide the sentiment.
#
# Output (in results/):
#   indicbert_zero_shot.csv   columns: id, sentiment
#   indicbert_one_shot.csv    columns: id, sentiment
#
# Run from anywhere:  python3 prompt_/indicbert_tsne.py [--n_samples N]

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
from transformers import AutoTokenizer, AutoModelForCausalLM

# config (paths are relative to the repo root, not to where you run from)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CARD = "ai4bharat/IndicBERT-v3-1B"
MODEL_DIR = REPO_ROOT / "models" / "IndicBERT-v3-1B"
ZERO_SHOT_PROMPT = REPO_ROOT / "prompts" / "zero_shot_sentiment_prompt.txt"
ONE_SHOT_PROMPT = REPO_ROOT / "prompts" / "one_shot_sentiment_prompt.txt"
RESULTS_DIR = REPO_ROOT / "results"
TAG = "indicbert"                           # output file prefix

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


def get_model(model_dir: Path = MODEL_DIR):
    # Load the local IndicBERT-v3-1B; download it into model_dir first if missing.
    # config.json alone is not enough: an interrupted download leaves it behind
    # without the weights, and that must count as "missing".
    if not ((model_dir / "config.json").exists() and has_weights(model_dir)):
        print("[+] downloading {} into '{}' ...".format(MODEL_CARD, model_dir))
        try:
            snapshot_download(MODEL_CARD, local_dir=str(model_dir))
        except Exception as err:
            raise RuntimeError(
                "Download failed ({}).\nThis repo is gated. 1) Open https://huggingface.co/{} "
                "while logged in and accept the access terms. 2) Log in on this machine "
                "with `hf auth login` (older versions: `huggingface-cli login`) or set "
                "the HF_TOKEN environment variable. Or copy the model folder to '{}'."
                .format(err, MODEL_CARD, model_dir))
        if not ((model_dir / "config.json").exists() and has_weights(model_dir)):
            raise RuntimeError("download finished but {} has no weights".format(model_dir))
    print("[+] loading model from", model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    model = model.float()       # full precision on every transformers version
    model.eval()
    return model, tokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_mask(tokenizer) -> tuple:
    mask_str = tokenizer.mask_token or "<mask>"
    mask_id = tokenizer.convert_tokens_to_ids(mask_str)
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        raise ValueError("tokenizer has no {!r} token".format(mask_str))
    return mask_str, mask_id


def label_id_groups(tokenizer) -> list:
    # For each label, the vocabulary ids that stand for it (same method as the
    # XLM-R file, so both models are scored the same way).
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
def build_prompt(template: str, sentence: str, tokenizer, mask_str: str) -> str:
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
    return rendered.rsplit("Output:", 1)[0] + "Output: " + mask_str


@torch.inference_mode()
def mask_logits(model, enc, pos: int) -> np.ndarray:
    # Vocabulary scores for the masked token.
    # MNTP: the hidden state one position BEFORE the mask (pos - 1) predicts it.
    # Only the logits from pos - 1 onwards are requested, because the vocabulary
    # is huge and the full (length x vocab) matrix is wasted work.
    total = enc["input_ids"].shape[1]
    keep = total - (pos - 1)
    try:
        logits = model(**enc, use_cache=False, logits_to_keep=keep).logits
    except TypeError:                       # custom code without that option
        logits = model(**enc, use_cache=False).logits
    index = 0 if logits.shape[1] == keep else pos - 1   # trimmed, or full length
    return logits[0, index].float().cpu().numpy()


def predict_label(model, tokenizer, prompt: str, label_groups: list, mask_id: int, device) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    positions = (enc["input_ids"][0] == mask_id).nonzero().flatten()
    if len(positions) != 1:
        raise ValueError("expected exactly one mask token, found {}".format(len(positions)))
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = mask_logits(model, enc, int(positions[0]))
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
    parser = argparse.ArgumentParser(description="IndicBERT zero/one-shot sentiment prompting")
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
    mask_str, mask_id = get_mask(tokenizer)
    label_groups = label_id_groups(tokenizer)

    def predict(template, sentence):
        prompt = build_prompt(template, sentence, tokenizer, mask_str)
        return predict_label(model, tokenizer, prompt, label_groups, mask_id, device)

    run_setting("zero_shot", ZERO_SHOT_PROMPT.read_text(encoding="utf-8"), df, predict)
    run_setting("one_shot", ONE_SHOT_PROMPT.read_text(encoding="utf-8"), df, predict)
