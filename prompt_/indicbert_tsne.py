# imports
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix)
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# config
MODEL_CARD = "ai4bharat/IndicBERT-v3-1B"    
MODEL_DIR = "models/IndicBERT-v3-1B"        

TEST_CSV = "dataset/SentiMix/test.csv"
ZERO_SHOT_PROMPT = "prompts/zero_shot_lid_prompt.txt"   
ONE_SHOT_PROMPT = "prompts/one_shot_lid_prompt.txt"
N_SAMPLES = 10                            
MAX_SENTENCE_TOKENS = 100                   # keeps prompt + sentence a sensible length
SAVE_EVERY = 250                            # partial save every N sentences
LABELS = ["positive", "negative", "neutral"]

PREDICTIONS_CSV = "predictions_indicbert.csv"
METRICS_CSV = "metrics_indicbert.csv"

# IndicBERT-v3 is a bidirectional encoder trained with masked NEXT-token
# prediction (MNTP): the hidden state at position i-1 predicts the token that
# was masked at position i. It cannot write an answer, so generate() is not
# used. A <mask> is put where the label goes and the scores of the label words
# at that slot decide the sentiment.


#  data
def fix_mojibake(token: str) -> str:
    """Repair tokens like 'â€¦' or 'ðÿ˜…' (UTF-8 bytes that were read as cp1252)."""
    if token.isascii():
        return token
    try:
        return token.replace("ÿ", "Ÿ").encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return token


def detokenize(tokens: list) -> str:
    """Join the token list back into a normal-looking sentence.

    Re-attaches @mentions, #hashtags, URLs split into 'https // t . co / x',
    underscores inside handles, and punctuation, then repairs broken emoji.
    """
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


def load_test_samples(csv_path: str, n=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("id", "sentence", "sentiment"):          # 'tags' is never used or sent
        if col not in df.columns:
            raise ValueError("Missing column: {}".format(col))
    df = df.dropna(subset=["id", "sentence", "sentiment"]).copy()
    if n:
        df = df.head(n).copy()
    df["text"] = df["sentence"].apply(clean_sentence)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    unknown = set(df["sentiment"]) - set(LABELS)
    if unknown:
        print("[!] gold labels outside {}: {}".format(LABELS, unknown))
    return df.reset_index(drop=True)


#  model
def find_local_model(model_path: str):
    """Look for the model folder next to where you run from, then next to this script.

    A relative path like 'models/IndicBERT-v3-1B' only works if you start the
    script from the folder that CONTAINS 'models'. If the folder is not found,
    transformers wrongly treats the string as a Hugging Face repo name, which
    is what produced the "couldn't connect to huggingface.co" error.
    """
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        candidate = base / model_path
        has_weights = (any(candidate.glob("*.safetensors"))
                       or any(candidate.glob("*.bin")))
        # config.json alone is not enough: an interrupted download leaves it
        # behind without the weights, and that must be treated as "missing".
        if (candidate / "config.json").exists() and has_weights:
            return candidate
    return None


def get_model(model_path: str = MODEL_DIR):
    """Load the local IndicBERT-v3-1B; download it into model_path first if missing."""
    local = find_local_model(model_path)
    if local is None:
        print("[!] no model found at '{}' (looked in {} and {})".format(
            model_path, Path.cwd(), Path(__file__).resolve().parent))
        print("[+] downloading {} into '{}' ...".format(MODEL_CARD, model_path))
        try:
            snapshot_download(MODEL_CARD, local_dir=model_path)
        except Exception as err:
            raise RuntimeError(
                "Download failed ({}).\nThis repo is gated. 1) Open https://huggingface.co/{} "
                "while logged in and accept the access terms. 2) Log in on this machine "
                "with `hf auth login` (older versions: `huggingface-cli login`) or set "
                "the HF_TOKEN environment variable. Or copy the model folder to '{}'."
                .format(err, MODEL_CARD, model_path))
        local = Path(model_path)
        if not ((local / "config.json").exists() and (any(local.glob("*.safetensors"))
                                                       or any(local.glob("*.bin")))):
            raise RuntimeError("download finished but {} has no weights".format(local))
    print("[+] loading model from", local)

    tokenizer = AutoTokenizer.from_pretrained(str(local), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(local), trust_remote_code=True, local_files_only=True)
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
        raise ValueError("tokenizer has no {!r} token; tell me what the mask token "
                         "of this model is".format(mask_str))
    return mask_str, mask_id


def label_id_groups(tokenizer) -> list:
    """For each label, the vocabulary ids that stand for it.

    Whole-word tokens ("positive" is one piece) are preferred; only if a label
    has none do we fall back to the first sub-word. An id claimed by two labels
    cannot tell them apart, so it is dropped.
    """
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
        raise ValueError("a label has no usable token; pick other label words: {}"
                         .format(groups))
    print("label tokens:", {l: tokenizer.convert_ids_to_tokens(g)
                            for l, g in zip(LABELS, groups)})
    return groups


#  prompting
def build_prompt(template: str, sentence: str, tokenizer, mask_str: str) -> str:
    """Fill {text}, then turn the final 'Output:' into a slot for the model.

    The prompt is cut at the last 'Output:' (dropping the trailing instruction
    in the zero-shot file, which an encoder cannot act on) and <mask> is put
    where the label goes.
    """
    ids = tokenizer.encode(sentence, add_special_tokens=False)[:MAX_SENTENCE_TOKENS]
    sentence = tokenizer.decode(ids)
    rendered = template.replace("{text}", sentence)
    if "Output:" not in rendered:
        raise ValueError("prompt needs an 'Output:' line")
    return rendered.rsplit("Output:", 1)[0] + "Output: " + mask_str


def encode_prompt(tokenizer, prompt: str, mask_id: int, device):
    enc = tokenizer(prompt, return_tensors="pt")
    positions = (enc["input_ids"][0] == mask_id).nonzero().flatten()
    if len(positions) != 1:
        raise ValueError("expected exactly one mask token, found {}".format(len(positions)))
    return {k: v.to(device) for k, v in enc.items()}, int(positions[0])


@torch.inference_mode()
def mask_logits(model, enc, pos: int) -> np.ndarray:
    """Vocabulary scores for the masked token.

    MNTP: the hidden state one position BEFORE the mask (pos - 1) predicts it.
    Only the logits from pos - 1 onwards are requested, because the vocabulary
    is huge and the full (length x vocab) matrix is wasted work.
    """
    total = enc["input_ids"].shape[1]
    keep = total - (pos - 1)
    try:
        logits = model(**enc, use_cache=False, logits_to_keep=keep).logits
    except TypeError:                       # custom code without that option
        logits = model(**enc, use_cache=False).logits
    index = 0 if logits.shape[1] == keep else pos - 1   # trimmed, or full length
    return logits[0, index].float().cpu().numpy()


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


def predict_label(model, tokenizer, prompt: str, label_groups: list,
                  mask_id: int, device) -> str:
    enc, pos = encode_prompt(tokenizer, prompt, mask_id, device)
    logits = mask_logits(model, enc, pos)
    scores = [logsumexp(logits[ids]) for ids in label_groups]
    return LABELS[int(np.argmax(scores))]


#  scoring
def score(y_true: list, y_pred: list) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    result = {"accuracy": accuracy_score(y_true, y_pred), "per_class": {}}
    for i, label in enumerate(LABELS):
        result["per_class"][label] = (p[i], r[i], f[i], int(support[i]))
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average=avg, zero_division=0)
        result[avg] = (pa, ra, fa)
    return result


def print_report(name: str, res: dict, y_true: list, y_pred: list) -> None:
    print("\n=== {} ===".format(name))
    print("{:<10}{:>10}{:>10}{:>10}{:>10}".format("class", "P", "R", "F1", "support"))
    for label, (p, r, f, s) in res["per_class"].items():
        print("{:<10}{:>10.3f}{:>10.3f}{:>10.3f}{:>10d}".format(label, p, r, f, s))
    for avg in ("macro", "weighted"):
        p, r, f = res[avg]
        print("{:<10}{:>10.3f}{:>10.3f}{:>10.3f}".format(avg, p, r, f))
    print("accuracy  {:.3f}".format(res["accuracy"]))
    print("confusion matrix (rows = true, cols = predicted, order {}):".format(LABELS))
    print(confusion_matrix(y_true, y_pred, labels=LABELS))


def metrics_rows(setting: str, res: dict) -> list:
    rows = [dict(setting=setting, cls=label, precision=p, recall=r, f1=f, support=s)
            for label, (p, r, f, s) in res["per_class"].items()]
    for avg in ("macro", "weighted"):
        p, r, f = res[avg]
        rows.append(dict(setting=setting, cls=avg, precision=p, recall=r, f1=f,
                         support=sum(v[3] for v in res["per_class"].values())))
    rows.append(dict(setting=setting, cls="accuracy", precision=res["accuracy"],
                     recall=res["accuracy"], f1=res["accuracy"],
                     support=sum(v[3] for v in res["per_class"].values())))
    return rows


def run_setting(name, template, df, predict_fn) -> list:
    preds = []
    for i, sentence in enumerate(tqdm(df["text"], desc=name)):
        preds.append(predict_fn(template, sentence))
        if (i + 1) % SAVE_EVERY == 0:                    # partial save, just in case
            part = df.iloc[:len(preds)][["id", "text", "sentiment"]].copy()
            part[name] = preds
            part.to_csv("partial_indicbert_{}.csv".format(name), index=False)
    res = score(df["sentiment"].tolist(), preds)
    print_report(name, res, df["sentiment"].tolist(), preds)
    return preds


# driving code
if __name__ == "__main__":
    df = load_test_samples(TEST_CSV, N_SAMPLES)
    print("sentences:", len(df))
    print(df["sentiment"].value_counts().to_string())

    model, tokenizer = get_model(MODEL_DIR)
    device = pick_device()
    print("device:", device)
    model.to(device)
    mask_str, mask_id = get_mask(tokenizer)
    label_groups = label_id_groups(tokenizer)

    def predict(template, sentence):
        prompt = build_prompt(template, sentence, tokenizer, mask_str)
        return predict_label(model, tokenizer, prompt, label_groups, mask_id, device)

    zero_template = Path(ZERO_SHOT_PROMPT).read_text(encoding="utf-8")
    one_template = Path(ONE_SHOT_PROMPT).read_text(encoding="utf-8")

    df["zero_shot"] = run_setting("zero_shot", zero_template, df, predict)
    df["one_shot"] = run_setting("one_shot", one_template, df, predict)

    df[["id", "text", "sentiment", "zero_shot", "one_shot"]].rename(
        columns={"text": "sentence", "sentiment": "actual_sentiment"}).to_csv(
        PREDICTIONS_CSV, index=False)
    print("\nwrote", PREDICTIONS_CSV)

    y_true = df["sentiment"].tolist()
    rows = (metrics_rows("zero_shot", score(y_true, df["zero_shot"].tolist()))
            + metrics_rows("one_shot", score(y_true, df["one_shot"].tolist())))
    pd.DataFrame(rows).rename(columns={"cls": "class"}).to_csv(METRICS_CSV, index=False)
    print("wrote", METRICS_CSV)