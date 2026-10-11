"""Load the SentiMix Hi-En splits produced by the shared `get_sentimix.py` (unchanged).

CSV columns written by the shared script:
    id, sentence (python list of lower-cased word tokens), tags (python list of word-level LID tags), sentiment

Tags are normalised to the shared script's final scheme {h, e, o}. The intermediate scheme
{hin, eng, o, emt} is accepted as well, in case the last post-processing step of `get_sentimix.py`
did not run (it looks up `dataset/sentimix` in lower case, which fails on case-sensitive file systems).
"""

import ast
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

LID_TAGS = ["h", "e", "o"]                      # hindi / english / other (scheme of get_sentimix.py)
LID2ID = {t: i for i, t in enumerate(LID_TAGS)}
LID_SPECIAL = len(LID_TAGS)                     # LID index for special tokens (CLS/SEP/BOS) and padding
# `emt` (emoticon) is mapped to `e` because that is what get_sentimix.replace_emt_tags does.
TAG_NORMALISE = {"h": "h", "e": "e", "o": "o", "hin": "h", "eng": "e", "emt": "e"}


def dataset_dir() -> Path:
    """`<DATA_HOME>/<datacard name>` as configured in .env (see .env.example)."""
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    data_home, datacard = os.getenv("DATA_HOME"), os.getenv("HG_DATACARD")
    if not data_home or not datacard:
        raise ValueError("DATA_HOME and HG_DATACARD must be defined in .env (copy .env.example)")
    base = Path(data_home)
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / datacard.split("/")[-1]


def load_split(split: str) -> pd.DataFrame:
    """Columns: id (str), tokens (list[str]), tags (list[str] in {h,e,o}), sentiment (str), label (int)."""
    path = dataset_dir() / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python3 get_sentimix.py` first")
    df = pd.read_csv(path, dtype=str)
    df["tokens"] = df["sentence"].apply(ast.literal_eval)
    df["tags"] = df["tags"].apply(lambda s: [TAG_NORMALISE.get(t.lower(), "o") for t in ast.literal_eval(s)])
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df = df[df["sentiment"].isin(LABEL2ID) & (df["tokens"].apply(len) > 0)].copy()
    assert (df["tokens"].apply(len) == df["tags"].apply(len)).all(), f"{split}: token/tag length mismatch"
    df["label"] = df["sentiment"].map(LABEL2ID)
    return df[["id", "tokens", "tags", "sentiment", "label"]].reset_index(drop=True)


def load_predicted_tags(path) -> dict:
    """id -> list[str] of word-level LID tags from a CSV with columns `id, pred_tags`."""
    df = pd.read_csv(path, dtype=str)
    return {r["id"]: [TAG_NORMALISE.get(t.lower(), "o") for t in ast.literal_eval(r["pred_tags"])]
            for _, r in df.iterrows()}


def encode_words(tokenizer, words, max_length: int):
    """Tokenise a pre-split tweet as natural text and align every sub-word to its word.

    Returns (input_ids, attention_mask, word_index) where word_index[j] is the index of the word that
    sub-word j belongs to, or None for special tokens. Works for WordPiece (mBERT), SentencePiece
    (XLM-R) and byte-level BPE (Qwen) tokenizers alike because the alignment uses character offsets
    of the space-joined text, so GPT-style tokenizers keep their usual leading-space tokens.
    """
    text, spans = "", []
    for w in words:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(w)))
        text += w
    try:
        enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
    except (NotImplementedError, TypeError):            # slow tokenizer: fall back to word_ids()
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
        return enc["input_ids"], enc["attention_mask"], enc.word_ids()
    special_ids = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}   # <unk> still covers a word
    word_index, j = [], 0
    for tok_id, (s, e) in zip(enc["input_ids"], offsets):
        if e == 0 or tok_id in special_ids:
            word_index.append(None)
            continue
        while j < len(spans) - 1 and spans[j][1] <= s:   # advance to the word containing offset s
            j += 1
        word_index.append(j)
    return enc["input_ids"], enc["attention_mask"], word_index


def lid_ids_for_subwords(word_index, tags) -> list:
    """Map the per-word LID tags onto sub-word positions (special tokens -> LID_SPECIAL)."""
    return [LID_SPECIAL if w is None else LID2ID[tags[w]] for w in word_index]
