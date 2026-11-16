"""Joint LID + sentiment - data. Loads the SentiMix Hi-En splits produced by the shared `get_sentimix.py`
(the `dataset/` symlink points to the processed CSVs of hlsa-exp2; the shared script is unchanged).

CSV columns: id, sentence (python list of lower-cased word tokens), tags (python list of word-level LID tags), sentiment
Tags are normalised to {h, e, o}; the intermediate scheme {hin, eng, o, emt} is accepted as well.
"""

import ast
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

LID_TAGS = ["h", "e", "o"]                      # hindi / english / other (scheme of get_sentimix.py)
LID2ID = {t: i for i, t in enumerate(LID_TAGS)}
ID2LID = {i: t for t, i in LID2ID.items()}
LID_IGNORE = -100                               # LID label of special tokens / padding (ignored by the losses)
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
        raise FileNotFoundError(f"{path} not found - run the shared `get_sentimix.py` first")
    df = pd.read_csv(path, dtype=str)
    df["tokens"] = df["sentence"].apply(ast.literal_eval)
    df["tags"] = df["tags"].apply(lambda s: [TAG_NORMALISE.get(t.lower(), "o") for t in ast.literal_eval(s)])
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df = df[df["sentiment"].isin(LABEL2ID) & (df["tokens"].apply(len) > 0)].copy()
    assert (df["tokens"].apply(len) == df["tags"].apply(len)).all(), f"{split}: token/tag length mismatch"
    df["label"] = df["sentiment"].map(LABEL2ID)
    return df[["id", "tokens", "tags", "sentiment", "label"]].reset_index(drop=True)


def encode_words(tokenizer, words, max_length: int):
    """Tokenise a pre-split tweet as natural text and align every sub-word to its word.

    Returns (input_ids, attention_mask, word_index) where word_index[j] is the index of the word that sub-word j
    belongs to, or None for special tokens. Uses character offsets of the space-joined text, so it works for
    SentencePiece (XLM-R, IndicBERT) and byte-level BPE (Qwen) tokenizers alike.
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


class JointDataset(Dataset):
    """One item = one tweet with sub-word LID labels (every sub-word of a word carries the word's tag) and the
    tweet sentiment. `first_mask` marks the first sub-word of each word: LID is *evaluated* there (word level),
    the LID losses use all word sub-words."""

    def __init__(self, df, tokenizer, max_length: int):
        self.rows, self.tok, self.max_length = df.to_dict("records"), tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, am, widx = encode_words(self.tok, r["tokens"], self.max_length)
        lid, first, seen = [], [], set()
        for w in widx:
            lid.append(LID_IGNORE if w is None else LID2ID[r["tags"][w]])
            first.append(int(w is not None and w not in seen))
            seen.add(w)
        return {"input_ids": ids, "attention_mask": am, "lid_labels": lid, "first_mask": first,
                "word_index": [-1 if w is None else w for w in widx], "label": int(r["label"]),
                "n_words": len(r["tokens"]), "id": r["id"]}


def make_collate(pad_id: int):
    def collate(batch):
        T = max(len(b["input_ids"]) for b in batch)
        pad = lambda key, val: torch.tensor([b[key] + [val] * (T - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
                "lid_labels": pad("lid_labels", LID_IGNORE), "first_mask": pad("first_mask", 0),
                "word_index": pad("word_index", -1), "labels": torch.tensor([b["label"] for b in batch]),
                "n_words": [b["n_words"] for b in batch], "ids": [b["id"] for b in batch]}
    return collate
