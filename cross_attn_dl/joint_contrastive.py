"""Joint contrastive LID + sentiment model - inference for the demo.

Loads a checkpoint written by `python3 main.py --mode joint_contrastive --save_model` (checkpoints/<run>.pt), rebuilds
the model around the pre-trained encoder named inside the checkpoint (the encoder and its tokenizer come from the
Hugging Face pre-trained checkpoint, the .pt file holds the weights that were trained), and exposes

    get_lid(sentence)         list of words  -> list of tags, one per word, each in {"h", "e", "o"}
    get_sentiment(sentence)   sentence (str) -> "negative" | "neutral" | "positive"

The file is self-contained: it does not import the training package `lib/`, so it can be copied anywhere together with
the .pt file. The model is loaded once, on the first call.

    python3 joint_contrastive.py "yaar movie bahut acchi thi but ending was terrible"
    python3 joint_contrastive.py --verify          # re-score the test split and compare with the scores stored in the .pt

Checkpoint and device are chosen with the environment variables JCL_CHECKPOINT (path of a .pt file) and JCL_DEVICE
("cpu", "cuda", "cuda:1"; default: cuda if available), or by calling load_model(checkpoint, device); model_info() tells
what is loaded. The XLM-R checkpoint runs on a CPU, the Qwen2.5-7B checkpoint needs a GPU.
"""

import argparse
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "xlm-roberta-base_joint_contrastive_seed42.pt"
FALLBACK_SENTIMENT = "neutral"          # returned for an empty sentence
FALLBACK_TAG = "o"                      # tag of a word that was cut off by the maximum length

_LOADED = {}                            # the active model: model, tokenizer, meta, device, path - set by load_model()
_CACHE = {}                             # (checkpoint, device) -> the same dict; switching back to a loaded model is instant


# ---------------------------------------------------------------------------------------------- model
class _CrossAttentionPooling(nn.Module):
    """A learned query attends over language-aware token states k_i = v_i = LayerNorm(h_i + W_e p_i), where p_i is
    the LID posterior of token i. Same module (and parameter names) as lib/model.CrossAttentionPooling."""

    def __init__(self, hidden: int, n_tags: int, n_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.lang_emb = nn.Linear(n_tags, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.mha = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)

    def forward(self, h, mask, lid_prob):
        kv = self.norm(h + self.lang_emb(lid_prob.to(h.dtype)))
        q = self.query.expand(h.size(0), -1, -1).to(kv.dtype)
        s, _ = self.mha(q, kv, kv, key_padding_mask=(mask == 0), need_weights=True)
        return s.squeeze(1)


class _JointModel(nn.Module):
    """encoder -> LID head on every sub-word; cross-attention pooling -> sentiment head (inference path only)."""

    def __init__(self, encoder, n_labels: int, n_tags: int, n_heads: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.lid_head = nn.Linear(hidden, n_tags)
        self.pooling = _CrossAttentionPooling(hidden, n_tags, n_heads, dropout)
        self.classifier = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask, word_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lid_logits = self.lid_head(h).float()                                  # [B,T,n_tags]
        s = self.pooling(h, word_mask, torch.softmax(lid_logits, dim=-1))
        logits = self.classifier(s.to(self.classifier.weight.dtype)).float()   # [B,n_labels]
        return logits, lid_logits


def load_model(checkpoint=None, device=None):
    """Make a .pt checkpoint the active model (default: JCL_CHECKPOINT or checkpoints/xlm-roberta-base_...pt).
    Called automatically by get_lid / get_sentiment; call it yourself to choose another checkpoint or device.
    Checkpoints that were loaded before stay in memory, so switching between them does not reload anything."""
    path = Path(checkpoint or os.environ.get("JCL_CHECKPOINT") or DEFAULT_CHECKPOINT).resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint {path} not found - create it with `python3 main.py --mode joint_contrastive --save_model`")
    device = torch.device(device or os.environ.get("JCL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (str(path), str(device))
    if key not in _CACHE:
        _CACHE[key] = _build(path, device)
    _predict.cache_clear()
    _LOADED.clear()
    _LOADED.update(_CACHE[key])
    return _LOADED["model"]


def model_info() -> dict:
    """What is loaded: checkpoint file, pre-trained model, device, and the test scores stored in the checkpoint."""
    st = _state()
    meta, test = st["meta"], st["meta"]["metrics"]["test"]
    return {"checkpoint": st["path"].name, "model_name": meta["model_name"], "mode": meta["mode"], "device": str(st["device"]),
            "best_epoch": meta["best_epoch"], "test_sentiment_macro_f1": test["sentiment"]["macro_f1"],
            "test_lid_macro_f1": test["lid"]["macro_f1"]}


def _build(path: Path, device) -> dict:
    """Rebuild the model of one checkpoint around its pre-trained encoder."""
    ckpt = torch.load(path, map_location="cpu")
    kw = ckpt["model_kwargs"]
    if not kw["use_lid"] or kw["fusion"] != "cross_attn":
        raise ValueError(f"{path.name} is a '{ckpt['mode']}' checkpoint without the LID head / cross-attention pooling")
    if ckpt["dtype"] == "bfloat16" and device.type != "cuda":
        raise RuntimeError(f"{path.name} is a bf16 checkpoint of {ckpt['model_name']}: it needs a CUDA GPU "
                           f"(about 16 GB of GPU memory), not {device}")

    name = ckpt["model_name"]                                  # the pre-trained checkpoint: tokenizer + frozen encoder
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    bf16 = ckpt["dtype"] == "bfloat16"                                      # same precision as in training
    encoder = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    encoder.config.pad_token_id = tokenizer.pad_token_id
    model = _JointModel(encoder, len(ckpt["labels"]), len(ckpt["lid_tags"]), kw["n_heads"], kw["dropout"])

    # the projection heads of the contrastive losses are only used during training
    state = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith(("lid_proj.", "sent_proj."))}
    params = dict(model.named_parameters())
    for k, v in state.items():                                 # trained weights are float32: match before loading
        if k in params and params[k].dtype != v.dtype:
            params[k].data = params[k].data.to(v.dtype)
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("encoder.")]      # frozen encoder weights are
    if missing or result.unexpected_keys:                                           # not in the .pt by design
        raise RuntimeError(f"checkpoint does not fit the model: missing={missing} unexpected={result.unexpected_keys}")
    model.to(device).eval()
    return dict(model=model, tokenizer=tokenizer, device=device, path=path, n_loaded=len(state),
                meta={k: v for k, v in ckpt.items() if k != "state_dict"})


def _state():
    if not _LOADED:
        load_model()
    return _LOADED


# ---------------------------------------------------------------------------------------------- text
def _encode_words(tokenizer, words, max_length: int):
    """Tokenise the space-joined words and align every sub-word to its word with character offsets.
    Returns (input_ids, attention_mask, word_index); word_index[j] = -1 for special tokens."""
    text, spans = "", []
    for w in words:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(w)))
        text += w
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    special = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    word_index, j = [], 0
    for tok_id, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        if e == 0 or tok_id in special:
            word_index.append(-1)
            continue
        while j < len(spans) - 1 and spans[j][1] <= s:
            j += 1
        word_index.append(j)
    return enc["input_ids"], enc["attention_mask"], word_index


def _as_words(sentence) -> list:
    """A sentence (split at white space) or an already split list of words; lower-cased like the training data."""
    words = sentence.split() if isinstance(sentence, str) else [str(w) for w in sentence]
    return [w.lower() for w in words]


@torch.no_grad()
def predict_batch(sentences) -> list:
    """sentences: list of word lists -> list of (sentiment, tags) with one tag per word."""
    st = _state()
    model, tok, meta, device = st["model"], st["tokenizer"], st["meta"], st["device"]
    encs = [_encode_words(tok, words, meta["max_length"]) for words in sentences]
    T = max(len(e[0]) for e in encs)
    pad = lambda i, val: torch.tensor([e[i] + [val] * (T - len(e[i])) for e in encs], device=device)
    ids, am, widx = pad(0, tok.pad_token_id), pad(1, 0), pad(2, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, lid_logits = model(ids, am, widx >= 0)
    sent_ids, tag_ids, widx = logits.argmax(-1).tolist(), lid_logits.argmax(-1).cpu(), widx.cpu()
    out = []
    for b, words in enumerate(sentences):
        tags, seen = [FALLBACK_TAG] * len(words), set()
        for t, w in enumerate(widx[b].tolist()):
            if w >= 0 and w not in seen:                       # a word is tagged at its first sub-word
                tags[w] = meta["lid_tags"][int(tag_ids[b, t])]
                seen.add(w)
        out.append((meta["labels"][sent_ids[b]], tags))
    return out


@lru_cache(maxsize=256)
def _predict(words: tuple):
    sentiment, tags = predict_batch([list(words)])[0]
    return sentiment, tuple(tags)


# ---------------------------------------------------------------------------------------------- public API
def get_lid(sentence: list) -> list:
    """Word-level language identification.
    sentence: list of words, e.g. ["movie", "bahut", "acchi", "thi"]
    returns:  list of tags of the same length, "h" = Hindi, "e" = English, "o" = other (punctuation, emoji, ...)."""
    words = _as_words(sentence)
    if not words:
        return []
    return list(_predict(tuple(words))[1])


def get_sentiment(sentence) -> str:
    """Tweet-level sentiment.
    sentence: the sentence as a string (a list of words is accepted too)
    returns:  "negative", "neutral" or "positive"."""
    words = _as_words(sentence)
    if not words:
        return FALLBACK_SENTIMENT
    return _predict(tuple(words))[0]


# ---------------------------------------------------------------------------------------------- command line
def _verify(test_csv: Path, batch_size: int = 64):
    """Re-score the SentiMix test split with the loaded checkpoint and compare with the scores stored in it."""
    import ast

    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support as prf

    norm = {"h": "h", "e": "e", "o": "o", "hin": "h", "eng": "e", "emt": "e"}
    df = pd.read_csv(test_csv, dtype=str)
    df["tokens"] = df["sentence"].apply(ast.literal_eval)
    df["tags"] = df["tags"].apply(lambda s: [norm.get(t.lower(), "o") for t in ast.literal_eval(s)])
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    st = _state()
    labels, lid_tags = st["meta"]["labels"], st["meta"]["lid_tags"]
    df = df[df["sentiment"].isin(labels) & (df["tokens"].apply(len) > 0)]
    rows = df.to_dict("records")
    pred_s, pred_t, gold_t = [], [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        for r, (s, tags) in zip(chunk, predict_batch([_as_words(r["tokens"]) for r in chunk])):
            pred_s.append(s)
            pred_t += tags
            gold_t += r["tags"]
    stored = st["meta"]["metrics"]["test"]
    print(f"checkpoint {st['path'].name} | device {st['device']} | {len(rows)} test tweets, {len(gold_t)} words")
    for task, gold, pred, classes in [("sentiment", [r["sentiment"] for r in rows], pred_s, labels), ("lid", gold_t, pred_t, lid_tags)]:
        p, r, f, _ = prf(gold, pred, labels=classes, average="macro", zero_division=0)
        ref = stored[task]
        print(f"  {task:9s} now    P={p:.4f} R={r:.4f} F1={f:.4f}\n  {'':9s} stored P={ref['macro_precision']:.4f} "
              f"R={ref['macro_recall']:.4f} F1={ref['macro_f1']:.4f}   (difference in F1: {f - ref['macro_f1']:+.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="joint contrastive LID + sentiment - inference")
    ap.add_argument("sentence", nargs="?", default="yaar movie bahut acchi thi but ending was terrible")
    ap.add_argument("--checkpoint", default=None, help="a .pt file written by main.py --save_model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--verify", action="store_true", help="re-score the test split and compare with the stored scores")
    ap.add_argument("--test_csv", default=str(ROOT / "dataset" / "SentiMix" / "test.csv"))
    ap.add_argument("--batch_size", type=int, default=64, help="--verify only; use the evaluation batch size of the training run "
                    "(64, or 32 for the Qwen run) to reproduce its bf16 rounding exactly")
    a = ap.parse_args()
    load_model(a.checkpoint, a.device)
    if a.verify:
        _verify(Path(a.test_csv), a.batch_size)
    else:
        words = a.sentence.split()
        print("sentence :", a.sentence)
        print("lid      :", list(zip(words, get_lid(words))))
        print("sentiment:", get_sentiment(a.sentence))
