"""Basic baseline: zero-shot / one-shot sentiment prompting of an instruction-tuned chat model, no training.

Same approach as the group's exp1 prompting baseline (prompt_/qwen_sentiment.py, gemma4_sentiment.py in the group repo):
the tweet is put into the prompt template (prompts/<setting>_sentiment_prompt.txt), the chat model writes a short answer
with greedy decoding, and the first of positive / negative / neutral in the answer is the prediction (neutral when the
answer names no label). Only the test split is read; no gold label is used except for scoring.

    python3 prompt_baseline.py --model_name Qwen/Qwen2.5-7B-Instruct                  # zero-shot + one-shot
    python3 prompt_baseline.py --model_name google/gemma-4-E4B-it                     # needs transformers >= 5.5
    python3 prompt_baseline.py --model_name Qwen/Qwen2.5-7B-Instruct --n_samples 64   # smoke test

Outputs in --results_dir: <tag>_prompt_<setting>.csv (id, predictions), <tag>_prompt_<setting>_raw.csv (the full
replies), <tag>_prompt_<setting>_metrics.json (macro P / R / F1 + per class, readable by evaluate.py), ground.csv.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.data import LABELS, load_split  # noqa: E402
from lib.train import prf_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent
SETTINGS = ["zero_shot", "one_shot"]
FALLBACK = "neutral"
LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b")


# ---------------------------------------------------------------------------------------------- text
def fix_mojibake(tok: str) -> str:
    """Repair UTF-8 bytes that were read as cp1252 (emoji / '…' in the SentiMix CSV). The dataset lower-cased the
    tokens, which turned the 'Ÿ' byte of many emoji into 'ÿ'; it is restored before decoding."""
    if tok.isascii():
        return tok
    try:
        return tok.replace("ÿ", "Ÿ").encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return tok


NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "%", "…"}
GLUE_AFTER = {"@", "#", "(", "["}


def detokenize(tokens) -> str:
    """Join the word tokens back into a sentence: re-attach punctuation, @mentions, #hashtags and URLs split as
    'https // t . co / x'."""
    toks = [fix_mojibake(str(t)) for t in tokens]
    pieces, i, glue_next = [], 0, False
    while i < len(toks):
        t = toks[i]
        if t in ("http", "https") and i + 1 < len(toks) and toks[i + 1] == "//":     # URL
            url, i = t + "://", i + 2
            while i < len(toks) and (toks[i] in {".", "/", "-"} or (i > 0 and toks[i - 1] in {".", "/", "-", "//"})):
                url += toks[i]
                i += 1
            pieces.append((url, glue_next))
            glue_next = False
            continue
        glue = glue_next or t in NO_SPACE_BEFORE or t == "_" or (pieces and pieces[-1][0].endswith("_"))
        pieces.append((t, glue))
        glue_next = t in GLUE_AFTER or t == "_"
        i += 1
    text = ""
    for p, glue in pieces:
        text += p if (glue or not text) else " " + p
    return text.strip()


def parse_label(reply: str):
    m = LABEL_RE.search(reply.lower())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------------------------- model
def load_model(name: str, dtype):
    """Qwen: tokenizer + causal LM. Gemma 4 (multimodal): processor + the multimodal LM class of transformers 5."""
    import transformers
    kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if "gemma-4" in name.lower():
        from transformers import AutoProcessor
        cls = getattr(transformers, "AutoModelForMultimodalLM", None) or getattr(transformers, "AutoModelForImageTextToText")
        proc = AutoProcessor.from_pretrained(name)
        model = cls.from_pretrained(name, **kwargs)
        return model, proc.tokenizer, proc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    return model, tok, None


def chat_text(prompt: str, tok, proc) -> str:
    """Render one user turn with the model's chat template (as text, so that a batch can be padded on the left)."""
    if proc is not None:                                        # Gemma 4: content list, no thinking block
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    msgs = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.inference_mode()
def generate_batch(model, tok, texts, max_new_tokens: int):
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(r, skip_special_tokens=True).strip() for r in new]


# ---------------------------------------------------------------------------------------------- run
def main():
    p = argparse.ArgumentParser(description="zero-/one-shot sentiment prompting baseline on SentiMix")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--settings", nargs="+", default=SETTINGS, choices=SETTINGS)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=10)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--n_samples", type=int, default=None, help="first N tweets only (smoke test)")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model_name.rstrip("/").split("/")[-1]
    df = load_split(args.split)
    if args.n_samples:
        df = df.head(args.n_samples).copy()
    df["text"] = df["tokens"].apply(detokenize)
    print(f"[+] {args.split}: {len(df)} tweets | model={args.model_name}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model, tok, proc = load_model(args.model_name, dtype)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[+] loaded on {model.device}, dtype={next(model.parameters()).dtype}")

    for setting in args.settings:
        t0 = time.time()
        template = (ROOT / "prompts" / f"{setting}_sentiment_prompt.txt").read_text(encoding="utf-8")
        texts = [chat_text(template.replace("{text}", s), tok, proc) for s in df["text"]]
        replies = []
        for i in tqdm(range(0, len(texts), args.batch_size), desc=setting):
            replies += generate_batch(model, tok, texts[i:i + args.batch_size], args.max_new_tokens)
        parsed = [parse_label(r) for r in replies]
        unparsed = sum(l is None for l in parsed)
        preds = [l or FALLBACK for l in parsed]
        gold = df["sentiment"].tolist()
        m = prf_metrics(gold, preds, LABELS)
        m["unparsed_replies"] = unparsed
        run = f"{tag}_prompt_{setting}"
        pd.DataFrame({"id": df["id"], "predictions": preds}).to_csv(results_dir / f"{run}.csv", index=False)
        pd.DataFrame({"id": df["id"], "reply": replies}).to_csv(results_dir / f"{run}_raw.csv", index=False)
        if args.split == "test" and not args.n_samples:
            df[["id", "sentiment"]].to_csv(results_dir / "ground.csv", index=False)
        summary = {"run_name": run, "args": {**vars(args), "mode": f"prompt_{setting}", "fusion": "-", "contrastive": "-", "seed": 0},
                   "model": f"{args.model_name} prompting, greedy decoding, max_new_tokens={args.max_new_tokens}",
                   args.split: {"sentiment": m}, "train_minutes": 0.0, "inference_minutes": (time.time() - t0) / 60}
        (results_dir / f"{run}_metrics.json").write_text(json.dumps(summary, indent=2))
        print(f"[+] {run}: macro P={m['macro_precision']:.4f} R={m['macro_recall']:.4f} F1={m['macro_f1']:.4f} | "
              f"unparsed={unparsed}/{len(preds)} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
