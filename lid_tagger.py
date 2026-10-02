"""Experiment 1, task 1 - word-level LID tagging (h / e / o) on SentiMix.

Fine-tunes an `AutoModelForTokenClassification` (default xlm-roberta-base) on the SentiMix word-level
language tags, or builds a train-set lexicon baseline (`--model_name lexicon`). Every word is
predicted from its first sub-word. The script reports token-level accuracy, per-tag P/R/F1 and
macro-F1 on dev/test and writes the predicted tags of every split to

    <out_dir>/{train,dev,test}_pred_tags.csv      (columns: id, pred_tags)
    <out_dir>/lid_metrics.json

so that `sentiment_lid.py --lid_source predicted --pred_tags_dir <out_dir>` can fuse *predicted*
tags. Once written, the tagger is FROZEN: the sentiment model never back-propagates into it.
`lid_eval.py` scores any such CSV (also those produced by other LID tools / LLM prompting).

    python3 lid_tagger.py --model_name xlm-roberta-base --out_dir results/exp1/lid/xlm-roberta-base_full
    python3 lid_tagger.py --model_name lexicon          --out_dir results/exp1/lid/lexicon
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.sentimix_data import LID2ID, LID_TAGS, encode_words, load_split  # noqa: E402
from lib.sentiment_model import find_transformer_layers, is_norm          # noqa: E402
from lib.trainer import seed_everything                                    # noqa: E402
from lid_eval import lid_scores                                            # noqa: E402


class TagDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.rows, self.tok, self.max_length = df.to_dict("records"), tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, am, widx = encode_words(self.tok, r["tokens"], self.max_length)
        labels, prev = [], None
        for w in widx:                                   # supervise only the first sub-word of a word
            labels.append(-100 if (w is None or w == prev) else LID2ID[r["tags"][w]])
            prev = w if w is not None else prev
        return {"input_ids": ids, "attention_mask": am, "labels": labels, "word_index": widx,
                "id": r["id"], "n_words": len(r["tokens"])}


def collate(batch, pad_id):
    T = max(len(b["input_ids"]) for b in batch)

    def pad(key, val):
        return torch.tensor([b[key] + [val] * (T - len(b[key])) for b in batch])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100), "meta": batch}


@torch.no_grad()
def predict(model, loader, device, use_amp):
    """id -> list of predicted tags (one per word; words lost to truncation default to `o`)."""
    model.eval()
    out = {}
    for batch in tqdm(loader, desc="predict", leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
        pred = logits.argmax(-1).cpu().tolist()
        for b, p in zip(batch["meta"], pred):
            tags, prev = ["o"] * b["n_words"], None
            for j, w in enumerate(b["word_index"]):
                if w is not None and w != prev:
                    tags[w] = LID_TAGS[p[j]]
                prev = w if w is not None else prev
            out[b["id"]] = tags
    return out


def lexicon_tagger(train_df):
    """Most frequent train tag per word; unseen words: `o` if no letter, else the majority alphabetic tag."""
    counts = defaultdict(Counter)
    alpha = Counter()
    for _, r in train_df.iterrows():
        for w, t in zip(r["tokens"], r["tags"]):
            counts[w][t] += 1
            if any(ch.isalpha() for ch in w):
                alpha[t] += 1
    majority = alpha.most_common(1)[0][0]
    lexicon = {w: c.most_common(1)[0][0] for w, c in counts.items()}

    def tag(words):
        return [lexicon.get(w, "o" if not any(ch.isalpha() for ch in w) else majority) for w in words]
    return tag


def write_predictions(pred, out_dir, split):
    pd.DataFrame({"id": list(pred), "pred_tags": [str(v) for v in pred.values()]}).to_csv(
        out_dir / f"{split}_pred_tags.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="xlm-roberta-base", help="HF model id, or `lexicon` for the baseline")
    ap.add_argument("--out_dir", default=None, help="default: results/exp1/lid/<model>_<full|lastK>")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--unfreeze_last", type=int, default=-1, help="blocks to train (-1 = all; 2 = project setting)")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_samples", type=int, default=None, help="smoke tests")
    ap.add_argument("--predict_only", action="store_true", help="reload <out_dir>/model and only write predictions")
    ap.add_argument("--save_model", action="store_true", help="save the fine-tuned tagger to <out_dir>/model")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    tag = "lexicon" if args.model_name == "lexicon" else \
        args.model_name.rstrip("/").split("/")[-1] + ("_full" if args.unfreeze_last < 0 else f"_last{args.unfreeze_last}")
    out_dir = Path(args.out_dir or f"results/exp1/lid/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {s: load_split(s) for s in ["train", "dev", "test"]}
    if args.max_train_samples:
        splits["train"] = splits["train"].sample(n=args.max_train_samples, random_state=args.seed)
    t0 = time.time()

    if args.model_name == "lexicon":
        tagger = lexicon_tagger(splits["train"])
        preds = {s: {r["id"]: tagger(r["tokens"]) for _, r in df.iterrows()} for s, df in splits.items()}
    else:
        src = str(out_dir / "model") if args.predict_only else args.model_name
        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForTokenClassification.from_pretrained(
            src, num_labels=len(LID_TAGS), id2label=dict(enumerate(LID_TAGS)), label2id=LID2ID,
            trust_remote_code=args.trust_remote_code).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        if args.unfreeze_last >= 0 and not args.predict_only:
            base = model.base_model
            for p in base.parameters():
                p.requires_grad = False
            layers = find_transformer_layers(base)
            for layer in list(layers)[len(layers) - args.unfreeze_last:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for _, m in base.named_children():
                if is_norm(m):
                    for p in m.parameters():
                        p.requires_grad = True
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[+] {tag}: trainable params {n_train / 1e6:.1f}M")
        pad_id = tokenizer.pad_token_id
        loaders = {s: DataLoader(TagDataset(df, tokenizer, args.max_length), batch_size=args.batch_size,
                                 shuffle=(s == "train" and not args.predict_only), num_workers=2,
                                 collate_fn=lambda b: collate(b, pad_id), generator=torch.Generator().manual_seed(args.seed))
                   for s, df in splits.items()}
        if not args.predict_only:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
            total = len(loaders["train"]) * args.epochs
            sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
            best, best_state = -1.0, None
            for ep in range(args.epochs):
                model.train()
                for batch in tqdm(loaders["train"], desc=f"LID epoch {ep + 1}/{args.epochs}"):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        loss = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                                     labels=batch["labels"].to(device)).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                dev_scores = lid_scores(predict(model, loaders["dev"], device, use_amp), splits["dev"])
                print(f"[epoch {ep + 1}] dev: acc={dev_scores['token_accuracy']:.4f} macroF1={dev_scores['macro_f1']:.4f}")
                if dev_scores["macro_f1"] > best:
                    best = dev_scores["macro_f1"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            if args.save_model:
                model.save_pretrained(out_dir / "model")
                tokenizer.save_pretrained(out_dir / "model")
        preds = {s: predict(model, loaders[s], device, use_amp) for s in splits}

    metrics = {"tagger": tag, "args": vars(args), "minutes": (time.time() - t0) / 60}
    for s, df in splits.items():
        write_predictions(preds[s], out_dir, s)
        metrics[s] = lid_scores(preds[s], df)
        print(f"[LID {tag} {s:5s}] " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics[s].items()}))
    (out_dir / "lid_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[+] predicted tags -> {out_dir}/{{train,dev,test}}_pred_tags.csv")


if __name__ == "__main__":
    main()
