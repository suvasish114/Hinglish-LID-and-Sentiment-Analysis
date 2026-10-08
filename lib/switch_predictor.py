"""Experiment 2, approach 5 - switch-point prediction after "Beyond Detection: Predicting Code-Switch Points in
Multilingual Conversations" (Xie, Zhang, Koshal, Sushmita; WiML @ NeurIPS 2025).

The paper frames code-switching as token-level prediction of the *upcoming* switch point and studies two paradigms:
  (1) window-based models: BERT embeddings of the preceding tokens, fed to a recurrent network, with fixed or flexible
      (= whole prefix) context windows;
  (2) a transformer-based token classifier built on multilingual pre-trained models (mBERT, XLM-RoBERTa).
Only these paradigms and the AUC metric are taken from the paper (its full text is not openly accessible); the window
sizes, the recurrent cell / hidden size, the loss weighting and the decision threshold below are our choices.

Labels come from the gold SentiMix LID tags via lib/switch.switch_points.
  --paradigm window        P(switch_{i+1} = 1 | words_0..i): frozen `bert-base-multilingual-cased` word embeddings
                           (first sub-word of each word) -> LSTM/GRU over the last --window words (or the whole prefix
                           with --window 0) -> sigmoid.  Causal: it never sees the word it predicts the switch for.
  --paradigm transformer   P(switch_i = 1 | whole tweet): AutoModelForTokenClassification (default xlm-roberta-base,
                           last --unfreeze_last blocks trained) on the first sub-word of every word.
Both write per-word switch flags for the sentiment model (window predictions are shifted by one word):
    <out_dir>/{train,dev,test}_pred_switch.csv   (id, pred_switch)      +  <out_dir>/switch_metrics.json
Metrics: ROC-AUC (overall and per direction h->e / e->h, positives of the other direction excluded, as in the paper's
"Chinese-to-English" AUC), precision / recall / F1 of the switch class at the threshold that maximises dev F1.

    python3 lib/switch_predictor.py --paradigm window --window 5
    python3 lib/switch_predictor.py --paradigm window --window 0 --rnn gru
    python3 lib/switch_predictor.py --paradigm transformer --model_name xlm-roberta-base
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import encode_words, load_split                                     # noqa: E402
from lib.model import find_transformer_layers, is_norm, seed_everything           # noqa: E402
from lib.switch import switch_directions, switch_points, upcoming_switch_labels   # noqa: E402


# ----------------------------------------------------------------------------------------------- metrics
def switch_scores(probs, golds, dirs=None, threshold=None):
    """probs / golds / dirs: flat lists over word positions. Returns AUC (+ per direction) and P/R/F1 of the
    switch class at `threshold` (chosen on dev when None)."""
    p, g = np.asarray(probs, dtype=float), np.asarray(golds, dtype=int)
    out = {"n_words": int(len(g)), "switch_rate": float(g.mean())}
    out["auc"] = float(roc_auc_score(g, p)) if 0 < g.sum() < len(g) else float("nan")
    if dirs is not None:
        d = np.asarray(dirs)
        for name, code in [("h_to_e", 1), ("e_to_h", 2)]:
            keep = (g == 0) | (d == code)                      # only this direction's positives vs. all negatives
            gk = (d[keep] == code).astype(int)
            out[f"auc_{name}"] = float(roc_auc_score(gk, p[keep])) if 0 < gk.sum() < len(gk) else float("nan")
    if threshold is None:
        best = (-1.0, 0.5)
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_at(p, g, t)["f1"]
            if f > best[0]:
                best = (f, float(t))
        threshold = best[1]
    out.update(f1_at(p, g, threshold))
    out["threshold"] = float(threshold)
    return out


def f1_at(p, g, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (g == 1)).sum())
    fp = int(((pred == 1) & (g == 0)).sum())
    fn = int(((pred == 0) & (g == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def flag_scores(pred_flags: dict, df) -> dict:
    """P/R/F1 of per-word switch flags (dict id -> list[int]) against the gold switches of a split."""
    p, g = [], []
    for _, r in df.iterrows():
        gold = switch_points(r["tags"])
        pred = (list(pred_flags.get(r["id"], [])) + [0] * len(gold))[:len(gold)]
        p += pred
        g += gold
    return {"n_words": len(g), **f1_at(np.asarray(p, dtype=float), np.asarray(g), 0.5)}


# ----------------------------------------------------------------------------------------------- paradigm 1
@torch.no_grad()
def bert_word_embeddings(df, model_name, device, max_length=128, batch_size=64):
    """Frozen BERT embeddings of every word (first sub-word state of the last layer): list of [n_words, H]."""
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    rows, out = df.to_dict("records"), []
    for b in tqdm(range(0, len(rows), batch_size), desc=f"embeddings {model_name.split('/')[-1]}", leave=False):
        batch = rows[b:b + batch_size]
        encs = [encode_words(tok, r["tokens"], max_length) for r in batch]
        T = max(len(e[0]) for e in encs)
        ids = torch.tensor([e[0] + [tok.pad_token_id] * (T - len(e[0])) for e in encs], device=device)
        am = torch.tensor([e[1] + [0] * (T - len(e[1])) for e in encs], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = enc(input_ids=ids, attention_mask=am).last_hidden_state.float()
        for r, (_, _, widx), hb in zip(batch, encs, h):
            first = {}
            for j, w in enumerate(widx):
                if w is not None and w not in first:
                    first[w] = j
            vec = torch.zeros(len(r["tokens"]), hb.size(-1))
            for w, j in first.items():                       # words lost to truncation keep a zero vector
                vec[w] = hb[j].cpu()
            out.append(vec)
    return out


class WindowRNN(nn.Module):
    def __init__(self, dim, hidden=128, rnn="lstm", window=5, dropout=0.2):
        super().__init__()
        self.window = window
        cell = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn]
        self.rnn = cell(dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        """x [B,T,D] word embeddings -> logits [B,T] for the upcoming-switch label of every position."""
        if self.window <= 0:                                   # flexible window = the whole prefix
            o, _ = self.rnn(self.drop(x))
            return self.out(o).squeeze(-1)
        B, T, D = x.shape
        w = self.window
        xp = torch.cat([x.new_zeros(B, w - 1, D), x], dim=1)   # left-pad so position i sees words i-w+1..i
        win = xp.unfold(1, w, 1).permute(0, 1, 3, 2).reshape(B * T, w, D)
        o, _ = self.rnn(self.drop(win))
        return self.out(o[:, -1]).view(B, T)


def run_window(args, splits, device, out_dir):
    embs = {s: bert_word_embeddings(df, args.embedding_model, device, args.max_length) for s, df in splits.items()}
    labels = {s: [upcoming_switch_labels(switch_points(t)) for t in df["tags"]] for s, df in splits.items()}
    dirs = {s: [switch_directions(t)[1:] + [0] for t in df["tags"]] for s, df in splits.items()}
    pos = sum(sum(l) for l in labels["train"])
    tot = sum(len(l) for l in labels["train"])
    model = WindowRNN(embs["train"][0].size(-1), args.hidden, args.rnn, args.window).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((tot - pos) / max(pos, 1), device=device), reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def batches(split, shuffle):
        idx = np.arange(len(embs[split]))
        if shuffle:
            np.random.shuffle(idx)
        for b in range(0, len(idx), args.batch_size):
            sel = idx[b:b + args.batch_size]
            T = max(embs[split][i].size(0) for i in sel)
            x = torch.zeros(len(sel), T, embs[split][0].size(-1))
            y = torch.zeros(len(sel), T)
            m = torch.zeros(len(sel), T)
            for k, i in enumerate(sel):
                n = embs[split][i].size(0)
                x[k, :n], y[k, :n], m[k, :n] = embs[split][i], torch.tensor(labels[split][i], dtype=torch.float), 1
            yield sel, x.to(device), y.to(device), m.to(device)

    @torch.no_grad()
    def predict(split):
        model.eval()
        probs = [None] * len(embs[split])
        for sel, x, y, m in batches(split, False):
            p = torch.sigmoid(model(x))
            for k, i in enumerate(sel):
                probs[i] = p[k, :embs[split][i].size(0)].cpu().tolist()
        return probs

    def flat(split, probs):
        return ([v for p in probs for v in p], [v for l in labels[split] for v in l], [v for d in dirs[split] for v in d])

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for _, x, y, m in batches("train", True):
            loss = (crit(model(x), y) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        dev = switch_scores(*flat("dev", predict("dev")))
        print(f"[window epoch {ep + 1}] dev auc={dev['auc']:.4f} f1={dev['f1']:.4f}")
        if dev["auc"] > best:
            best, best_state = dev["auc"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    probs = {s: predict(s) for s in splits}
    dev_scores = switch_scores(*flat("dev", probs["dev"]))
    thr = dev_scores["threshold"]
    metrics = {s: switch_scores(*flat(s, probs[s]), threshold=thr) for s in splits}
    flags = {s: {r["id"]: [0] + [int(v >= thr) for v in p[:-1]]                     # shift: switch at word i+1
                 for r, p in zip(splits[s].to_dict("records"), probs[s])} for s in splits}
    return metrics, flags


# ----------------------------------------------------------------------------------------------- paradigm 2
def run_transformer(args, splits, device, out_dir):
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    if args.unfreeze_last >= 0:
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
    print(f"[transformer] trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    def items(df):
        out = []
        for r in df.to_dict("records"):
            ids, am, widx = encode_words(tok, r["tokens"], args.max_length)
            sw = switch_points(r["tags"])
            lab, prev = [], None
            for w in widx:
                lab.append(-100 if (w is None or w == prev) else sw[w])
                prev = w if w is not None else prev
            out.append({"input_ids": ids, "attention_mask": am, "labels": lab, "word_index": widx,
                        "id": r["id"], "n_words": len(r["tokens"]), "dirs": switch_directions(r["tags"]), "switch": sw})
        return out

    def collate(batch):
        T = max(len(b["input_ids"]) for b in batch)
        pad = lambda key, val: torch.tensor([b[key] + [val] * (T - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", tok.pad_token_id), "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", -100), "meta": batch}

    data = {s: items(df) for s, df in splits.items()}
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train"), collate_fn=collate,
                             generator=torch.Generator().manual_seed(args.seed)) for s, d in data.items()}
    pos = sum(l for d in data["train"] for l in d["labels"] if l == 1)
    tot = sum(1 for d in data["train"] for l in d["labels"] if l != -100)
    weight = torch.tensor([1.0, (tot - pos) / max(pos, 1)], device=device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total = len(loaders["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    @torch.no_grad()
    def predict(split):
        model.eval()
        probs, golds, dirs = {}, {}, {}
        for batch in loaders[split]:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
            p = torch.softmax(logits.float(), -1)[..., 1].cpu()
            for b, pb in zip(batch["meta"], p):
                probs_w, prev = [0.0] * b["n_words"], None
                for j, w in enumerate(b["word_index"]):
                    if w is not None and w != prev:
                        probs_w[w] = float(pb[j])
                    prev = w if w is not None else prev
                probs[b["id"]], golds[b["id"]], dirs[b["id"]] = probs_w, b["switch"], b["dirs"]
        order = [b["id"] for b in data[split]]
        flat = lambda d: [v for i in order for v in d[i]]
        return probs, (flat(probs), flat(golds), flat(dirs))

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for batch in tqdm(loaders["train"], desc=f"transformer epoch {ep + 1}/{args.epochs}", leave=False):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
            loss = nn.functional.cross_entropy(logits.float().view(-1, 2), batch["labels"].to(device).view(-1),
                                               weight=weight, ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        _, dev_flat = predict("dev")
        dev = switch_scores(*dev_flat)
        print(f"[transformer epoch {ep + 1}] dev auc={dev['auc']:.4f} f1={dev['f1']:.4f}")
        if dev["auc"] > best:
            best, best_state = dev["auc"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    preds = {s: predict(s) for s in splits}
    thr = switch_scores(*preds["dev"][1])["threshold"]
    metrics = {s: switch_scores(*preds[s][1], threshold=thr) for s in splits}
    flags = {s: {i: [int(v >= thr) for v in p] for i, p in preds[s][0].items()} for s in splits}
    for s in splits:                                              # switch_0 = 0 by convention
        for i in flags[s]:
            if flags[s][i]:
                flags[s][i][0] = 0
    return metrics, flags


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paradigm", default="window", choices=["window", "transformer"])
    ap.add_argument("--out_dir", default=None, help="default: results/exp2/switch/<paradigm tag>")
    # paradigm 1
    ap.add_argument("--embedding_model", default="bert-base-multilingual-cased", help="frozen BERT for the window model")
    ap.add_argument("--window", type=int, default=5, help="words of left context (0 = flexible window = whole prefix)")
    ap.add_argument("--rnn", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--hidden", type=int, default=128)
    # paradigm 2
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--unfreeze_last", type=int, default=2)
    # shared
    ap.add_argument("--epochs", type=int, default=None, help="default: 10 (window) / 3 (transformer)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 (window) / 2e-5 (transformer)")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_samples", type=int, default=None, help="smoke tests")
    args = ap.parse_args()
    args.epochs = args.epochs or (10 if args.paradigm == "window" else 3)
    args.lr = args.lr or (1e-3 if args.paradigm == "window" else 2e-5)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = (f"window_{args.rnn}_w{args.window or 'flex'}" if args.paradigm == "window"
           else f"transformer_{args.model_name.split('/')[-1]}_last{args.unfreeze_last}")
    out_dir = Path(args.out_dir or f"results/exp2/switch/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {s: load_split(s) for s in ["train", "dev", "test"]}
    if args.max_train_samples:
        splits["train"] = splits["train"].sample(n=args.max_train_samples, random_state=args.seed).reset_index(drop=True)
    t0 = time.time()
    metrics, flags = (run_window if args.paradigm == "window" else run_transformer)(args, splits, device, out_dir)
    for s in splits:
        metrics[s]["flag_vs_gold_switch"] = flag_scores(flags[s], splits[s])
        pd.DataFrame({"id": list(flags[s]), "pred_switch": [str(v) for v in flags[s].values()]}).to_csv(
            out_dir / f"{s}_pred_switch.csv", index=False)
        m = metrics[s]
        print(f"[{tag} {s:5s}] auc={m['auc']:.4f} (h->e {m.get('auc_h_to_e', float('nan')):.4f}, e->h {m.get('auc_e_to_h', float('nan')):.4f}) "
              f"P/R/F1={m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f} @thr={m['threshold']:.2f} | per-word flags F1={m['flag_vs_gold_switch']['f1']:.3f}")
    metrics["predictor"], metrics["args"], metrics["minutes"] = tag, vars(args), (time.time() - t0) / 60
    (out_dir / "switch_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[+] predicted switch points -> {out_dir}/{{train,dev,test}}_pred_switch.csv")


if __name__ == "__main__":
    main()
