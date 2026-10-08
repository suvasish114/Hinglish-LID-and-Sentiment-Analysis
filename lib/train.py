"""Experiment 2 - training / evaluation loop shared by the five approaches.

Switch points come from --switch_source:
    gold       gold SentiMix LID tags                 -> lib/switch.switch_points
    lid        tags predicted by the frozen tagger    (<switch_dir>/{split}_pred_tags.csv, lib/lid_tagger.py)
    predictor  switch flags of lib/switch_predictor.py (<switch_dir>/{split}_pred_switch.csv)   = approach 5
Per run (in --results_dir): <run>.csv (id, predictions) + ground.csv for the shared evaluate_all.py, and
<run>_metrics.json with dev/test accuracy, weighted / macro P-R-F1 and per-class F1.
"""

import ast
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from lib.data import ID2LABEL, LABELS, encode_words, load_predicted_tags, load_split
from lib.model import BIN_PAD, DIST_PAD, SentimentModel, seed_everything
from lib.switch import align_to_subwords, language_view, switch_distances, switch_points

SWITCH_SOURCES = ["gold", "lid", "predictor"]


def sentiment_metrics(y_true, y_pred) -> dict:
    out = {"n": len(y_true), "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))) if len(y_true) else 0.0}
    for avg in ["weighted", "macro"]:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average=avg, zero_division=0)
        out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = float(p), float(r), float(f)
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out.update({f"{c}_f1": float(v) for c, v in zip(LABELS, f)})
    return out


def load_predicted_switches(path) -> dict:
    df = pd.read_csv(path, dtype=str)
    return {r["id"]: [int(v) for v in ast.literal_eval(r["pred_switch"])] for _, r in df.iterrows()}


class SwitchDataset(Dataset):
    def __init__(self, df, tokenizer, args, switch_lookup=None):
        self.rows, self.tok, self.args, self.lookup = df.to_dict("records"), tokenizer, args, switch_lookup

    def __len__(self):
        return len(self.rows)

    def switches_for(self, r):
        n = len(r["tokens"])
        if self.lookup is None:
            return switch_points(r["tags"])
        v = list(self.lookup.get(r["id"], []))
        if self.args.switch_source == "lid":
            v = switch_points((v + ["o"] * n)[:n])
        return (v + [0] * n)[:n]

    def __getitem__(self, i):
        r = self.rows[i]
        sw = self.switches_for(r)
        if self.args.pooling == "distance":
            feat, pad = switch_distances(sw), DIST_PAD
        else:
            feat, pad = sw, BIN_PAD
        ids, am, widx = encode_words(self.tok, r["tokens"], self.args.max_length)
        return {"input_ids": ids, "attention_mask": am, "pool_mask": [0 if w is None else 1 for w in widx],
                "switch_feat": align_to_subwords(feat, widx, pad), "label": int(r["label"]), "id": r["id"],
                "n_switches": int(sum(sw)), "langs": sorted({t for t in r["tags"] if t in ("h", "e")})}


def make_collate(pad_id):
    def collate(batch):
        T = max(len(b["input_ids"]) for b in batch)
        pad = lambda key, val: torch.tensor([b[key] + [val] * (T - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
                "pool_mask": pad("pool_mask", 0), "switch_feat": pad("switch_feat", 0),
                "labels": torch.tensor([b["label"] for b in batch]), "ids": [b["id"] for b in batch],
                "n_switches": [b["n_switches"] for b in batch], "langs": [b["langs"] for b in batch]}
    return collate


@torch.no_grad()
def teacher_table(model, tokenizer, df, args, device_pad):
    """Approach 4: frozen-teacher embeddings of the Hindi and English language views of every training tweet,
    computed with the encoder as loaded (before training). Returns {id: {"h": vec, "e": vec}} on CPU."""
    model.eval()
    rows = df.to_dict("records")
    table = {}
    for lang in ("h", "e"):
        for b in tqdm(range(0, len(rows), 2 * args.batch_size), desc=f"teacher views ({lang})", leave=False):
            batch = rows[b:b + 2 * args.batch_size]
            encs = [encode_words(tokenizer, language_view(r["tokens"], r["tags"], lang), args.max_length) for r in batch]
            T = max(len(e[0]) for e in encs)
            ids = torch.tensor([e[0] + [device_pad] * (T - len(e[0])) for e in encs])
            am = torch.tensor([e[1] + [0] * (T - len(e[1])) for e in encs])
            pm = torch.tensor([[0 if w is None else 1 for w in e[2]] + [0] * (T - len(e[2])) for e in encs])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available() and not args.no_amp):
                emb = model.teacher_embed(ids, am, pm).cpu().to(torch.float16)
            for r, v in zip(batch, emb):
                table.setdefault(r["id"], {})[lang] = v
    return table


@torch.no_grad()
def run_eval(model, loader, use_amp, desc="eval"):
    model.eval()
    ids, preds, golds, nsw = [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(batch["input_ids"], batch["attention_mask"], batch["pool_mask"], batch["switch_feat"])
        preds += out["logits"].argmax(-1).cpu().tolist()
        golds += batch["labels"].tolist()
        ids += batch["ids"]
        nsw += batch["n_switches"]
    y_true, y_pred = [ID2LABEL[g] for g in golds], [ID2LABEL[p] for p in preds]
    m = sentiment_metrics(y_true, y_pred)
    return m, pd.DataFrame({"id": ids, "predictions": y_pred, "n_switches": nsw, "gold": y_true})


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def parse_max_memory(spec):
    if not spec:
        return None
    return {(int(k) if k.strip().isdigit() else k.strip()): v.strip() for k, v in (x.split(":") for x in spec.split(","))}


def default_run_name(args):
    name = f"{args.model_name.rstrip('/').split('/')[-1]}_A{args.approach_id}-{args.approach}"
    if args.approach in ("binary", "distance", "beyond_detection"):
        name += f"_{args.pooling}_{args.switch_source}"
    if args.approach == "contrastive":
        name += f"_w{args.contrastive_weight}" + ("_filter" if args.contrastive_filter else "")
    return f"{name}_seed{args.seed}"


def train_and_evaluate(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or default_run_name(args)

    splits = {s: load_split(s) for s in ["train", "dev", "test"]}
    if args.max_train_samples:
        splits["train"] = splits["train"].sample(n=min(args.max_train_samples, len(splits["train"])), random_state=args.seed)
    if args.max_eval_samples:
        for s in ["dev", "test"]:
            splits[s] = splits[s].sample(n=min(args.max_eval_samples, len(splits[s])), random_state=args.seed)
    lookups = {s: None for s in splits}
    if args.pooling != "none" and args.switch_source != "gold":
        for s in splits:
            f = Path(args.switch_dir) / (f"{s}_pred_tags.csv" if args.switch_source == "lid" else f"{s}_pred_switch.csv")
            lookups[s] = load_predicted_tags(f) if args.switch_source == "lid" else load_predicted_switches(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collate = make_collate(tokenizer.pad_token_id)
    eval_bs = args.eval_batch_size or 2 * args.batch_size
    loaders = {s: DataLoader(SwitchDataset(df, tokenizer, args, lookups[s]), batch_size=args.batch_size if s == "train" else eval_bs,
                             shuffle=(s == "train"), collate_fn=collate, num_workers=2,
                             generator=torch.Generator().manual_seed(args.seed)) for s, df in splits.items()}

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device_map = None if args.device_map in (None, "", "none") else args.device_map
    model = SentimentModel(args.model_name, pooling=args.pooling, switch_dim=args.switch_dim, unfreeze_last=args.unfreeze_last,
                           dropout=args.dropout, dtype=dtype, device_map=device_map, max_memory=parse_max_memory(args.max_memory),
                           trust_remote_code=args.trust_remote_code, gradient_checkpointing=args.gradient_checkpointing,
                           contrastive=(args.approach == "contrastive"), contrastive_queue=args.contrastive_queue,
                           contrastive_tau=args.contrastive_tau, contrastive_filter=args.contrastive_filter)
    if device_map is None:
        model.to(device)
    model.encoder.config.pad_token_id = tokenizer.pad_token_id
    print(f"[+] run={run_name}\n[+] {model.describe()}")

    teachers = None
    if args.approach == "contrastive":
        teachers = teacher_table(model, tokenizer, splits["train"], args, tokenizer.pad_token_id)
        model.contrastive.set_center(torch.stack([v.float() for d in teachers.values() for v in d.values()]).mean(0))
    view_rng = random.Random(args.seed)

    head_names = ("pooling", "classifier")
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.split(".")[0] in head_names]
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.split(".")[0] not in head_names]
    groups = [{"params": enc, "lr": args.lr}] + ([{"params": head, "lr": args.head_lr or args.lr}] if head else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = (len(loaders["train"]) + args.grad_accum - 1) // args.grad_accum
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    print(f"[+] train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])} | batch={args.batch_size * args.grad_accum} steps={total} amp={use_amp}")

    best, best_state, history, t0 = -1.0, None, [], time.time()
    for ep in range(args.epochs):
        model.train()
        losses, co_losses = [], []
        bar = tqdm(loaders["train"], desc=f"epoch {ep + 1}/{args.epochs}")
        opt.zero_grad()
        for step, batch in enumerate(bar, start=1):
            teacher = None
            if teachers is not None:          # positive view: a random language present in the tweet
                teacher = torch.stack([teachers[i][view_rng.choice(langs) if langs else "h"].float()
                                       for i, langs in zip(batch["ids"], batch["langs"])])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch["input_ids"], batch["attention_mask"], batch["pool_mask"], batch["switch_feat"],
                            labels=batch["labels"], teacher_embed=teacher, contrastive_weight=args.contrastive_weight)
            (out["loss"] / args.grad_accum).backward()
            losses.append(out["ce_loss"].item())
            if out["co_loss"] is not None:
                co_losses.append(out["co_loss"].item())
            if step % args.grad_accum == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % 20 == 0:
                bar.set_postfix(ce=f"{np.mean(losses[-20:]):.3f}", **({"co": f"{np.mean(co_losses[-20:]):.3f}"} if co_losses else {}))
        dev_m, _ = run_eval(model, loaders["dev"], use_amp, "dev")
        history.append({"epoch": ep + 1, "train_ce": float(np.mean(losses)), "train_contrastive": float(np.mean(co_losses)) if co_losses else None,
                        "dev": dev_m, "minutes": (time.time() - t0) / 60})
        print(f"[epoch {ep + 1}] ce={np.mean(losses):.4f}" + (f" co={np.mean(co_losses):.4f}" if co_losses else "") +
              f" dev: wF1={dev_m['weighted_f1']:.4f} macroF1={dev_m['macro_f1']:.4f} acc={dev_m['accuracy']:.4f} ({(time.time() - t0) / 60:.1f} min)")
        if dev_m[args.select_metric] > best:
            best, best_state = dev_m[args.select_metric], trainable_state(model)

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    dev_m, _ = run_eval(model, loaders["dev"], use_amp, "dev")
    test_m, test_pred = run_eval(model, loaders["test"], use_amp, "test")
    for lo, hi, name in [(1, 2, "1-2"), (3, 5, "3-5"), (6, 999, "6+")]:   # breakdown by number of switch points
        sub = test_pred[(test_pred.n_switches >= lo) & (test_pred.n_switches <= hi)]
        if len(sub):
            test_m[f"switches_{name}"] = {"n": len(sub), **{k: v for k, v in sentiment_metrics(sub["gold"], sub["predictions"]).items() if k in ("weighted_f1", "macro_f1", "accuracy")}}
    test_pred[["id", "predictions"]].to_csv(results_dir / f"{run_name}.csv", index=False)
    splits["test"][["id", "sentiment"]].to_csv(results_dir / "ground.csv", index=False)
    summary = {"run_name": run_name, "args": vars(args), "model": model.describe(), f"best_dev_{args.select_metric}": best,
               "dev": dev_m, "test": test_m, "history": history, "train_minutes": (time.time() - t0) / 60}
    (results_dir / f"{run_name}_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[+] TEST wF1={test_m['weighted_f1']:.4f} macroF1={test_m['macro_f1']:.4f} acc={test_m['accuracy']:.4f} | "
          f"predictions -> {results_dir / (run_name + '.csv')}")
    return summary
