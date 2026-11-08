"""Training / evaluation loop shared by Experiment 0 (baselines) and Experiment 1 (LID fusion).

Per run it writes, in `--results_dir`:
    <run_name>.csv            id, predictions      (consumed by the shared evaluate_all.py)
    ground.csv                id, sentiment        (the shared ground-truth file for the test split)
    <run_name>_metrics.json   args, per-epoch dev history, final dev/test metrics
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from lib.metrics import paper_header, paper_row, sentiment_metrics
from lib.sentimix_data import (ID2LABEL, LID_SPECIAL, encode_words, lid_ids_for_subwords,
                               load_predicted_tags, load_split)
from lib.sentiment_model import SentimentClassifier


class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, pred_tags=None):
        self.rows = df.to_dict("records")
        self.tok, self.max_length, self.pred_tags = tokenizer, max_length, pred_tags

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        tags = r["tags"] if self.pred_tags is None else self.pred_tags.get(r["id"], [])
        if len(tags) < len(r["tokens"]):                       # tagger truncated the tweet: pad with `o`
            tags = list(tags) + ["o"] * (len(r["tokens"]) - len(tags))
        ids, am, widx = encode_words(self.tok, r["tokens"], self.max_length)
        return {"input_ids": ids, "attention_mask": am, "lid_ids": lid_ids_for_subwords(widx, tags),
                "label": int(r["label"]), "id": r["id"]}


def make_collate(pad_id):
    def collate(batch):
        T = max(len(b["input_ids"]) for b in batch)

        def pad(key, val):
            return torch.tensor([b[key] + [val] * (T - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
                "lid_ids": pad("lid_ids", LID_SPECIAL), "labels": torch.tensor([b["label"] for b in batch]),
                "ids": [b["id"] for b in batch]}
    return collate


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model, loader, use_amp, desc="eval"):
    model.eval()
    ids, preds, golds = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(batch["input_ids"], batch["attention_mask"], lid_ids=batch["lid_ids"])
        preds += out["logits"].argmax(-1).cpu().tolist()
        golds += batch["labels"].tolist()
        ids += batch["ids"]
    y_true = [ID2LABEL[g] for g in golds]
    y_pred = [ID2LABEL[p] for p in preds]
    return sentiment_metrics(y_true, y_pred), pd.DataFrame({"id": ids, "predictions": y_pred})


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def parse_max_memory(spec):
    """'0:10GiB,1:22GiB' -> {0: '10GiB', 1: '22GiB'}"""
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        k, v = item.split(":")
        out[int(k) if k.strip().isdigit() else k.strip()] = v.strip()
    return out


def default_run_name(args):
    name = args.model_name.rstrip("/").split("/")[-1]
    name += "_full" if args.unfreeze_last < 0 else f"_last{args.unfreeze_last}"
    fusion = getattr(args, "lid_fusion", "none")
    if fusion != "none":
        name += f"_lid-{fusion}@{args.lid_inject_layer}_{args.lid_source}"
    return f"{name}_seed{args.seed}"


def train_and_evaluate(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or default_run_name(args)
    lid_fusion = getattr(args, "lid_fusion", "none")
    lid_source = getattr(args, "lid_source", "gold")

    # ---- data -----------------------------------------------------------------------------------
    splits = {s: load_split(s) for s in ["train", "dev", "test"]}
    if args.max_train_samples:
        splits["train"] = splits["train"].sample(n=min(args.max_train_samples, len(splits["train"])), random_state=args.seed)
    if args.max_eval_samples:
        for s in ["dev", "test"]:
            splits[s] = splits[s].sample(n=min(args.max_eval_samples, len(splits[s])), random_state=args.seed)
    pred_tags = None
    if lid_fusion != "none" and lid_source == "predicted":
        pred_tags = {s: load_predicted_tags(Path(args.pred_tags_dir) / f"{s}_pred_tags.csv") for s in splits}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:                             # decoder-only models (Qwen/Llama)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collate = make_collate(tokenizer.pad_token_id)
    eval_bs = args.eval_batch_size or 2 * args.batch_size
    loaders = {s: DataLoader(SentimentDataset(df, tokenizer, args.max_length, pred_tags[s] if pred_tags else None),
                             batch_size=args.batch_size if s == "train" else eval_bs, shuffle=(s == "train"),
                             collate_fn=collate, num_workers=2, generator=torch.Generator().manual_seed(args.seed))
               for s, df in splits.items()}

    # ---- model ----------------------------------------------------------------------------------
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    device_map = None if args.device_map in (None, "", "none") else args.device_map
    model = SentimentClassifier(args.model_name, pooling=args.pooling, unfreeze_last=args.unfreeze_last,
                                dropout=args.dropout, lid_fusion=lid_fusion, lid_dim=getattr(args, "lid_dim", 16),
                                lid_inject_layer=getattr(args, "lid_inject_layer", 0), dtype=dtype,
                                device_map=device_map, max_memory=parse_max_memory(args.max_memory),
                                trust_remote_code=args.trust_remote_code,
                                gradient_checkpointing=args.gradient_checkpointing)
    if device_map is None:
        model.to(device)
    model.encoder.config.pad_token_id = tokenizer.pad_token_id
    print(f"[+] run={run_name}\n[+] {model.describe()}")

    head_names = ("classifier", "lid_emb", "lid_proj")
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.split(".")[0] in head_names]
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.split(".")[0] not in head_names]
    groups = [{"params": enc_params, "lr": args.lr}]
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr or args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = (len(loaders["train"]) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    print(f"[+] train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])} | "
          f"effective batch={args.batch_size * args.grad_accum} steps={total_steps} amp={use_amp}")

    # ---- training -------------------------------------------------------------------------------
    best, best_state, history, t0 = -1.0, None, [], time.time()
    for ep in range(args.epochs):
        model.train()
        losses, bar = [], tqdm(loaders["train"], desc=f"epoch {ep + 1}/{args.epochs}")
        opt.zero_grad()
        for step, batch in enumerate(bar, start=1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch["input_ids"], batch["attention_mask"], lid_ids=batch["lid_ids"], labels=batch["labels"])
            (out["loss"] / args.grad_accum).backward()
            losses.append(out["loss"].item())
            if step % args.grad_accum == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % 20 == 0:
                bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")
        dev_m, _ = run_eval(model, loaders["dev"], use_amp, desc="dev")
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "dev": dev_m,
                        "minutes": (time.time() - t0) / 60})
        print(f"[epoch {ep + 1}] loss={np.mean(losses):.4f} dev: wF1={dev_m['weighted_f1']:.4f} "
              f"macroF1={dev_m['macro_f1']:.4f} acc={dev_m['accuracy']:.4f} ({(time.time() - t0) / 60:.1f} min)")
        if dev_m[args.select_metric] > best:
            best, best_state = dev_m[args.select_metric], trainable_state(model)

    # ---- final evaluation with the best dev checkpoint --------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    dev_m, dev_pred = run_eval(model, loaders["dev"], use_amp, desc="dev")
    test_m, test_pred = run_eval(model, loaders["test"], use_amp, desc="test")
    test_pred.to_csv(results_dir / f"{run_name}.csv", index=False)
    splits["test"][["id", "sentiment"]].to_csv(results_dir / "ground.csv", index=False)
    summary = {"run_name": run_name, "args": vars(args), "model": model.describe(),
               "best_dev_" + args.select_metric: best, "dev": dev_m, "test": test_m, "history": history,
               "train_minutes": (time.time() - t0) / 60}
    (results_dir / f"{run_name}_metrics.json").write_text(json.dumps(summary, indent=2))
    if args.save_model:
        ckpt = Path("checkpoints") / run_name
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(trainable_state(model), ckpt / "trainable_weights.pt")
    print("[+] TEST  " + paper_header())
    print("[+] TEST  " + paper_row(run_name, test_m))
    print(f"[+] predictions -> {results_dir / (run_name + '.csv')} | ground -> {results_dir / 'ground.csv'} | "
          f"metrics -> {results_dir / (run_name + '_metrics.json')}")
    return summary


def add_common_args(p):
    p.add_argument("--model_name", default="xlm-roberta-base", help="HF model id or local path")
    p.add_argument("--run_name", default=None)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=None, help="lr of classifier/LID modules (default: --lr)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze_last", type=int, default=2, help="transformer blocks to train (-1 = all)")
    p.add_argument("--pooling", default="auto", choices=["auto", "cls", "last", "mean"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--select_metric", default="weighted_f1", choices=["weighted_f1", "macro_f1", "accuracy"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device_map", default="none", help="'auto' to shard a large model over all GPUs")
    p.add_argument("--max_memory", default=None, help="e.g. '0:10GiB,1:22GiB' (with --device_map auto)")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true", help="needed for ai4bharat/IndicBERT-v3-1B")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None, help="smoke tests")
    p.add_argument("--max_eval_samples", type=int, default=None, help="smoke tests")
    p.add_argument("--save_model", action="store_true")
    return p
