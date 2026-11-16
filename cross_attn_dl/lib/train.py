"""Training / evaluation loop of the joint LID + sentiment model.

Per run (in --results_dir):
    <run>.csv             id, predictions           tweet sentiment on the test split
    <run>_lid.csv         id, pred_tags             word-level LID tags on the test split
    <run>_metrics.json    dev / test macro P-R-F1 (+ per-class) of both tasks, training history, arguments
    ground.csv            id, sentiment             test gold (written once per results dir)
Metrics are precision, recall and F1 only (macro-averaged, plus per class); accuracy is not reported.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from lib.data import ID2LABEL, ID2LID, LABELS, LID_TAGS, JointDataset, load_split, make_collate
from lib.model import JointModel, seed_everything


def prf_metrics(y_true, y_pred, classes) -> dict:
    """Macro precision / recall / F1 and the per-class values (no accuracy)."""
    out = {"n": len(y_true)}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    out["macro_precision"], out["macro_recall"], out["macro_f1"] = float(p), float(r), float(f)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average=None, zero_division=0)
    out["per_class"] = {c: {"precision": float(pi), "recall": float(ri), "f1": float(fi)} for c, pi, ri, fi in zip(classes, p, r, f)}
    return out


@torch.no_grad()
def run_eval(model, loader, use_amp, desc="eval"):
    """Sentiment: one prediction per tweet. LID: the tag of a word = argmax at its first sub-word; words cut off by
    max_length (rare) are tagged `o`, so every gold word is scored."""
    model.eval()
    ids, preds, golds, lid_pred_rows, lid_true, lid_pred = [], [], [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(batch["input_ids"], batch["attention_mask"], batch["lid_labels"])
        preds += out["logits"].argmax(-1).cpu().tolist()
        golds += batch["labels"].tolist()
        ids += batch["ids"]
        if out["lid_logits"] is not None:
            tag_pred = out["lid_logits"].argmax(-1).cpu()
            for b in range(len(batch["ids"])):
                words = ["o"] * batch["n_words"][b]
                for t in torch.nonzero(batch["first_mask"][b]).flatten().tolist():
                    words[int(batch["word_index"][b, t])] = ID2LID[int(tag_pred[b, t])]
                lid_pred_rows.append(words)
    y_true, y_pred = [ID2LABEL[g] for g in golds], [ID2LABEL[p] for p in preds]
    metrics = {"sentiment": prf_metrics(y_true, y_pred, LABELS)}
    rows = loader.dataset.rows
    if lid_pred_rows:
        for r, words in zip(rows, lid_pred_rows):
            lid_true += r["tags"]
            lid_pred += words
        metrics["lid"] = prf_metrics(lid_true, lid_pred, LID_TAGS)
    frame = pd.DataFrame({"id": ids, "predictions": y_pred, "gold": y_true})
    lid_frame = pd.DataFrame({"id": [r["id"] for r in rows], "pred_tags": [str(w) for w in lid_pred_rows]}) if lid_pred_rows else None
    return metrics, frame, lid_frame


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def save_checkpoint(model, args, run_name, best_epoch, dev_m, test_m):
    """<checkpoint_dir>/<run>.pt = the weights that were trained (last blocks of the encoder + all heads) at the best
    dev epoch, plus everything needed to rebuild the model around the pre-trained encoder `model_name`. The frozen
    encoder weights are not stored: they are identical to the pre-trained checkpoint. Read by joint_contrastive.py."""
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {"format": "joint-lid-sentiment/trainable-v1", "run_name": run_name, "model_name": args.model_name,
            "mode": args.mode, "seed": args.seed, "best_epoch": best_epoch, "dtype": args.dtype, "max_length": args.max_length,
            "model_kwargs": {"use_lid": model.use_lid, "fusion": model.fusion, "contrastive": model.contrastive,
                             "proj_dim": args.proj_dim, "n_heads": args.n_heads, "unfreeze_last": args.unfreeze_last,
                             "dropout": args.dropout},
            "labels": list(LABELS), "lid_tags": list(LID_TAGS), "metrics": {"dev": dev_m, "test": test_m},
            "state_dict": trainable_state(model)}
    path = ckpt_dir / f"{run_name}.pt"
    torch.save(ckpt, path)
    n = sum(v.numel() for v in ckpt["state_dict"].values())
    print(f"[+] checkpoint -> {path} ({n / 1e6:.1f}M trained parameters, {path.stat().st_size / 2**20:.0f} MB, best epoch {best_epoch})")


def parse_max_memory(spec):
    if not spec:
        return None
    return {(int(k) if k.strip().isdigit() else k.strip()): v.strip() for k, v in (x.split(":") for x in spec.split(","))}


def default_run_name(args):
    tag = args.model_name.rstrip("/").split("/")[-1]
    return f"{tag}_{args.mode}_seed{args.seed}"


def fmt(m: dict, key: str) -> str:
    return f"P={m[key]['macro_precision']:.4f} R={m[key]['macro_recall']:.4f} F1={m[key]['macro_f1']:.4f}" if key in m else "-"


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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collate = make_collate(tokenizer.pad_token_id)
    eval_bs = args.eval_batch_size or 2 * args.batch_size
    loaders = {s: DataLoader(JointDataset(df, tokenizer, args.max_length), batch_size=args.batch_size if s == "train" else eval_bs,
                             shuffle=(s == "train"), collate_fn=collate, num_workers=2,
                             generator=torch.Generator().manual_seed(args.seed)) for s, df in splits.items()}

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device_map = None if args.device_map in (None, "", "none") else args.device_map
    model = JointModel(args.model_name, use_lid=args.use_lid, fusion=args.fusion, contrastive=args.contrastive,
                       proj_dim=args.proj_dim, n_heads=args.n_heads, unfreeze_last=args.unfreeze_last, dropout=args.dropout,
                       dtype=dtype, device_map=device_map, max_memory=parse_max_memory(args.max_memory),
                       trust_remote_code=args.trust_remote_code, gradient_checkpointing=args.gradient_checkpointing)
    if device_map is None:
        model.to(device)
    model.encoder.config.pad_token_id = tokenizer.pad_token_id
    print(f"[+] run={run_name}\n[+] {model.describe()}")

    head_names = set(model.head_parameter_names())
    head = [p for n, p in model.named_parameters() if n in head_names]
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n not in head_names]
    groups = [{"params": enc, "lr": args.lr}] + ([{"params": head, "lr": args.head_lr or args.lr}] if head else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = (len(loaders["train"]) + args.grad_accum - 1) // args.grad_accum
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    print(f"[+] train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])} | "
          f"batch={args.batch_size * args.grad_accum} steps={total} amp={use_amp}")

    best, best_state, best_epoch, history, t0 = -1.0, None, None, [], time.time()
    keys = ["ce_sent", "ce_lid", "con_lid", "con_sent"]
    for ep in range(args.epochs):
        model.train()
        sums = {k: [] for k in keys}
        bar = tqdm(loaders["train"], desc=f"epoch {ep + 1}/{args.epochs}")
        opt.zero_grad()
        for step, batch in enumerate(bar, start=1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch["input_ids"], batch["attention_mask"], batch["lid_labels"], labels=batch["labels"],
                            lid_weight=args.lid_weight, con_weight_lid=args.con_weight_lid,
                            con_weight_sent=args.con_weight_sent, tau=args.tau, max_con_tokens=args.max_con_tokens)
            (out["loss"] / args.grad_accum).backward()
            for k in keys:
                if out[k] is not None:
                    sums[k].append(out[k].item())
            if step % args.grad_accum == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % 20 == 0:
                bar.set_postfix({k: f"{np.mean(v[-20:]):.3f}" for k, v in sums.items() if v})
        dev_m, _, _ = run_eval(model, loaders["dev"], use_amp, "dev")
        epoch_losses = {k: (float(np.mean(v)) if v else None) for k, v in sums.items()}
        history.append({"epoch": ep + 1, "train_loss": epoch_losses, "dev": dev_m, "minutes": (time.time() - t0) / 60})
        print(f"[epoch {ep + 1}] " + " ".join(f"{k}={v:.4f}" for k, v in epoch_losses.items() if v is not None) +
              f" | dev sentiment {fmt(dev_m, 'sentiment')} | dev lid {fmt(dev_m, 'lid')} ({(time.time() - t0) / 60:.1f} min)")
        score = dev_m["sentiment"]["macro_f1"]
        if score > best:
            best, best_state, best_epoch = score, trainable_state(model), ep + 1

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    dev_m, _, _ = run_eval(model, loaders["dev"], use_amp, "dev")
    test_m, test_pred, test_lid = run_eval(model, loaders["test"], use_amp, "test")
    test_pred[["id", "predictions"]].to_csv(results_dir / f"{run_name}.csv", index=False)
    if test_lid is not None:
        test_lid.to_csv(results_dir / f"{run_name}_lid.csv", index=False)
    splits["test"][["id", "sentiment"]].to_csv(results_dir / "ground.csv", index=False)
    summary = {"run_name": run_name, "args": vars(args), "model": model.describe(), "best_dev_sentiment_macro_f1": best,
               "dev": dev_m, "test": test_m, "history": history, "train_minutes": (time.time() - t0) / 60}
    (results_dir / f"{run_name}_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[+] TEST sentiment {fmt(test_m, 'sentiment')} | lid {fmt(test_m, 'lid')} | predictions -> {results_dir / (run_name + '.csv')}")
    if getattr(args, "save_model", False):
        save_checkpoint(model, args, run_name, best_epoch, dev_m, test_m)
    return summary
