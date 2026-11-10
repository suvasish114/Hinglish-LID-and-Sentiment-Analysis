"""Joint language identification and sentiment training.

Run: python3 routing_cross_attention.py xlm-roberta-base
     python3 routing_cross_attention.py IndicBERT-v3-1B --self-test
     python3 routing_cross_attention.py gemma-4-E4B-it
     python3 routing_cross_attention.py Qwen2.5-7B-Instruct

Every block applies a shared adapter, predicts language probabilities, and
softly mixes Hindi/English/other adapters. Gold tags supervise language loss;
they never select adapters. Language supervision uses first subwords so each
retained word counts once. All layers receive language supervision; final-layer
predictions provide word-level metrics. Sentiment uses content-token mean pooling.
Backbones are frozen by default, but gradients traverse their computations.
Joint loss = mean sentiment CE + lid_loss_weight * mean per-layer word CE.
Checkpoints are selected by dev sentiment macro-F1. New joint checkpoints are
incompatible with the previous hard-routed architecture. There is no additional
cross-attention module. Paths are relative to this file.
"""
MODEL_TYPES = {
    "xlm-roberta-base": "xlm-roberta",
    "IndicBERT-v3-1B": "gemma3_text",
    "gemma-4-E4B-it": "gemma4_text",
    "Qwen2.5-7B-Instruct": "qwen2",
}
HUB_MODELS = {
    "gemma-4-E4B-it": "google/gemma-4-E4B-it",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}

HYPERPARAMETERS = {
    "model_path": "models/xlm-roberta-base",  # actual local directory
    "train_csv": "dataset/SentiMix/train.csv",
    "dev_csv": "dataset/SentiMix/dev.csv",
    "test_csv": "dataset/SentiMix/test.csv",
    "output_dir": "model_checkpoints/xlm-roberta-base_joint_routed_adapters",
    "sentence_column": "sentence", "tags_column": "tags",
    "label_column": "sentiment", "id_column": "id",
    "language_to_id": {"hindi": 0, "english": 1, "other": 2},
    "tag_aliases": {"h": "hindi", "hi": "hindi", "hindi": "hindi",
                    "e": "english", "en": "english", "english": "english",
                    "o": "other", "other": "other"},
    "label_to_id": {"negative": 0, "neutral": 1, "positive": 2},
    "train_backbone": False,
    "lid_loss_weight": 1.0,
    "attention_implementation": "eager",  # MPS SDPA cannot train with attention dropout.
    "language_bottleneck": 32, "task_bottleneck": 32,
    "adapter_dropout": 0.2, "classifier_dropout": 0.3,
    "adapter_activation": "relu", "adapter_init_std": 0.02,
    "layer_norm_eps": 1e-5,
    "max_length": 128, "epochs": 10,
    "batch_size": 8, "eval_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "adapter_learning_rate": 1e-4, "backbone_learning_rate": 1e-5,
    "label_smoothing": 0.05, "weight_decay": 0.03, "adam_betas": (0.9, 0.999),
    "adam_epsilon": 1e-8, "warmup_ratio": 0.1,
    "max_grad_norm": 1.0, "early_stopping_patience": 3,
    "seed": 42, "num_workers": 0, "device": "auto",
    "settings_filename": "training_config.json",
    "checkpoint_filename": "best.pt", "history_filename": "history.json",
    "prediction_filename": "test_predictions.csv",
    "test_metrics_filename": "test_metrics.json",
    "loss_plot_filename": "loss.png", "accuracy_plot_filename": "accuracy.png",
    "plot_dpi": 150, "plot_figsize": (8, 5),
}

import argparse
import ast
import csv
import json
import itertools
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, XLMRobertaConfig, XLMRobertaModel, Gemma3TextConfig
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_settings(cfg):
    if not math.isfinite(cfg["lid_loss_weight"]) or cfg["lid_loss_weight"] < 0:
        raise ValueError("lid_loss_weight must be finite and nonnegative")
    if not 0 <= cfg["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if cfg["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    for name in ("epochs", "batch_size", "eval_batch_size",
                 "gradient_accumulation_steps", "language_bottleneck", "task_bottleneck"):
        if cfg[name] < 1:
            raise ValueError(f"{name} must be positive")
    if cfg["max_length"] < 3:
        raise ValueError("max_length must leave space for content and special tokens")
    for name in ("language_to_id", "label_to_id"):
        if sorted(cfg[name].values()) != list(range(len(cfg[name]))):
            raise ValueError(f"{name} must use consecutive IDs starting at zero")
    if set(cfg["language_to_id"]) != {"hindi", "english", "other"}:
        raise ValueError("Exactly Hindi, English and other adapters are required")
    for setting in ("adapter_dropout", "classifier_dropout"):
        if not 0 <= cfg[setting] < 1:
            raise ValueError(f"{setting} must be in [0, 1)")
    if cfg["adapter_activation"] not in {"relu", "gelu"}:
        raise ValueError("adapter_activation must be relu or gelu")


class SentimentDataset(Dataset):
    def __init__(self, path, cfg, require_labels=True):
        self.rows = []
        with open(path, encoding="utf-8", newline="") as stream:
            for number, row in enumerate(csv.DictReader(stream), start=2):
                words = ast.literal_eval(row[cfg["sentence_column"]])
                tags = ast.literal_eval(row[cfg["tags_column"]])
                if (not isinstance(words, list) or not words
                        or not all(isinstance(w, str) for w in words)
                        or not any(w.strip() for w in words)):
                    raise ValueError(f"{path}:{number}: expected a list with nonblank content")
                if not isinstance(tags, list) or len(words) != len(tags):
                    raise ValueError(f"{path}:{number}: word/tag mismatch")
                routes = []
                for tag in tags:
                    key = str(tag).strip().lower()
                    if key not in cfg["tag_aliases"]:
                        raise ValueError(f"{path}:{number}: unknown language tag {tag!r}")
                    routes.append(cfg["language_to_id"][cfg["tag_aliases"][key]])
                label = row.get(cfg["label_column"], "").strip().lower()
                if label and label not in cfg["label_to_id"]:
                    raise ValueError(f"{path}:{number}: unknown sentiment {label!r}")
                if require_labels and not label:
                    raise ValueError(f"{path}:{number}: missing sentiment")
                self.rows.append({"words": words, "routes": routes,
                                  "label": cfg["label_to_id"].get(label),
                                  "id": row.get(cfg["id_column"], str(number - 2))})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Collator:
    def __init__(self, tokenizer, cfg):
        if not tokenizer.is_fast:
            raise ValueError("Fast tokenizer required for word_ids alignment")
        self.tokenizer, self.cfg = tokenizer, cfg

    def __call__(self, rows):
        encoded = self.tokenizer([r["words"] for r in rows], is_split_into_words=True,
                                 padding=True, truncation=True,
                                 max_length=self.cfg["max_length"], return_tensors="pt")
        language_labels = torch.full_like(encoded["input_ids"], -100)
        word_starts = torch.zeros_like(language_labels, dtype=torch.bool)
        word_indices = []
        for i, row in enumerate(rows):
            previous = None
            indices = []
            for j, word in enumerate(encoded.word_ids(batch_index=i)):
                if word is not None:
                    if word != previous:
                        word_starts[i, j] = True
                        indices.append(word)
                        if row["routes"][word] >= 0:
                            language_labels[i, j] = row["routes"][word]
                previous = word
            word_indices.append(indices)
        if (word_starts.sum(dim=1) == 0).any():
            raise ValueError("Tokenization produced an example with no content tokens")
        batch = {"input_ids": encoded["input_ids"],
                 "attention_mask": encoded["attention_mask"],
                 "content_mask": torch.tensor([[w is not None for w in encoded.word_ids(i)]
                                                for i in range(len(rows))], dtype=torch.bool),
                 "language_labels": language_labels, "word_starts": word_starts,
                 "word_indices": word_indices,
                 "ids": [r["id"] for r in rows]}
        if all(r["label"] is not None for r in rows):
            batch["labels"] = torch.tensor([r["label"] for r in rows], dtype=torch.long)
        return batch


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden, bottleneck, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, eps=cfg["layer_norm_eps"])
        self.down = nn.Linear(hidden, bottleneck)
        self.activation = nn.ReLU() if cfg["adapter_activation"] == "relu" else nn.GELU()
        self.dropout = nn.Dropout(cfg["adapter_dropout"])
        self.up = nn.Linear(bottleneck, hidden)
        nn.init.normal_(self.down.weight, std=cfg["adapter_init_std"])
        nn.init.zeros_(self.down.bias)
        # Small nonzero up weights permit gradients to both projections at step 1.
        nn.init.normal_(self.up.weight, std=cfg["adapter_init_std"])
        nn.init.zeros_(self.up.bias)

    def forward(self, states):
        return states + self.dropout(self.up(self.activation(self.down(self.norm(states)))))


class RoutedOutput(nn.Module):
    """Shared adaptation followed by differentiable predicted-language routing."""
    def __init__(self, original, hidden, cfg):
        super().__init__()
        self.original = original
        self.task_adapter = BottleneckAdapter(hidden, cfg["task_bottleneck"], cfg)
        self.language_adapters = nn.ModuleList([
            BottleneckAdapter(hidden, cfg["language_bottleneck"], cfg)
            for _ in cfg["language_to_id"]])
        self.content_mask = None
        self.language_classifier = None  # Non-owning callable supplied per forward.
        self.language_logits = None

    def forward(self, hidden_states, input_tensor):
        states = self.original(hidden_states, input_tensor)
        if self.content_mask is None or self.content_mask.shape != states.shape[:2]:
            raise RuntimeError("Missing content mask or incompatible sequence shape")
        flat = states.reshape(-1, states.shape[-1])
        indices = self.content_mask.reshape(-1).nonzero(as_tuple=True)[0]
        shared = self.task_adapter(flat.index_select(0, indices).to(self.task_adapter.down.weight.dtype))
        logits = self.language_classifier(shared)
        probabilities = logits.softmax(-1)
        mixed = sum(probabilities[:, k:k + 1] * adapter(shared)
                    for k, adapter in enumerate(self.language_adapters))
        self.language_logits = logits.new_zeros((*states.shape[:2], logits.shape[-1]))
        self.language_logits = self.language_logits.reshape(-1, logits.shape[-1]).index_copy(
            0, indices, logits).reshape(*states.shape[:2], logits.shape[-1])
        return flat.index_copy(0, indices, mixed.to(states.dtype)).reshape_as(states)


class IdentityOutput(nn.Module):
    """Adapters receive the completed Gemma block, including its residual."""
    def forward(self, hidden_states, input_tensor=None):
        return hidden_states


def apply_layer_adapters(layer, inputs, output):
    # Transformers 4.x returns tuples; newer versions return a tensor.
    if isinstance(output, tuple):
        return (layer.output(output[0], None),) + output[1:]
    return layer.output(output, None)


def indicbert_class(cfg):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers import AutoConfig
    path = local_path(cfg["model_path"])
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != "gemma3_text":
        raise ValueError("These scripts require the text-only IndicBERT-v3 checkpoint")
    if not (path / "bidirection_gemma3.py").is_file():
        raise ValueError("IndicBERT checkpoint must include bidirection_gemma3.py")
    return get_class_from_dynamic_module(
        "bidirection_gemma3.BidirectionalGemma3Model", str(path), local_files_only=True)


def load_indicbert(cfg):
    # Use the supplied bidirectional backbone directly, without an LM head.
    # The checkpoint declares bfloat16, while adapters/classifier use float32.
    # Use float32 explicitly so backward through the frozen backbone has the
    # same dtype as the trainable modules (including on MPS).
    model = indicbert_class(cfg).from_pretrained(
        local_path(cfg["model_path"]), local_files_only=True,
        dtype=torch.float32,
        attn_implementation=cfg["attention_implementation"])
    model.config.use_cache = False
    return model


def model_name(cfg):
    return cfg.get("model_name", Path(cfg["model_path"]).name)


def model_source(cfg):
    path = local_path(cfg["model_path"])
    if path.is_dir() or cfg["model_path"] not in HUB_MODELS.values():
        return str(path), True
    return cfg["model_path"], False


def load_tokenizer(cfg):
    source, local_only = model_source(cfg)
    options = {"add_prefix_space": True} if model_name(cfg) == "Qwen2.5-7B-Instruct" else {}
    return AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=local_only, **options)


def load_backbone(cfg):
    name = model_name(cfg)
    if name == "IndicBERT-v3-1B":
        return load_indicbert(cfg)
    source, local_only = model_source(cfg)
    options = dict(local_files_only=local_only, attn_implementation=cfg["attention_implementation"])
    if name == "xlm-roberta-base":
        return AutoModel.from_pretrained(source, add_pooling_layer=False, **options)
    if name == "gemma-4-E4B-it":
        from transformers import Gemma4Config, Gemma4Model

        config = Gemma4Config.from_pretrained(source, local_files_only=local_only)
        config.vision_config = None
        config.audio_config = None
        model, loading = Gemma4Model.from_pretrained(
            source, config=config, dtype=torch.bfloat16, output_loading_info=True, **options)
        if loading["missing_keys"] or loading["mismatched_keys"]:
            raise ValueError(f"Gemma 4 text weights did not load completely: {loading}")
        return model.language_model
    model, loading = AutoModel.from_pretrained(
        source, dtype=torch.bfloat16, output_loading_info=True, **options)
    if loading["missing_keys"] or loading["mismatched_keys"]:
        raise ValueError(f"Qwen text weights did not load completely: {loading}")
    return model


class RoutedSentimentModel(nn.Module):
    def __init__(self, cfg, backbone=None):
        super().__init__()
        expected_type = MODEL_TYPES[model_name(cfg)]
        self.backbone = backbone if backbone is not None else load_backbone(cfg)
        if self.backbone.config.model_type != expected_type:
            raise ValueError(f"Expected a {expected_type} backbone")
        self.use_layer_hooks = expected_type != "xlm-roberta"
        max_tokens = self.backbone.config.max_position_embeddings
        if self.use_layer_hooks:
            self.backbone.config.use_cache = False
        else:
            self.backbone.set_attn_implementation(cfg["attention_implementation"])
            max_tokens -= self.backbone.config.pad_token_id + 1
        if cfg["max_length"] > max_tokens:
            raise ValueError(f"max_length exceeds model limit {max_tokens}")
        self.backbone.gradient_checkpointing_disable()
        self.backbone.config.chunk_size_feed_forward = 0
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(cfg["train_backbone"])
        for layer in self.adapter_layers:
            layer.chunk_size_feed_forward = 0
            original = IdentityOutput() if self.use_layer_hooks else layer.output
            layer.output = RoutedOutput(original, self.backbone.config.hidden_size, cfg)
            if self.use_layer_hooks:
                layer.register_forward_hook(apply_layer_adapters)
        self.dropout = nn.Dropout(cfg["classifier_dropout"])
        self.classifier = nn.Linear(self.backbone.config.hidden_size, len(cfg["label_to_id"]))
        self.num_languages = len(cfg["language_to_id"])
        self.language_classifier = nn.Linear(self.backbone.config.hidden_size, self.num_languages)

    @property
    def adapter_layers(self):
        # Return existing modules without registering duplicate state-dict paths.
        return self.backbone.layers if self.use_layer_hooks else self.backbone.encoder.layer

    def forward(self, input_ids, attention_mask, content_mask):
        if content_mask.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
            raise ValueError("IDs, attention masks, and content masks must have identical shapes")
        content = content_mask.bool() & attention_mask.bool()
        if (content.sum(dim=1) == 0).any():
            raise ValueError("Each example must contain at least one content token")
        for layer in self.adapter_layers:
            layer.output.content_mask = content
            # A bound method avoids registering the same classifier in every block.
            layer.output.language_classifier = self.language_classifier.forward
        try:
            states = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                   return_dict=True,
                                   **({"use_cache": False} if self.use_layer_hooks else {})).last_hidden_state
            language_logits = tuple(layer.output.language_logits for layer in self.adapter_layers)
        finally:
            for layer in self.adapter_layers:
                layer.output.content_mask = None
                layer.output.language_classifier = None
                layer.output.language_logits = None
        pooled = (states.to(self.classifier.weight.dtype) * content.unsqueeze(-1)).sum(1) / content.sum(1, keepdim=True)
        return {"sentiment_logits": self.classifier(self.dropout(pooled)),
                "language_logits": language_logits}

    @torch.no_grad()
    def predict(self, words, tokenizer, cfg, device):
        """Predict sentiment and retained word tags without input language tags.

        Truncation may omit words beyond max_length; returned word indices refer
        to the supplied words and only cover retained first subwords.
        """
        batch = Collator(tokenizer, cfg)([{"words": words, "routes": [-1] * len(words),
                                         "label": None, "id": "inference"}])
        was_training = self.training
        self.eval()
        try:
            output = self(**model_inputs(batch, device))
            probabilities = output["sentiment_logits"].softmax(-1)[0].cpu().tolist()
            tags = output["language_logits"][-1][0].argmax(-1).cpu()[batch["word_starts"][0]].tolist()
        finally:
            self.train(was_training)
        labels = {v: k for k, v in cfg["label_to_id"].items()}
        languages = {v: k for k, v in cfg["language_to_id"].items()}
        return {"sentiment": {labels[i]: p for i, p in enumerate(probabilities)},
                "language_tags": [languages[k] for k in tags],
                "word_indices": batch["word_indices"][0]}


def training_loss(logits, labels, cfg):
    """Sum smoothed CE; the training loop normalizes by accumulated examples."""
    return nn.functional.cross_entropy(
        logits, labels, reduction="sum", label_smoothing=cfg["label_smoothing"])


def model_inputs(batch, device):
    return {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "content_mask")}


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def language_loss(output, labels):
    """Summed first-subword CE, averaged across layers (not across words)."""
    return sum(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           labels.reshape(-1), ignore_index=-100, reduction="sum")
               for logits in output["language_logits"]) / len(output["language_logits"])


def classification_metrics(truth, predictions, names):
    label_ids = list(range(len(names)))
    return {"accuracy": accuracy_score(truth, predictions),
            "macro_f1": f1_score(truth, predictions, labels=label_ids, average="macro", zero_division=0),
            "per_class": classification_report(truth, predictions, labels=label_ids,
                target_names=[k for k, v in sorted(names.items(), key=lambda x: x[1])],
                output_dict=True, zero_division=0)}


@torch.no_grad()
def evaluate(model, loader, device, cfg, prediction_path=None):
    model.eval()
    truth, predictions, ids, language_truth, language_predictions = [], [], [], [], []
    sentiment_sum, lid_sum, sentence_count, word_count = 0.0, 0.0, 0, 0
    prediction_rows = []
    language_names = {v: k for k, v in cfg["language_to_id"].items()}
    for batch in loader:
        output = model(**model_inputs(batch, device))
        logits = output["sentiment_logits"]
        predictions.extend(logits.argmax(-1).cpu().tolist())
        ids.extend(batch["ids"])
        tags = batch["language_labels"].to(device)
        predicted_tags = output["language_logits"][-1].argmax(-1)
        valid = tags != -100
        lid_sum += language_loss(output, tags).item()
        word_count += valid.sum().item()
        language_truth.extend(tags[valid].cpu().tolist())
        language_predictions.extend(predicted_tags[valid].cpu().tolist())
        for i, identifier in enumerate(batch["ids"]):
            starts = batch["word_starts"][i]
            values = predicted_tags[i].cpu()[starts].tolist()
            prediction_rows.append((identifier, json.dumps(batch["word_indices"][i]),
                                    json.dumps([language_names[k] for k in values])))
        if "labels" in batch:
            labels = batch["labels"].to(device)
            sentiment_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            sentence_count += len(labels)
            truth.extend(labels.cpu().tolist())
    metrics = {"language": classification_metrics(language_truth, language_predictions, cfg["language_to_id"])}
    metrics["language"]["loss"] = lid_sum / word_count
    if truth:
        if len(truth) != len(predictions):
            raise ValueError("Evaluation split has partially missing sentiment labels")
        metrics.update(classification_metrics(truth, predictions, cfg["label_to_id"]))
        metrics["loss"] = sentiment_sum / sentence_count
        metrics["joint_loss"] = metrics["loss"] + cfg["lid_loss_weight"] * metrics["language"]["loss"]
    if prediction_path is not None:
        with Path(prediction_path).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([cfg["id_column"], "word_indices", "predicted_language_tags"])
            writer.writerows(prediction_rows)
    return metrics, ids, predictions


def plot_training_history(history, output, cfg):
    """Save paired learning curves for every completed epoch, including early stopping."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print("Skipping plots: install matplotlib to generate learning curves.", flush=True)
        return

    epochs = [row["epoch"] for row in history]
    curves = [
        ("Loss", [row["train_loss"] for row in history],
         [row["dev"]["loss"] for row in history], cfg["loss_plot_filename"]),
        ("Accuracy", [row["train_accuracy"] for row in history],
         [row["dev"]["accuracy"] for row in history], cfg["accuracy_plot_filename"]),
    ]
    curves.extend([
        ("Language loss", [row["train_language"]["loss"] for row in history],
         [row["dev"]["language"]["loss"] for row in history], "language_loss.png"),
        ("Language accuracy", [row["train_language"]["accuracy"] for row in history],
         [row["dev"]["language"]["accuracy"] for row in history], "language_accuracy.png"),
        ("Joint loss", [row["train_joint_loss"] for row in history],
         [row["dev"]["joint_loss"] for row in history], "joint_loss.png"),
    ])
    for metric, training, validation, filename in curves:
        fig, ax = plt.subplots(figsize=cfg["plot_figsize"])
        ax.plot(epochs, training, marker="o", label="Train")
        ax.plot(epochs, validation, marker="o", label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric, title=f"Train vs validation {metric.lower()}")
        ax.set_xticks(epochs)
        if "accuracy" in metric.lower():
            ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=cfg["plot_dpi"])
        plt.close(fig)
        print(f"Saved {metric.lower()} plot: {path}", flush=True)


def main(cfg):
    validate_settings(cfg)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    device = choose_device(cfg["device"])
    output = local_path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / cfg["settings_filename"]).write_text(json.dumps(cfg, indent=2))
    tokenizer = load_tokenizer(cfg)
    collator = Collator(tokenizer, cfg)
    generator = torch.Generator().manual_seed(cfg["seed"])
    datasets = {name: SentimentDataset(local_path(cfg[f"{name}_csv"]), cfg,
                                       require_labels=name != "test")
                for name in ("train", "dev", "test")}
    if any(len(d) == 0 for d in datasets.values()):
        raise ValueError("Train/dev/test splits must be nonempty")
    test_labeled = [r["label"] is not None for r in datasets["test"].rows]
    if any(test_labeled) and not all(test_labeled):
        raise ValueError("Test labels must be present for all rows or none")
    loaders = {name: DataLoader(dataset, batch_size=cfg["batch_size"] if name == "train"
                               else cfg["eval_batch_size"], shuffle=name == "train",
                               collate_fn=collator, num_workers=cfg["num_workers"],
                               generator=generator) for name, dataset in datasets.items()}
    model = RoutedSentimentModel(cfg).to(device)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        adapter = "language_adapters" in name or "task_adapter" in name or name.startswith(("classifier.", "language_classifier."))
        lr = cfg["adapter_learning_rate"] if adapter else cfg["backbone_learning_rate"]
        decay = 0.0 if parameter.ndim < 2 else cfg["weight_decay"]
        groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": decay}
                                  for (lr, decay), params in groups.items()],
                                 betas=cfg["adam_betas"], eps=cfg["adam_epsilon"])
    accumulation = cfg["gradient_accumulation_steps"]
    total_steps = math.ceil(len(loaders["train"]) / accumulation) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer,
                int(total_steps * cfg["warmup_ratio"]), total_steps)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}; trainable: {trainable:,}; total: {sum(p.numel() for p in model.parameters()):,}")
    best, stale, history = -1.0, 0, []
    checkpoint_path = output / cfg["checkpoint_filename"]
    tokenizer.save_pretrained(output / "tokenizer")
    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_sum, train_count, train_correct = 0.0, 0, 0
        lid_sum, word_count, lid_correct = 0.0, 0, 0
        lid_truth, lid_predictions = [], []
        iterator = iter(loaders["train"])
        progress = tqdm(total=len(loaders["train"]), desc=f"Epoch {epoch + 1}")
        while True:
            # Buffer input batches only; release each computation graph after backward.
            window = list(itertools.islice(iterator, accumulation))
            if not window:
                break
            sentences = sum(len(batch["labels"]) for batch in window)
            words = sum((batch["language_labels"] != -100).sum().item() for batch in window)
            for batch in window:
                labels = batch["labels"].to(device)
                tags = batch["language_labels"].to(device)
                output_batch = model(**model_inputs(batch, device))
                logits = output_batch["sentiment_logits"]
                sentiment_loss = training_loss(logits, labels, cfg)
                lid_loss = language_loss(output_batch, tags)
                loss = sentiment_loss / sentences + cfg["lid_loss_weight"] * lid_loss / words
                loss.backward()
                train_sum += sentiment_loss.item()
                train_count += len(labels)
                train_correct += (logits.detach().argmax(-1) == labels).sum().item()
                valid = tags != -100
                predicted = output_batch["language_logits"][-1].detach().argmax(-1)
                lid_sum += lid_loss.item()
                word_count += valid.sum().item()
                lid_correct += (predicted[valid] == tags[valid]).sum().item()
                lid_truth.extend(tags[valid].cpu().tolist())
                lid_predictions.extend(predicted[valid].cpu().tolist())
                progress.update(1)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        progress.close()
        metrics, _, _ = evaluate(model, loaders["dev"], device, cfg)
        train_language = classification_metrics(lid_truth, lid_predictions, cfg["language_to_id"])
        train_language["loss"] = lid_sum / word_count
        history.append({"epoch": epoch + 1, "train_loss": train_sum / train_count,
                        "train_accuracy": train_correct / train_count,
                        "train_language": train_language,
                        "train_joint_loss": train_sum / train_count + cfg["lid_loss_weight"] * lid_sum / word_count,
                        "dev": metrics})
        (output / cfg["history_filename"]).write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch + 1}/{cfg['epochs']} | "
              f"train sentiment loss/accuracy: {train_sum / train_count:.4f}/{train_correct / train_count:.4f}; "
              f"language loss/accuracy/F1: {lid_sum / word_count:.4f}/{lid_correct / word_count:.4f}/{train_language['macro_f1']:.4f}; "
              f"joint loss: {history[-1]['train_joint_loss']:.4f} | "
              f"dev sentiment loss/accuracy/F1: {metrics['loss']:.4f}/{metrics['accuracy']:.4f}/{metrics['macro_f1']:.4f}; "
              f"language loss/accuracy/F1: {metrics['language']['loss']:.4f}/{metrics['language']['accuracy']:.4f}/{metrics['language']['macro_f1']:.4f}; "
              f"joint loss: {metrics['joint_loss']:.4f}", flush=True)
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            # Adapter-only checkpoints rely on the unchanged local base checkpoint.
            names = {name for name, p in model.named_parameters() if p.requires_grad}
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                     if cfg["train_backbone"] or k in names}
            torch.save({"model": state, "settings": cfg, "epoch": epoch + 1,
                        "dev": metrics, "base_model_path": model_source(cfg)[0]},
                       checkpoint_path)
        else:
            stale += 1
            if cfg["early_stopping_patience"] and stale >= cfg["early_stopping_patience"]:
                break
    plot_training_history(history, output, cfg)
    load_checkpoint(model, checkpoint_path, device)
    metrics, ids, predictions = evaluate(model, loaders["test"], device, cfg, output / "test_language_predictions.csv")
    label_names = {v: k for k, v in cfg["label_to_id"].items()}
    with (output / cfg["prediction_filename"]).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([cfg["id_column"], "predicted_sentiment"])
        writer.writerows((i, label_names[p]) for i, p in zip(ids, predictions))
    (output / cfg["test_metrics_filename"]).write_text(json.dumps(metrics, indent=2))
    print(f"Best dev macro-F1: {best:.4f}; test metrics: {metrics}; outputs: {output}")


def load_checkpoint(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    required = {name for name, p in model.named_parameters() if p.requires_grad}
    if incompatible.unexpected_keys or required.intersection(incompatible.missing_keys):
        raise ValueError("Checkpoint does not match the configured adapter architecture")
    return checkpoint


def self_test(cfg):
    """Check subword routing, actual gradients, masking, checkpointing, and training modes."""
    import tempfile
    validate_settings(cfg)
    # Independently verify the smoothed targets and summed loss used in training.
    probe = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 0.2, 1.5]], requires_grad=True)
    labels = torch.tensor([0, 2])
    targets = nn.functional.one_hot(labels, num_classes=3).float()
    targets = targets * (1 - cfg["label_smoothing"]) + cfg["label_smoothing"] / 3
    expected_loss = -(targets * probe.log_softmax(-1)).sum()
    assert torch.allclose(training_loss(probe, labels, cfg), expected_loss)
    training_loss(probe, labels, cfg).backward()
    assert torch.isfinite(probe.grad).all()
    torch.manual_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    collator = Collator(tokenizer, cfg)
    rows = [{"words": ["namaste", "excellent", "!"], "routes": [0, 1, 2], "label": 2, "id": "a"},
            {"words": ["bad"], "routes": [1], "label": 0, "id": "b"}]
    batch = collator(rows)
    encoded = tokenizer([r["words"] for r in rows], is_split_into_words=True,
                        padding=True, return_tensors="pt")
    assert (batch["language_labels"] != -100).sum().item() == 4
    for i, row in enumerate(rows):
        previous = None
        for j, word in enumerate(encoded.word_ids(i)):
            target = row["routes"][word] if word is not None and word != previous else -100
            assert batch["language_labels"][i, j].item() == target
            assert batch["content_mask"][i, j].item() == (word is not None)
            previous = word
    name = model_name(cfg)
    if name == "IndicBERT-v3-1B":
        tiny = Gemma3TextConfig(vocab_size=32, hidden_size=24, intermediate_size=48,
                               num_hidden_layers=2, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=6,
                               max_position_embeddings=514, pad_token_id=0,
                               attention_dropout=0.0, use_cache=False)
        tiny._attn_implementation = cfg["attention_implementation"]
        make_backbone = lambda: indicbert_class(cfg)(tiny)
    elif name == "gemma-4-E4B-it":
        from transformers import Gemma4TextConfig, Gemma4TextModel

        tiny = Gemma4TextConfig(vocab_size=32, vocab_size_per_layer_input=32,
                                hidden_size=24, hidden_size_per_layer_input=8,
                                intermediate_size=48, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2,
                                head_dim=8, global_head_dim=8, num_kv_shared_layers=1,
                                layer_types=["full_attention"] * 2, pad_token_id=0)
        tiny._attn_implementation = cfg["attention_implementation"]
        make_backbone = lambda: Gemma4TextModel(tiny)
    elif name == "Qwen2.5-7B-Instruct":
        from transformers import Qwen2Config, Qwen2Model

        tiny = Qwen2Config(vocab_size=32, hidden_size=24, intermediate_size=48,
                           num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, pad_token_id=0)
        tiny._attn_implementation = cfg["attention_implementation"]
        make_backbone = lambda: Qwen2Model(tiny)
    else:
        tiny = XLMRobertaConfig(vocab_size=32, hidden_size=24, intermediate_size=48,
                               num_hidden_layers=2, num_attention_heads=4,
                               max_position_embeddings=514, pad_token_id=1,
                               hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        make_backbone = lambda: XLMRobertaModel(tiny, add_pooling_layer=False)
    model = RoutedSentimentModel(cfg, make_backbone())
    # Keep the real alignment but replace vocabulary IDs for a small test network.
    batch["input_ids"] = batch["input_ids"].remainder(30) + 2
    batch["input_ids"].masked_fill_(~batch["attention_mask"].bool(), tiny.pad_token_id)
    output = model(**model_inputs(batch, "cpu"))
    loss = (training_loss(output["sentiment_logits"], batch["labels"], cfg) / 2
            + language_loss(output, batch["language_labels"]) / 4)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name
    model.eval()
    # Gold targets cannot influence either prediction head.
    changed = dict(batch, language_labels=(batch["language_labels"] + 1).remainder(3))
    expected = model(**model_inputs(batch, "cpu"))
    actual = model(**model_inputs(changed, "cpu"))
    assert torch.equal(expected["sentiment_logits"], actual["sentiment_logits"])
    assert torch.equal(expected["language_logits"][-1], actual["language_logits"][-1])
    # Each task independently updates shared adapters; sentiment also trains routing.
    for task in ("sentiment", "language"):
        model.zero_grad(set_to_none=True)
        result = model(**model_inputs(batch, "cpu"))
        loss = (training_loss(result["sentiment_logits"], batch["labels"], cfg)
                if task == "sentiment" else language_loss(result, batch["language_labels"]))
        loss.backward()
        for layer in model.adapter_layers:
            assert layer.output.task_adapter.down.weight.grad.abs().sum() > 0
        assert model.language_classifier.weight.grad.abs().sum() > 0
    # Adapter bypass and exact weighted mixture, including non-content positions.
    layer = model.adapter_layers[0].output
    hidden = torch.randn(1, 4, 24 if model.use_layer_hooks else 48)
    residual = torch.randn(1, 4, 24)
    layer.content_mask = torch.tensor([[True, True, True, False]])
    layer.language_classifier = model.language_classifier.forward
    before = layer.original(hidden, residual)
    after = layer(hidden, residual)
    assert torch.equal(before[:, -1], after[:, -1])
    shared = layer.task_adapter(before[:, :3])
    weights = model.language_classifier(shared).softmax(-1)
    mixture = sum(weights[..., k:k + 1] * adapter(shared)
                  for k, adapter in enumerate(layer.language_adapters))
    assert torch.allclose(after[:, :3], mixture)
    layer.content_mask = layer.language_classifier = layer.language_logits = None
    # Independent normalization equals full-batch gradients for unequal word counts.
    model.zero_grad(set_to_none=True)
    full = model(**model_inputs(batch, "cpu"))
    (training_loss(full["sentiment_logits"], batch["labels"], cfg) / 2
     + language_loss(full, batch["language_labels"]) / 4).backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for i in range(2):
        single = {k: v[i:i+1] for k, v in model_inputs(batch, "cpu").items()}
        result = model(**single)
        (training_loss(result["sentiment_logits"], batch["labels"][i:i+1], cfg) / 2
         + language_loss(result, batch["language_labels"][i:i+1]) / 4).backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            assert torch.allclose(parameter.grad, gradients[name], atol=2e-6, rtol=1e-4), name
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"
        state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"model": state}, path)
        with torch.no_grad():
            model.classifier.weight.add_(1)
        load_checkpoint(model, path)
        restored = model(**model_inputs(batch, "cpu"))
        assert torch.allclose(expected["sentiment_logits"], restored["sentiment_logits"])
        assert torch.allclose(expected["language_logits"][-1], restored["language_logits"][-1])
        metrics, _, _ = evaluate(model, [batch], "cpu", cfg, Path(temporary) / "tags.csv")
        assert "joint_loss" in metrics and "macro_f1" in metrics["language"]
    joint_cfg = dict(cfg, train_backbone=True)
    joint = RoutedSentimentModel(joint_cfg, make_backbone())
    joint(**model_inputs(batch, "cpu"))["sentiment_logits"].sum().backward()
    assert joint.backbone.get_input_embeddings().weight.grad is not None
    # Actual tokenizer inference alignment without any supplied language tags.
    untagged = collator([dict(rows[0], routes=[-1] * 3)])
    assert untagged["word_starts"].sum() == 3
    assert (untagged["language_labels"] == -100).all()
    for split in ("train", "dev", "test"):
        data = SentimentDataset(local_path(cfg[f"{split}_csv"]), cfg, split != "test")
        for start in range(0, len(data), cfg["eval_batch_size"]):
            collator(data.rows[start:start + cfg["eval_batch_size"]])
        print(f"Validated {split}: {len(data)} rows and subword alignment")
    print("Self-test passed: joint gradients, predicted routing, no target leakage, bypass, accumulation, alignment, save/load, and backbone tuning.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    models = (*MODEL_TYPES, *HUB_MODELS.values())
    parser.add_argument("model_name", nargs="?", choices=models, help="Backbone name or supported Hub ID")
    parser.add_argument("--model-name", "--model", dest="model_option", choices=models)
    parser.add_argument("--self-test", action="store_true", help="Test tiny models and validate dataset alignment")
    parser.add_argument("--model-path", help="Use an existing local checkpoint directory")
    parser.add_argument("--train-csv", help="Training CSV path")
    parser.add_argument("--dev-csv", help="Validation CSV path")
    parser.add_argument("--test-csv", help="Test CSV path")
    parser.add_argument("--output-dir", help="Checkpoint and metrics directory")
    parser.add_argument("--lid-loss-weight", type=float, default=None, help="Language loss multiplier (default: 1)")
    parser.add_argument("--device", help="auto, cpu, cuda, or mps")
    args = parser.parse_args(argv)
    hub_names = {repo: name for name, repo in HUB_MODELS.items()}
    positional = hub_names.get(args.model_name, args.model_name)
    option = hub_names.get(args.model_option, args.model_option)
    if positional and option and positional != option:
        parser.error("Conflicting model names")
    name = option or positional
    if name is None:
        parser.error("A model name is required")
    cfg = dict(HYPERPARAMETERS)
    cfg["model_name"] = name
    local_model = f"models/{name}"
    cfg["model_path"] = args.model_path or (
        local_model if local_path(local_model).is_dir() else HUB_MODELS.get(name, local_model))
    cfg["output_dir"] = f"model_checkpoints/{name}_joint_routed_adapters"
    for key in ("train_csv", "dev_csv", "test_csv", "output_dir", "device", "lid_loss_weight"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return args, cfg


if __name__ == "__main__":
    args, cfg = parse_args()
    if args.self_test:
        self_test(cfg)
    else:
        main(cfg)
