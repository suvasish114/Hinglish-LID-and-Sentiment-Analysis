"""Experiment 2 - sentiment model with switch-point information. One class per idea, nothing mixed:

  SwitchAttentionPooling      approaches 1-3 (and the pooling used by approach 5)
      none     (A1)  a_i = softmax_i( w^T h_i + b )                           plain attention pooling, no switch input
      binary   (A2)  a_i = softmax_i( W_h h_i + W_s e(sw_i) + b ),  sw_i in {0, 1}
      distance (A3)  a_i = softmax_i( W_h h_i + W_s e(d_i) + b ),   d_i = min(distance to nearest switch, 3)
                     sentiment vector s = sum_i a_i h_i ,  logits = W_c dropout(s) + b_c
  ContrastiveDistillation     approach 4 - LASER3-CO (Tan et al., EACL 2023) adapted to code-mixed tweets
  SentimentModel              encoder + pooling + classifier (+ contrastive loss); only the last `unfreeze_last`
                              transformer blocks and the final norm of the encoder are trained

Approach 5 (Beyond Detection) is not a model variant: it is approach 2/3 pooling fed with switch points predicted by
lib/switch_predictor.py instead of switch points derived from LID tags.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

POOLING_MODES = ["none", "binary", "distance"]
N_DISTANCE_BUCKETS = 4          # 0, 1, 2, 3+
DIST_PAD, BIN_PAD = 3, 0        # feature value given to special tokens / padding
CAUSAL_MODEL_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3", "gemma3_text",
                      "phi", "phi3", "gpt2", "gptj", "gpt_neox", "falcon", "bloom", "opt", "olmo", "olmo2"}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_transformer_layers(encoder) -> nn.ModuleList:
    """The ModuleList of transformer blocks, located generically (BERT: encoder.layer, Llama/Qwen: layers)."""
    n = encoder.config.num_hidden_layers
    for _, module in encoder.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n:
            return module
    raise ValueError("could not locate the transformer blocks of this model")


def is_norm(module) -> bool:
    return isinstance(module, nn.LayerNorm) or "norm" in type(module).__name__.lower()


class SwitchAttentionPooling(nn.Module):
    """Approaches 1-3."""

    def __init__(self, hidden_size: int, mode: str, switch_dim: int = 16):
        super().__init__()
        assert mode in POOLING_MODES, mode
        self.mode = mode
        self.w_h = nn.Linear(hidden_size, 1)                                    # W_h h_i + b
        if mode == "none":
            self.switch_emb, self.w_s = None, None
        else:
            n_values = 2 if mode == "binary" else N_DISTANCE_BUCKETS
            self.switch_emb = nn.Embedding(n_values, switch_dim)                 # e(sw_i) or e(d_i)
            self.w_s = nn.Linear(switch_dim, 1, bias=False)                     # W_s
            nn.init.zeros_(self.w_s.weight)                                     # identical to approach 1 at step 0

    def forward(self, h, pool_mask, switch_feat):
        """h [B,T,H]; pool_mask [B,T] (1 = word sub-token); switch_feat [B,T] long."""
        scores = self.w_h(h).squeeze(-1).float()                                # [B,T]
        if self.switch_emb is not None:
            scores = scores + self.w_s(self.switch_emb(switch_feat.to(h.device))).squeeze(-1).float()
        scores = scores.masked_fill(pool_mask.to(h.device) == 0, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)                                    # a_i
        pooled = torch.bmm(attn.to(h.dtype).unsqueeze(1), h).squeeze(1)         # s = sum_i a_i h_i
        return pooled, attn


class ContrastiveDistillation(nn.Module):
    """Approach 4 - contrastive distillation of LASER3-CO (Tan, Heffernan, Schwenk, Koehn, EACL 2023), eq. (1)/(2):

        L = - log  exp(q . k+ / tau) / ( exp(q . k+ / tau) + sum_{i in S} exp(q . k_i / tau) )

    q   = L2-normalised student representation of the full code-mixed tweet (the sentiment vector s)
    k+  = L2-normalised FROZEN-teacher embedding of a positive view of the same tweet
    k_i = teacher embeddings of other tweets' views kept in a FIFO queue of size N (N = 4096 in the paper), filled
          with the previous batches (MoCo-style, no momentum teacher)
    S   = all queue entries, or - LASER3-CO-Filter - only those with cos(k+, k_i) < sigma (sigma = 0.9 in the paper),
          which removes "extremely hard" negatives that are near-duplicates of the positive
    tau = 0.05 as in the paper.

    Adaptation (SentiMix has no parallel sentences): the paper's (source sentence, target translation) pair becomes
    (code-mixed tweet, one of its monolingual language views), where a language view is the sequence of the words of
    the tweet's Hindi (or English) segments, i.e. the maximal runs delimited by the switch points
    (lib/switch.language_view). The teacher is the pre-trained encoder itself before fine-tuning, mean-pooled; its
    embeddings are pre-computed once, so it is frozen by construction. The loss is added to the sentiment
    cross-entropy with weight --contrastive_weight.
    """

    def __init__(self, dim: int, queue_size: int = 4096, tau: float = 0.05, filter_sigma=None):
        super().__init__()
        self.tau, self.filter_sigma, self.queue_size = tau, filter_sigma, queue_size
        self.register_buffer("queue", torch.zeros(queue_size, dim))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long))
        # Mean-pooled encoder states are anisotropic (cosine > 0.9 between any two tweets), which makes the
        # InfoNCE similarities uninformative and the sigma filter remove everything. Both q and k are therefore
        # centred with the mean of the pre-computed teacher embeddings (set by train.py) before normalisation.
        self.register_buffer("center", torch.zeros(dim))

    @torch.no_grad()
    def set_center(self, mu):
        self.center.copy_(mu.to(self.center.device).float())

    @torch.no_grad()
    def _enqueue(self, k):
        n = k.size(0)
        idx = (self.ptr + torch.arange(n, device=k.device)) % self.queue_size
        self.queue[idx] = k.to(self.queue.dtype)
        self.ptr.fill_((self.ptr + n) % self.queue_size)
        self.filled.fill_(min(int(self.filled) + n, self.queue_size))

    def forward(self, q, k_pos):
        q = F.normalize(q.float() - self.center, dim=-1)
        k_pos = F.normalize(k_pos.to(q.device).float() - self.center, dim=-1)
        l_pos = (q * k_pos).sum(-1, keepdim=True)                               # [B,1]
        n = int(self.filled)
        if n == 0:                                                              # first batch: no negatives yet
            self._enqueue(k_pos)
            return q.new_zeros(())
        neg = self.queue[:n]                                                    # [N,H]
        l_neg = q @ neg.T                                                       # [B,N]
        if self.filter_sigma is not None:
            l_neg = l_neg.masked_fill((k_pos @ neg.T) >= self.filter_sigma, float("-inf"))
        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau
        loss = F.cross_entropy(logits, torch.zeros(q.size(0), dtype=torch.long, device=q.device))
        self._enqueue(k_pos)
        return loss


class SentimentModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, pooling: str = "none", switch_dim: int = 16,
                 unfreeze_last: int = 2, dropout: float = 0.1, dtype=None, device_map=None, max_memory=None,
                 trust_remote_code: bool = False, gradient_checkpointing: bool = False, trainable_fp32: bool = True,
                 contrastive: bool = False, contrastive_queue: int = 4096, contrastive_tau: float = 0.05,
                 contrastive_filter=None):
        super().__init__()
        kwargs = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
            if max_memory:
                kwargs["max_memory"] = max_memory
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code, **kwargs)
        cfg = self.encoder.config
        self.is_causal = cfg.model_type in CAUSAL_MODEL_TYPES or bool(getattr(cfg, "is_decoder", False))
        hidden = cfg.hidden_size
        self.layers = find_transformer_layers(self.encoder)

        # ---- freezing: everything except the last `unfreeze_last` blocks (+ final norm) -------------------
        for p in self.encoder.parameters():
            p.requires_grad = unfreeze_last < 0
        if unfreeze_last >= 0:
            for layer in list(self.layers)[len(self.layers) - unfreeze_last:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for _, module in self.encoder.named_children():
                if is_norm(module):
                    for p in module.parameters():
                        p.requires_grad = True
        if trainable_fp32:
            for p in self.encoder.parameters():
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.float()
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(cfg, "use_cache"):
            cfg.use_cache = False

        # ---- heads ----------------------------------------------------------------------------------------
        out_device = next(self.layers[-1].parameters()).device
        self.pooling = SwitchAttentionPooling(hidden, pooling, switch_dim).to(out_device)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels).to(out_device)
        self.contrastive = ContrastiveDistillation(hidden, contrastive_queue, contrastive_tau,
                                                   contrastive_filter).to(out_device) if contrastive else None

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def describe(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.trainable_parameters())
        unfrozen = [i for i, l in enumerate(self.layers) if any(p.requires_grad for p in l.parameters())]
        return (f"model_type={self.encoder.config.model_type} pooling={self.pooling.mode} layers={len(self.layers)} "
                f"trainable_blocks={unfrozen} contrastive={self.contrastive is not None} "
                f"params={total / 1e6:.0f}M trainable={train / 1e6:.1f}M ({100 * train / total:.2f}%)")

    def encode(self, input_ids, attention_mask):
        in_device = self.encoder.get_input_embeddings().weight.device
        return self.encoder(input_ids=input_ids.to(in_device), attention_mask=attention_mask.to(in_device)).last_hidden_state

    @torch.no_grad()
    def teacher_embed(self, input_ids, attention_mask, pool_mask):
        """Frozen-teacher sentence embedding for approach 4: mean of the word sub-token states of the
        encoder as loaded (call before any training step)."""
        h = self.encode(input_ids, attention_mask)
        m = pool_mask.to(h.device).unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1).clamp(min=1.0)).float()

    def forward(self, input_ids, attention_mask, pool_mask, switch_feat, labels=None,
                teacher_embed=None, contrastive_weight: float = 0.0):
        # with device_map the encoder output comes back on the input device: move it to the heads' device
        h = self.encode(input_ids, attention_mask).to(self.pooling.w_h.weight.device)
        s, attn = self.pooling(h, pool_mask, switch_feat)
        s = s.to(self.classifier.weight.dtype)
        logits = self.classifier(self.dropout(s)).float()
        out = {"logits": logits, "attention": attn, "loss": None, "ce_loss": None, "co_loss": None}
        if labels is not None:
            ce = F.cross_entropy(logits, labels.to(logits.device))
            out["loss"], out["ce_loss"] = ce, ce.detach()
            if self.contrastive is not None and teacher_embed is not None and self.training:
                co = self.contrastive(s, teacher_embed)
                out["loss"] = ce + contrastive_weight * co
                out["co_loss"] = co.detach()
        return out
