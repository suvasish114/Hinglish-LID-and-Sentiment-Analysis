"""Joint contrastive learning of LID and sentiment on code-mixed tweets.

One shared encoder (only the last `unfreeze_last` transformer blocks + final norm are trained), two tasks:

  LID (token level)         lid_logits_i = W_l h_i                                    CE over the word sub-words
  sentiment (tweet level)   cross-attention pooling: a learned query q attends over the language-aware token states
                                k_i = v_i = h_i + W_e softmax(lid_logits_i)           (the LID posterior is injected,
                                s = MultiHeadAttention(q, k, v)                        gradients flow back into LID)
                            logits = W_c dropout(s)                                    CE

and two supervised contrastive terms (SupCon, Khosla et al., NeurIPS 2020):

  L_con(Z, y) = - 1/|A| sum_{a in A} 1/|P(a)| sum_{p in P(a)} log  exp(z_a.z_p / tau) / sum_{n != a} exp(z_a.z_n / tau)

  LID        z_i = normalise(g_l(h_i)) of the word sub-words of the batch, y = the LID tag  (same language -> together)
  sentiment  z_b = normalise(g_s(s_b)) of the tweets of the batch,         y = the sentiment label

  L = L_CE^sent + alpha L_CE^lid + lambda_lid L_con^lid + lambda_sent L_con^sent

Variants (ablations) are selected by the constructor flags: `use_lid` off = single-task sentiment model;
`fusion="attn"` = plain attention pooling (no LID injection); `contrastive` in {none, lid, sent, both}.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from lib.data import LID_IGNORE, LID_TAGS

CAUSAL_MODEL_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3", "gemma3_text",
                      "phi", "phi3", "gpt2", "gptj", "gpt_neox", "falcon", "bloom", "opt", "olmo", "olmo2"}
CONTRASTIVE_MODES = ["none", "lid", "sent", "both"]
FUSION_MODES = ["cross_attn", "attn"]


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


def supcon_loss(z, y, tau: float) -> torch.Tensor:
    """Supervised contrastive loss over L2-normalised rows z [N,D] with integer labels y [N].
    Anchors without any positive in the batch are skipped; returns 0 if there is none."""
    z = F.normalize(z.float(), dim=-1)
    n = z.size(0)
    if n < 2:
        return z.new_zeros(())
    sim = z @ z.T / tau                                                     # [N,N]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)              # log softmax over the other rows
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask                   # [N,N] positives
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return z.new_zeros(())
    mean_log_prob_pos = (log_prob.masked_fill(~pos, 0.0).sum(1) / n_pos.clamp(min=1))
    return -mean_log_prob_pos[valid].mean()


class Projector(nn.Module):
    """g(.): 2-layer MLP projection head used by the contrastive losses (as in SimCLR / SupCon)."""

    def __init__(self, dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim))

    def forward(self, x):
        return self.net(x)


class AttentionPooling(nn.Module):
    """Plain attention pooling (baseline of the previous experiment): a_i = softmax_i(w^T h_i + b), s = sum a_i h_i."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w = nn.Linear(hidden, 1)

    def forward(self, h, mask, lid_prob=None):
        scores = self.w(h).squeeze(-1).float().masked_fill(mask == 0, torch.finfo(torch.float32).min)
        attn = torch.softmax(scores, dim=-1)
        return torch.bmm(attn.to(h.dtype).unsqueeze(1), h).squeeze(1), attn


class CrossAttentionPooling(nn.Module):
    """A learned query attends over language-aware token states k_i = v_i = h_i + W_e p_i, where p_i is the LID
    posterior of token i (softmax of the LID logits). W_e is zero-initialised, so at step 0 the module is a
    multi-head attention pooling without language information; the LID gradient reaches the LID head through p_i."""

    def __init__(self, hidden: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.lang_emb = nn.Linear(len(LID_TAGS), hidden, bias=False)
        nn.init.zeros_(self.lang_emb.weight)
        self.norm = nn.LayerNorm(hidden)
        self.mha = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)

    def forward(self, h, mask, lid_prob):
        kv = self.norm(h + self.lang_emb(lid_prob.to(h.dtype)))              # [B,T,H]
        q = self.query.expand(h.size(0), -1, -1).to(kv.dtype)
        s, attn = self.mha(q, kv, kv, key_padding_mask=(mask == 0), need_weights=True)
        return s.squeeze(1), attn.squeeze(1)


class JointModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, use_lid: bool = True, fusion: str = "cross_attn",
                 contrastive: str = "both", proj_dim: int = 128, n_heads: int = 8, unfreeze_last: int = 2,
                 dropout: float = 0.1, dtype=None, device_map=None, max_memory=None, trust_remote_code: bool = False,
                 gradient_checkpointing: bool = False, trainable_fp32: bool = True):
        super().__init__()
        assert fusion in FUSION_MODES and contrastive in CONTRASTIVE_MODES
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
        self.use_lid, self.fusion, self.contrastive = use_lid, fusion, contrastive
        if not use_lid:
            self.fusion = "attn"
            self.contrastive = {"both": "sent", "lid": "none"}.get(contrastive, contrastive)

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

        # ---- heads (on the device of the last block when the encoder is sharded) ---------------------------
        dev = next(self.layers[-1].parameters()).device
        self.dropout = nn.Dropout(dropout)
        self.lid_head = nn.Linear(hidden, len(LID_TAGS)).to(dev) if use_lid else None
        self.lid_proj = Projector(hidden, proj_dim).to(dev) if self.contrastive in ("lid", "both") else None
        self.pooling = (CrossAttentionPooling(hidden, n_heads, dropout) if self.fusion == "cross_attn"
                        else AttentionPooling(hidden)).to(dev)
        self.classifier = nn.Linear(hidden, num_labels).to(dev)
        self.sent_proj = Projector(hidden, proj_dim).to(dev) if self.contrastive in ("sent", "both") else None

    # ------------------------------------------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def head_parameter_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad and not n.startswith("encoder.")]

    def describe(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.trainable_parameters())
        unfrozen = [i for i, l in enumerate(self.layers) if any(p.requires_grad for p in l.parameters())]
        return (f"model_type={self.encoder.config.model_type} layers={len(self.layers)} trainable_blocks={unfrozen} "
                f"lid={self.use_lid} fusion={self.fusion} contrastive={self.contrastive} "
                f"params={total / 1e6:.0f}M trainable={train / 1e6:.1f}M ({100 * train / total:.2f}%)")

    def encode(self, input_ids, attention_mask):
        in_device = self.encoder.get_input_embeddings().weight.device
        return self.encoder(input_ids=input_ids.to(in_device), attention_mask=attention_mask.to(in_device)).last_hidden_state

    def forward(self, input_ids, attention_mask, lid_labels, labels=None, lid_weight: float = 1.0,
                con_weight_lid: float = 0.1, con_weight_sent: float = 0.1, tau: float = 0.1, max_con_tokens: int = 2048):
        dev = self.classifier.weight.device
        h = self.encode(input_ids, attention_mask).to(dev)                   # [B,T,H]
        lid_labels = lid_labels.to(dev)
        word_mask = lid_labels != LID_IGNORE                                 # word sub-words only (no specials / pad)

        lid_logits, lid_prob = None, None
        if self.use_lid:
            lid_logits = self.lid_head(self.dropout(h)).float()              # [B,T,3]
            lid_prob = torch.softmax(lid_logits, dim=-1)
        s, attn = self.pooling(h, word_mask, lid_prob)
        logits = self.classifier(self.dropout(s.to(self.classifier.weight.dtype))).float()

        out = {"logits": logits, "lid_logits": lid_logits, "attention": attn, "loss": None,
               "ce_sent": None, "ce_lid": None, "con_lid": None, "con_sent": None}
        if labels is None:
            return out
        labels = labels.to(dev)
        loss = F.cross_entropy(logits, labels)
        out["ce_sent"] = loss.detach()
        if self.use_lid:
            ce_lid = F.cross_entropy(lid_logits.view(-1, lid_logits.size(-1)), lid_labels.view(-1), ignore_index=LID_IGNORE)
            loss = loss + lid_weight * ce_lid
            out["ce_lid"] = ce_lid.detach()
        if self.training and self.lid_proj is not None and con_weight_lid > 0:
            tok = h[word_mask]                                               # [N,H]
            y = lid_labels[word_mask]
            if tok.size(0) > max_con_tokens:                                 # cap the N x N similarity matrix
                keep = torch.randperm(tok.size(0), device=dev)[:max_con_tokens]
                tok, y = tok[keep], y[keep]
            con_lid = supcon_loss(self.lid_proj(tok.to(self.lid_proj.net[0].weight.dtype)), y, tau)
            loss = loss + con_weight_lid * con_lid
            out["con_lid"] = con_lid.detach()
        if self.training and self.sent_proj is not None and con_weight_sent > 0:
            con_sent = supcon_loss(self.sent_proj(s.to(self.sent_proj.net[0].weight.dtype)), labels, tau)
            loss = loss + con_weight_sent * con_sent
            out["con_sent"] = con_sent.detach()
        out["loss"] = loss
        return out
