"""Sentence-level sentiment classifier on top of any HuggingFace encoder / decoder, with optional
fusion of word-level LID (language identification) tags.

    h = Encoder(x)                      any AutoModel (bidirectional: XLM-R, mBERT, IndicBERT;
                                        causal: Qwen2.5 - the LM head is not loaded)
    s = pool(h)                         `cls` for bidirectional encoders, `last` non-pad token for
                                        causal decoders (same choice as *ForSequenceClassification)
    logits = W_c dropout(s) + b_c

Layer freezing (project hyper-parameter "last 2 layers to unfreeze"): every encoder weight is frozen
except the last `unfreeze_last` transformer blocks, the final norm (if any) and the classifier.
`unfreeze_last=-1` fine-tunes everything (used for the paper baseline reproduction).

LID fusion (Experiment 1). Each sub-word gets the LID tag of its word (special tokens / padding get
a fixed zero vector). The tag is embedded (`lid_dim`, learned) and fused into the hidden state
entering transformer block `lid_inject_layer` (0 = the input embeddings):

    concat :  h_i <- W_f [h_i ; e_i^lid] + b_f     W_f initialised to [I | 0], b_f = 0
    add    :  h_i <- h_i + W_f e_i^lid               W_f initialised to 0

Both initialisations make the fused model *identical to the baseline at step 0*; whatever the model
learns about the LID signal is therefore attributable to the tags. The LID tags come from gold
annotations or from a separately trained tagger that is frozen - no gradient ever reaches it.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from lib.sentimix_data import LID_SPECIAL

LID_FUSIONS = ["none", "concat", "add"]
CAUSAL_MODEL_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3", "gemma3_text",
                      "phi", "phi3", "gpt2", "gptj", "gpt_neox", "falcon", "bloom", "opt", "olmo", "olmo2"}


def find_transformer_layers(encoder) -> nn.ModuleList:
    """The ModuleList of transformer blocks (encoder.encoder.layer for BERT-likes, encoder.layers for
    Llama-likes, ...), located generically so the same code serves every architecture."""
    n = encoder.config.num_hidden_layers
    for _, module in encoder.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n:
            return module
    raise ValueError("could not locate the transformer blocks of this model")


def is_norm(module) -> bool:
    return isinstance(module, nn.LayerNorm) or "norm" in type(module).__name__.lower()


class SentimentClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, pooling: str = "auto", unfreeze_last: int = 2,
                 dropout: float = 0.1, lid_fusion: str = "none", lid_dim: int = 16, lid_inject_layer: int = 0,
                 dtype=None, device_map=None, max_memory=None, trust_remote_code: bool = False,
                 gradient_checkpointing: bool = False, trainable_fp32: bool = True):
        super().__init__()
        assert lid_fusion in LID_FUSIONS, lid_fusion
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
        self.pooling = ("last" if self.is_causal else "cls") if pooling == "auto" else pooling
        hidden = cfg.hidden_size
        self.layers = find_transformer_layers(self.encoder)

        # --- freezing --------------------------------------------------------------------------
        for p in self.encoder.parameters():
            p.requires_grad = unfreeze_last < 0
        if unfreeze_last >= 0:
            for layer in list(self.layers)[len(self.layers) - unfreeze_last:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for name, module in self.encoder.named_children():          # final norm after the blocks
                if is_norm(module):
                    for p in module.parameters():
                        p.requires_grad = True
        if trainable_fp32:                                                # fp32 master weights for what we train
            for p in self.encoder.parameters():
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.float()
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(cfg, "use_cache"):
            cfg.use_cache = False

        # --- LID fusion ------------------------------------------------------------------------
        self.lid_fusion, self.lid_inject_layer = lid_fusion, lid_inject_layer
        self._lid_ids = None
        if lid_fusion != "none":
            assert 0 <= lid_inject_layer < len(self.layers), "lid_inject_layer out of range"
            self.lid_emb = nn.Embedding(LID_SPECIAL + 1, lid_dim, padding_idx=LID_SPECIAL)
            if lid_fusion == "concat":
                self.lid_proj = nn.Linear(hidden + lid_dim, hidden)
                with torch.no_grad():
                    self.lid_proj.weight.zero_()
                    self.lid_proj.weight[:, :hidden] = torch.eye(hidden)
                    self.lid_proj.bias.zero_()
            else:
                self.lid_proj = nn.Linear(lid_dim, hidden, bias=False)
                nn.init.zeros_(self.lid_proj.weight)
            inject_device = next(self.layers[lid_inject_layer].parameters()).device
            self.lid_emb.to(inject_device)
            self.lid_proj.to(inject_device)
            self.layers[lid_inject_layer].register_forward_pre_hook(self._fuse_hook, with_kwargs=True)

        # --- head --------------------------------------------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.classifier.to(next(self.layers[-1].parameters()).device)

    # LID fusion is applied to the hidden states entering block `lid_inject_layer`
    def _fuse(self, hidden_states):
        lid = self._lid_ids.to(hidden_states.device)
        e = self.lid_emb(lid)
        if self.lid_fusion == "concat":
            fused = self.lid_proj(torch.cat([hidden_states.to(self.lid_proj.weight.dtype), e], dim=-1))
        else:
            fused = hidden_states + self.lid_proj(e)
        return fused.to(hidden_states.dtype)

    def _fuse_hook(self, module, args, kwargs):
        if self._lid_ids is None:
            return None
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = self._fuse(kwargs["hidden_states"])
            return args, kwargs
        return (self._fuse(args[0]),) + tuple(args[1:]), kwargs

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def describe(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.trainable_parameters())
        unfrozen = [i for i, l in enumerate(self.layers) if any(p.requires_grad for p in l.parameters())]
        return (f"model_type={self.encoder.config.model_type} pooling={self.pooling} layers={len(self.layers)} "
                f"trainable_blocks={unfrozen} lid_fusion={self.lid_fusion}@{self.lid_inject_layer} "
                f"params={total / 1e6:.0f}M trainable={train / 1e6:.1f}M ({100 * train / total:.2f}%)")

    def forward(self, input_ids, attention_mask, lid_ids=None, labels=None):
        in_device = self.encoder.get_input_embeddings().weight.device
        input_ids, attention_mask = input_ids.to(in_device), attention_mask.to(in_device)
        # kept (not cleared) after the forward pass: gradient checkpointing re-runs the block - and
        # therefore the fusion hook - during backward, and must see the same LID ids
        self._lid_ids = lid_ids if self.lid_fusion != "none" else None
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.to(h.device)
        if self.pooling == "cls":
            pooled = h[:, 0]
        elif self.pooling == "last":                                 # last non-padding token (right padding)
            idx = (mask.sum(dim=1) - 1).clamp(min=0)
            pooled = h[torch.arange(h.size(0), device=h.device), idx]
        elif self.pooling == "mean":
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            raise ValueError(self.pooling)
        pooled = pooled.to(self.classifier.weight.device).to(self.classifier.weight.dtype)
        logits = self.classifier(self.dropout(pooled)).float()
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.to(logits.device))
        return {"loss": loss, "logits": logits}
