
"""
Bidirectional Gemma3 Architecture.
This file defines a Gemma3 model with non-causal (bidirectional) attention
intended for masked language modeling or similar tasks.

Masking backend is chosen at import time from the installed transformers version:
  - 4.50.x-4.52.x: legacy `_update_causal_mask` (called by parent forward)
  - 4.53.x-4.56.x: custom v4 forward + `create_causal_mask` with bidirectional or-mask
  - 4.57.x-4.x:   parent `Gemma3TextModel.forward` + `use_bidirectional_attention`
  - 5.0.0+:       custom v5 forward + `create_bidirectional_mask`
"""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers
from packaging import version

from transformers import Gemma3ForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3TextModel,
)

try:
    from transformers.models.gemma3.modeling_gemma3 import _bidirectional_window_overlay
except ImportError:
    _bidirectional_window_overlay = None

_TF_VERSION = version.parse(transformers.__version__)

create_causal_mask = None
create_sliding_window_causal_mask = None
create_bidirectional_mask = None
DynamicCache = None
capture_outputs = None

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    pass

try:
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    pass

try:
    from transformers.masking_utils import create_bidirectional_mask
except ImportError:
    pass

try:
    from transformers.utils.output_capturing import capture_outputs
except ImportError:
    pass


def _identity_capture(func=None, **_kwargs):
    if func is None:
        return lambda f: f
    return func


_CAPTURE = capture_outputs if capture_outputs is not None else _identity_capture


def _configure_bidirectional_config(config) -> None:
    if config is None:
        return
    config.use_bidirectional_attention = True
    config.is_causal = False


def _get_text_config(config):
    """Use nested text_config for multimodal Gemma3 wrappers (e.g. IndicBERT-v3-4B)."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(config, "model_type", None) == "gemma3":
        return text_config
    return config


def _resolve_masking_mode() -> str:
    if _TF_VERSION < version.parse("4.50.0"):
        raise ImportError(
            "IndicBERT-v3 requires transformers>=4.50.0 (Gemma3 support). "
            f"Found {transformers.__version__}."
        )
    if _TF_VERSION >= version.parse("5.0.0") and create_bidirectional_mask is not None:
        return "forward_v5"
    if _TF_VERSION >= version.parse("4.57.0"):
        return "parent_bidirectional"
    if _TF_VERSION >= version.parse("4.53.0") and create_causal_mask is not None:
        return "forward_v4_or_mask"
    return "update_causal_mask"


_MASKING_MODE = _resolve_masking_mode()


def _legacy_bidirectional_mask(
    attention_mask: torch.Tensor | None,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    config,
    past_key_values=None,
    output_attentions: bool = False,
):
    """Bidirectional mask with key-padding (transformers 4.50-4.52 path)."""
    if config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    using_static_cache = past_key_values is not None and hasattr(
        past_key_values, "get_max_length"
    )

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]

    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    if attention_mask is not None and attention_mask.dim() == 4:
        if attention_mask.max() != 0:
            raise ValueError(
                "Custom 4D attention mask should be passed in inverted form with max==0"
            )
        return attention_mask

    causal_mask = torch.zeros(
        (sequence_length, target_length), dtype=dtype, device=device
    )
    causal_mask = causal_mask[None, None, :, :].expand(
        input_tensor.shape[0], 1, -1, -1
    )

    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    if (
        config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        try:
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )
        except ImportError:
            pass

    return causal_mask


class NonCausalGemma3Attention(Gemma3Attention):
    """Gemma3Attention forced to non-causal mode."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.sliding_window = None


class NonCausalGemma3DecoderLayer(Gemma3DecoderLayer):
    """Decoder layer using non-causal attention."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = NonCausalGemma3Attention(config, layer_idx)


class BidirectionalGemma3Model(Gemma3TextModel):
    """Base Gemma3 Model with bidirectional attention masking."""

    _no_split_modules = ["NonCausalGemma3DecoderLayer"]
    _can_record_outputs = {
        "hidden_states": Gemma3DecoderLayer,
        "attentions": Gemma3Attention,
    }

    def __init__(self, config):
        _configure_bidirectional_config(config)
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                NonCausalGemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()

    def _resolve_forward_outputs(self, kwargs: dict) -> tuple[bool, bool]:
        output_attentions = kwargs.pop("output_attentions", None)
        output_hidden_states = kwargs.pop("output_hidden_states", None)
        if output_attentions is None:
            output_attentions = getattr(self.config, "output_attentions", False)
        if output_hidden_states is None:
            output_hidden_states = getattr(self.config, "output_hidden_states", False)
        return output_attentions, output_hidden_states

    def _build_v4_mask_mapping(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values,
        position_ids: torch.Tensor,
    ) -> dict:
        """Mask dict for transformers 4.53-4.56 (input_embeds kwarg, or-mask forced)."""
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        or_mask_function = lambda *args: torch.tensor(True, dtype=torch.bool)
        mask_kwargs["or_mask_function"] = or_mask_function
        sliding_mask_kwargs = mask_kwargs.copy()
        if self.config.sliding_window is not None and _bidirectional_window_overlay is not None:
            sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                self.config.sliding_window
            )
        else:
            sliding_mask_kwargs["or_mask_function"] = or_mask_function

        return {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
        }

    def _mask_fn_params(self, fn) -> set:
        import inspect

        return set(inspect.signature(fn).parameters)

    def _filter_mask_kwargs(self, fn, kwargs: dict) -> dict:
        if fn is None:
            return kwargs
        allowed = self._mask_fn_params(fn)
        return {k: v for k, v in kwargs.items() if k in allowed}

    def _build_v5_mask_mapping(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values,
        position_ids: torch.Tensor | None,
        cache_position: torch.Tensor | None = None,
    ) -> dict:
        """Mask dict for transformers 5.0+."""
        embeds_key = (
            "inputs_embeds"
            if "inputs_embeds" in self._mask_fn_params(create_bidirectional_mask)
            else "input_embeds"
        )
        mask_kwargs = {
            "config": self.config,
            embeds_key: inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
        sliding_mask_kwargs = mask_kwargs.copy()
        if self.config.sliding_window is not None and _bidirectional_window_overlay is not None:
            sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                self.config.sliding_window
            )
        return {
            "full_attention": create_bidirectional_mask(
                **self._filter_mask_kwargs(create_bidirectional_mask, mask_kwargs)
            ),
            "sliding_attention": create_sliding_window_causal_mask(
                **self._filter_mask_kwargs(
                    create_sliding_window_causal_mask, sliding_mask_kwargs
                )
            ),
        }

    def _forward_v4_or_mask(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Forward compatible with Gemma3 4.53-4.56 APIs."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None and DynamicCache is not None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = self._build_v4_mask_mapping(
                inputs_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
            )

        output_attentions, output_hidden_states = self._resolve_forward_outputs(kwargs)

        hidden_states = inputs_embeds
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _forward_v5(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Forward compatible with Gemma3 5.0+ APIs."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and DynamicCache is not None:
            past_key_values = DynamicCache(config=self.config)

        cache_position = kwargs.pop("cache_position", None)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = self._build_v5_mask_mapping(
                inputs_embeds,
                attention_mask,
                past_key_values,
                position_ids,
                cache_position,
            )

        output_attentions, output_hidden_states = self._resolve_forward_outputs(kwargs)

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        all_hidden_states = () if output_hidden_states else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_output = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None if not output_attentions else (),
        )


if _MASKING_MODE == "update_causal_mask":

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values=None,
        output_attentions: bool = False,
    ):
        return _legacy_bidirectional_mask(
            attention_mask,
            input_tensor,
            cache_position,
            self.config,
            past_key_values,
            output_attentions,
        )

    BidirectionalGemma3Model._update_causal_mask = _update_causal_mask

elif _MASKING_MODE == "forward_v4_or_mask":
    BidirectionalGemma3Model.forward = BidirectionalGemma3Model._forward_v4_or_mask

elif _MASKING_MODE == "forward_v5":
    BidirectionalGemma3Model.forward = BidirectionalGemma3Model._forward_v5

# parent_bidirectional: inherit Gemma3TextModel.forward unchanged


class BidirectionalGemma3ForCausalLM(Gemma3ForCausalLM):
    """Gemma3 Causal LM wrapper that uses the Bidirectional backbone."""

    def __init__(self, config):
        _configure_bidirectional_config(config)
        model_config = _get_text_config(config)
        if model_config is not config:
            _configure_bidirectional_config(model_config)
        super().__init__(model_config)
        self.model = BidirectionalGemma3Model(model_config)
