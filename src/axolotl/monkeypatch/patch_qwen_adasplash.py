# patch_qwen_sparse_flashattention.py
from axolotl.monkeypatch.adasplash import adasplash_attention_forward

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
)

_original_forward = Qwen2Attention.forward

def patched_qwen_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_value,
    cache_position,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    # check if current layer is in the list of layers to apply adasplash
    layers_to_transform = list(range(self.config.num_hidden_layers))
    if hasattr(self.config, "alpha_scheduler_layers") and self.config.alpha_scheduler_layers is not None:
        layers_to_transform = self.config.alpha_scheduler_layers

    if sliding_window is None and self.layer_idx in layers_to_transform:
        attention_interface = adasplash_attention_forward
        kwargs['alpha'] = getattr(self.config, 'adasplash_alpha', 1.5)
    else:
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                print(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.layer_idx == 0:
        # compute the effective sequence length for the first layer
        eff_len = hidden_states.shape[1]
        if attention_mask is not None:
            mask = attention_mask[..., :eff_len]
            valid_mask = mask[:, 0, 0] == 0 if mask.dim() > 2 else mask == 1
            eff_len = valid_mask.sum(-1).long().mean().item()
        setattr(self.config, "adasplash_effective_sequence_len", eff_len)

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def apply_qwen_adasplash_patch():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Applying Qwen/Adasplash patch...")
    Qwen2Attention.forward = patched_qwen_attention_forward
