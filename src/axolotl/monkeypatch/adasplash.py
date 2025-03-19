import torch
from transformers.models.qwen2.modeling_qwen2 import repeat_kv

try:
    from adasplash import adasplash, adasplash_no_block_mask
except ImportError:
    raise ImportError("Please install adasplash first via `pip install adasplash`")


def adasplash_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    causal=True,
    alpha=0.5,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # compute varlen
    if attention_mask is not None:
        mask = attention_mask[..., : key_states.shape[-2]]
        if mask.dim() > 2:
            valid_mask = mask[:, 0, 0] == 0
        else:
            valid_mask = mask == 1
        varlen = valid_mask.sum(-1).long().contiguous()
    else:
        varlen = torch.tensor([key_states.shape[-2]] * key_states.shape[0], device=key_states.device)
        # varlen = None

    # ensure that the input is contiguous
    query = query.contiguous()
    key = key_states.contiguous()
    value = value_states.contiguous()

    # choose the adasplash function based on the alpha value
    adasplash_fn = adasplash if alpha >= 1.5 else adasplash_no_block_mask

    attn_output = adasplash_fn(
        query, key, value,
        alpha=alpha,
        is_causal=causal,
        varlen=varlen,
        niter=4,
    )
    return attn_output, None
