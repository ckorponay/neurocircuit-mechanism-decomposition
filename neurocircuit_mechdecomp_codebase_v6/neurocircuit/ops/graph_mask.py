import torch

def apply_graph_mask(attn_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply hard anatomical mask to attention logits.

    Parameters
    ----------
    attn_logits : tensor with last dims matching (R_tgt, R_src) or (R_src, R_tgt)
    mask : (R_src, R_tgt) boolean / {0,1}

    Returns
    -------
    masked_logits : tensor with -inf where mask==0
    """
    m = mask.to(dtype=torch.bool)
    if attn_logits.shape[-2:] == m.shape:
        return attn_logits.masked_fill(~m, float("-inf"))
    if attn_logits.shape[-2:] == m.T.shape:
        return attn_logits.masked_fill(~m.T, float("-inf"))
    raise ValueError(f"Mask shape {m.shape} incompatible with logits {attn_logits.shape}")
