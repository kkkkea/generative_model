from __future__ import annotations

from typing import Optional

import torch


def causal_mask(
    seq_len: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Create a causal attention mask for autoregressive self-attention.

    Args:
        seq_len: Sequence length.
        device: Device for the returned mask.
        dtype: Output dtype. `torch.bool` returns an allow-mask compatible with
            `torch.nn.functional.scaled_dot_product_attention`.

    Returns:
        A `[seq_len, seq_len]` lower-triangular mask.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()

    if dtype == torch.bool:
        return mask

    return mask.to(dtype=dtype)
