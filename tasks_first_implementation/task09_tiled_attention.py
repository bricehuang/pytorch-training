"""Task 9: exact tiled attention forward pass with online softmax."""

from __future__ import annotations

import torch

def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_block_size: int,
    key_block_size: int,
    causal: bool = False,
) -> torch.Tensor:
    """Compute exact attention without a full ``[Tq, Tk]`` score matrix.

    Args:
        q: Queries with shape ``[B, H, Tq, D]``.
        k: Keys with shape ``[B, H, Tk, D]``.
        v: Values with shape ``[B, H, Tk, Dv]``.
        query_block_size: Number of query rows in a tile.
        key_block_size: Number of key rows in a tile.
        causal: Apply the usual lower-triangular mask. Public causal tests use
            ``Tq == Tk``.

    Returns:
        Output with shape ``[B, H, Tq, Dv]``.

    Constraints:
        Python loops over tiles are allowed. Do not materialize a tensor whose
        query and key dimensions are simultaneously the full ``Tq`` and ``Tk``.
        Maintain a running maximum, denominator, and weighted-value accumulator
        for each query row.
    """
    B,H,Tq,D = q.shape
    _,_,_,Dv = v.shape
    # num_q_blocks = q.shape[-2] // query_block_size
    # num_kv_blocks = k.shape[-2] // key_block_size
    q_blocks = q.split(split_size=query_block_size, dim=-2)
    k_blocks = k.split(split_size=key_block_size, dim=-2)
    v_blocks = v.split(split_size=key_block_size, dim=-2)
    out = torch.empty((B,H,Tq,Dv), dtype=v.dtype, device=v.device)
    for q_idx, q_block in enumerate(q_blocks):
        # q_block: [B, H, tq, D]
        start = q_idx * query_block_size
        end = start + query_block_size
        # compute out[:, :, start:end, :] [B, H, tq, Dv]
        # with online softmax

        # max_{i=1}^k a_i
        running_max = torch.empty_like(
            out[:, :, start:end, :1]
        ) # [B, H, tq, 1]
        running_max.fill_(float("-inf")) 

        # sum_{i=1}^k e^{a_i - running max(a)}}
        running_den = torch.zeros_like(running_max) # [B, H, tq, 1]

        # sum_{i=1}^k e^{a_i - running max(a)} v_i
        running_sum = torch.zeros_like(out[:, :, start:end, :]) # [B, H, tq, Dv]
        for kv_idx, (k_block, v_block) in enumerate(zip(k_blocks, v_blocks)):
            # k_block: [B, H, tk, D]
            # v_block: [B, H, tk, Dv]
            attn_block: torch.Tensor = q_block @ k_block.transpose(-1,-2) * (D ** -0.5) # [B, H, tq, tk]
            if causal:
                # q's start at index q_idx * query_block_size
                # k's start at index kv_idx * key_block_size
                offset = q_idx * query_block_size - kv_idx * key_block_size
                mask = torch.tril(
                    input=torch.ones(
                        size=attn_block.shape[-2:],
                        dtype=torch.bool,
                        device=attn_block.device
                    ),
                    diagonal=offset,
                ).unsqueeze(0).unsqueeze(0)
                attn_block.masked_fill_(~mask, float("-inf"))
            row_max = attn_block.amax(dim=-1, keepdim=True) # [B, H, tq, 1]
            if causal: 
                valid_rows = mask.any(dim=-1, keepdim=True)
                row_max = torch.where(
                    valid_rows,
                    row_max,
                    torch.zeros_like(row_max),
                )
            block_exp = torch.exp(attn_block - row_max) # [B, H, tq, tk]
            den_piece = block_exp.sum(dim=-1, keepdim=True) # [B, H, tq, 1]
            sum_piece = block_exp @ v_block

            # update online softmax state. all max, wts [B, H, tq, 1]
            old_max = running_max
            running_max = old_max.maximum(row_max)
            old_wts = torch.exp(old_max - running_max)
            new_wts = torch.exp(row_max - running_max)

            running_den = old_wts * running_den + new_wts * den_piece
            running_sum = old_wts * running_sum + new_wts * sum_piece
        out[:, :, start:end, :] = running_sum / running_den
    return out
