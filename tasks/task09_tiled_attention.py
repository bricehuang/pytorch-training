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
    B, H, Tq, D = q.shape
    _, _, _, Dv = v.shape
    q_blocks = q.split(query_block_size, dim=-2)
    k_blocks = k.split(key_block_size, dim=-2)
    v_blocks = v.split(key_block_size, dim=-2)
    out = q.new_empty((B, H, Tq, Dv))
    for q_idx, q_block in enumerate(q_blocks):
        start = q_idx * query_block_size
        end = start + q_block.shape[-2]
        # compute out[:, :, start:end, :] # [B, H, tq, Dv]
        out_slice = out[:, :, start:end, :]
        running_max = torch.full_like(
            input=out_slice[:,:,:,:1],
            fill_value=float("-inf")
        ) # [B, H, tq, 1]
        running_den = torch.zeros_like(out_slice[:,:,:,:1]) # [B, H, tq, 1]
        running_num = torch.zeros_like(out_slice) # [B, H, tq, Dv]
        for kv_idx, (k_block, v_block) in enumerate(zip(k_blocks, v_blocks)):
            # q_block [B, H, tq, D]
            # k_block [B, H, tk, D]
            # v_block [B, H, tk, Dv]
            attn_block: torch.Tensor = q_block @ k_block.transpose(-1,-2) * (D ** -0.5) # [B, H, tq, tk]
            if causal:
                offset = q_idx * query_block_size - kv_idx * key_block_size
                mask = torch.tril(
                    input=torch.ones(
                        size=attn_block.shape[-2:],
                        dtype=torch.bool,
                        device=attn_block.device,
                    ),
                    diagonal=offset,
                ).unsqueeze(0).unsqueeze(0) # [1, 1, tq, tk]
                attn_block = attn_block.masked_fill(~mask, float("-inf"))
            row_max = attn_block.amax(dim=-1, keepdim=True) # [B, H, tq, 1]
            if causal:
                # make row_max safe
                valid_rows = mask.any(dim=-1, keepdim=True) # [1, 1, tq, 1]
                safe_row_max = row_max.masked_fill_(~valid_rows, 0.0)
            else:
                safe_row_max = row_max
            block_exp = torch.exp(attn_block - safe_row_max) # [B, H, tq, tk]
            den_piece = block_exp.sum(dim=-1, keepdim=True) # [B, H, tq, 1]
            num_piece = block_exp @ v_block # [B, H, tq, Dv]

            # update softmax state
            old_max = running_max
            running_max = old_max.maximum(row_max)
            old_wts = torch.exp(old_max - running_max)
            new_wts = torch.exp(row_max - running_max)
            running_den = old_wts * running_den + new_wts * den_piece
            running_num = old_wts * running_num + new_wts * num_piece
        out[:, :, start:end, :] = running_num / running_den
    return out
