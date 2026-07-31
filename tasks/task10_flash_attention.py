"""Task 10: FlashAttention-style tiled forward and manually derived backward."""

from __future__ import annotations

import torch


class FlashAttentionFunction(torch.autograd.Function):
    """Custom exact attention operation that avoids saving an O(T^2) matrix."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_block_size: int,
        key_block_size: int,
        causal: bool,
    ) -> torch.Tensor:
        """Run tiled online-softmax attention and save compact backward state."""
        B, H, Tq, D = q.shape
        _, _, _, Dv = v.shape
        q_blocks = q.split(query_block_size, dim=-2)
        k_blocks = k.split(key_block_size, dim=-2)
        v_blocks = v.split(key_block_size, dim=-2)
        out = q.new_empty((B, H, Tq, Dv))
        logsumexps = q.new_empty((B, H, Tq, 1))
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
                attn_block = q_block @ k_block.transpose(-1,-2) * (D ** -0.5) # [B, H, tq, tk]
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
                    safe_row_max = row_max.masked_fill(~valid_rows, 0.0)
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
            logsumexps[:, :, start:end, :] = running_max + running_den.log()
        ctx.save_for_backward(q, k, v, logsumexps, out)
        ctx.causal = causal
        ctx.query_block_size = query_block_size
        ctx.key_block_size = key_block_size
        return out

    @staticmethod
    def prob_block(
        q_block: torch.Tensor, # [B, H, tq, D]
        k_block: torch.Tensor, # [B, H, tk, D]
        logsumexp_block: torch.Tensor, # [B, H, tq, 1]
        causal: bool,
        offset: int,
    ) -> torch.Tensor:
        D = q_block.shape[-1]
        attn_block = q_block @ k_block.transpose(-1,-2) * (D**-0.5) # [B, H, tq, tk]
        if causal:
            mask = torch.tril(
                torch.ones(
                    size=attn_block.shape[-2:],
                    dtype=torch.bool,
                    device=attn_block.device,
                ),
                diagonal=offset,
            )
            attn_block = attn_block.masked_fill(~mask, float("-inf"))
        prob_block = torch.exp(attn_block - logsumexp_block)
        return prob_block

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        """Recompute probability tiles and return ``dq``, ``dk``, and ``dv``."""
        q, k, v, logsumexps, out = ctx.saved_tensors
        attn_scale = q.shape[-1] ** -0.5
        # q: [B, H, Tq, D]
        # k: [B, H, Tk, D]
        # v: [B, H, Tk, Dv]
        # logsumexps: [B, H, Tq, 1]
        # out: [B, H, Tq, Dv]
        query_block_size, key_block_size, causal = ctx.query_block_size, ctx.key_block_size, ctx.causal
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        q_blocks = q.split(query_block_size, dim=-2)
        out_blocks = out.split(query_block_size, dim=-2)
        grad_output_blocks = grad_output.split(query_block_size, -2)
        logsumexp_blocks = logsumexps.split(query_block_size, dim=-2)
        k_blocks = k.split(key_block_size, dim=-2)
        v_blocks = v.split(key_block_size, dim=-2)
        for q_idx, (q_block, out_block, grad_output_block, logsumexp_block) in enumerate(
            zip(
                q_blocks,
                out_blocks,
                grad_output_blocks,
                logsumexp_blocks,
            )
        ):
            q_start = q_idx * query_block_size
            q_end = q_start + q_block.shape[-2]
            # q_block: [B, H, tq, D]
            # out_block: [B, H, tq, Dv]
            # grad_output_block: [B, H, tq, Dv]
            # logsumexp_block: [B, H, tq, 1]
            avg_grad_prob = (out_block * grad_output_block).sum(dim=-1, keepdim=True)
            for kv_idx, (k_block, v_block) in enumerate(zip(k_blocks, v_blocks)):
                kv_start = kv_idx * key_block_size
                kv_end = kv_start + k_block.shape[-2]
                # k_block: [B, H, tk, D]
                # v_block: [B, H, tk, Dv]
                prob_block = FlashAttentionFunction.prob_block(
                    q_block=q_block,
                    k_block=k_block,
                    logsumexp_block=logsumexp_block,
                    causal=causal,
                    offset=q_start - kv_start
                ) # [B, H, tq, tk]
                # compute grad_prob = dL/d(prob_block)
                # compute grad_attn = dL/d(attn_block)
                grad_prob = grad_output_block @ v_block.transpose(-1, -2) # [B, H, tq, tk]
                grad_attn = prob_block * (grad_prob - avg_grad_prob) # [B, H, tq, tk]
                grad_q[:, :, q_start:q_end, :] += attn_scale * grad_attn @ k_block
                grad_k[:, :, kv_start:kv_end, :] += attn_scale * grad_attn.transpose(-1,-2) @ q_block
                grad_v[:, :, kv_start:kv_end, :] += prob_block.transpose(-1,-2) @ grad_output_block

        return (grad_q, grad_k, grad_v, None, None, None)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_block_size: int = 64,
    key_block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """User-facing wrapper for :class:`FlashAttentionFunction`.

    Inputs have shapes ``q=[B,H,Tq,D]``, ``k=[B,H,Tk,D]``, and
    ``v=[B,H,Tk,Dv]``. Public causal tests use ``Tq == Tk``.
    """
    return FlashAttentionFunction.apply(q, k, v, query_block_size, key_block_size, causal)
