from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):

    def __init__(self, head_dim, context_length, theta=10000):
        super().__init__()
        if head_dim % 2 == 1:
            raise ValueError(f"head_dim={head_dim} must be even")
        half_head_dim = head_dim // 2
        exponents = -torch.arange(half_head_dim) / half_head_dim
        phases = theta ** exponents
        angles = torch.outer(
            torch.arange(context_length),
            phases,
        ) # [context_length, half_head_dim]
        cosines = torch.cos(angles)
        sines = torch.sin(angles)
        self.register_buffer("cosines", cosines)
        self.register_buffer("sines", sines)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., T, Dh]
        T, Dh = x.shape[-2:]
        cosines, sines = self.cosines, self.sines
        context_length, half_Dh = cosines.shape
        if Dh != 2 * half_Dh:
            raise ValueError(f"Dh={Dh} does not equal twice half_dh={half_Dh}")
        if T > context_length:
            raise ValueError(f"tokens={T} exceeds context length={context_length}")
        # reshape to [..., T, half_Dh]
        cosines = cosines[:T,:].reshape((1,) * (x.ndim-2) + (T, half_Dh))
        sines = sines[:T,:].reshape((1,) * (x.ndim-2) + (T, half_Dh))
        # [..., T, half_Dh] each
        x_even, x_odd = x.reshape(
            *x.shape[:-1], half_Dh, 2
        ).unbind(dim=-1)
        y_even = cosines * x_even - sines * x_odd
        y_odd = sines * x_even + cosines * x_odd
        return torch.stack(
            (y_even, y_odd),
            dim=-1,
        ).flatten(-2)


class AttentionBlock(nn.Module):

    def __init__(self, d_model, num_heads, context_length, gqa_factor=1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"num_heads={num_heads} must divide d_model={d_model}")
        if num_heads % gqa_factor != 0:
            raise ValueError(f"gqa_factor={gqa_factor} must divide num_heads={num_heads}")
        self.D = d_model
        self.H = num_heads
        self.gqa_factor = gqa_factor
        self.kvH = num_heads // gqa_factor
        self.Dh = d_model // num_heads
        self.rope = RoPE(self.Dh, context_length)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * self.kvH * self.Dh, bias=False)
    
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, kvH, Dh, gqa_factor = self.H, self.kvH, self.Dh, self.gqa_factor
        q = self.q(x) # [B, T, D]
        kv = self.kv(x) #[B, T, 2 * kvH * Dh]
        k, v = kv.chunk(2, dim=-1) #[B, T, kvH * Dh] each

        q = q.reshape(B, T, H, Dh).transpose(-2, -3) # [B, H, T, Dh]
        k = k.reshape(B, T, kvH, Dh).transpose(-2, -3) # [B, kvH, T, Dh]
        v = v.reshape(B, T, kvH, Dh).transpose(-2, -3) # [B, kvH, T, Dh]

        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(gqa_factor, dim=1)
        v = v.repeat_interleave(gqa_factor, dim=1)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(-2,-3).reshape(B, T, D)
        return self.out(attn_out)


class MLPBlock(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.lin = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.out = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.lin(x)) * self.gate(x))

class TransformerBlock(nn.Module):

    def __init__(self, d_model, d_ff, num_heads, context_length, gqa_factor=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.attn_block = AttentionBlock(d_model, num_heads, context_length, gqa_factor=gqa_factor)
        self.ln2 = nn.LayerNorm(d_model, bias=False)
        self.mlp_block = MLPBlock(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_block(self.ln1(x))
        return x + self.mlp_block(self.ln2(x))

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, context_length, gqa_factor=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, context_length, gqa_factor=gqa_factor)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model, bias=False)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T] token ids
        h = self.embed(x) # [B, T, d_model]
        for block in self.blocks:
            h = block(h)
        return self.out(self.ln(h))
