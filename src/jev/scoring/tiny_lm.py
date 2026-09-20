"""Tiny causal LM with KV cache — CPU reference for naïve vs cached scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    """True where attention is forbidden. Allows q to see all prefix keys."""
    q_pos = torch.arange(k_len - q_len, k_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    return k_pos > q_pos


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must divide n_head")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, d = x.shape
        h = self.n_head
        hd = self.head_dim
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, h, hd).transpose(1, 2)
        k = k.view(b, t, h, hd).transpose(1, 2)
        v = v.view(b, t, h, hd).transpose(1, 2)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        k_len = k.size(2)
        forbidden = _causal_mask(t, k_len, x.device)
        attn = (q @ k.transpose(-2, -1)) / (hd**0.5)
        attn = attn.masked_fill(forbidden, torch.finfo(attn.dtype).min)
        weights = torch.softmax(attn, dim=-1)
        y = (weights @ v).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 32,
        n_layer: int = 2,
        n_head: int = 4,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(TinyBlock(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        b, t = input_ids.shape
        past_len = 0 if not past_key_values else past_key_values[0][0].size(2)
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(positions)[None, :, :]
        new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, kv = block(x, past=past)
            new_past.append(kv)
        logits = self.head(self.ln_f(x))
        return logits, (new_past if use_cache else None)


@dataclass
class ByteTokenizer:
    """Deterministic byte-ish tokenizer for TinyCausalLM tests."""

    vocab_size: int = 128

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = [min(ord(ch), self.vocab_size - 1) for ch in text]
        return ids or [0]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def seeded_tiny_lm(seed: int = 0, **kwargs: int) -> TinyCausalLM:
    torch.manual_seed(seed)
    model = TinyCausalLM(**kwargs)
    model.eval()
    return model
