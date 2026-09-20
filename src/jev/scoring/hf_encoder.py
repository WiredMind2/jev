"""Frozen Hugging Face encoder for the option-attention head.

Uses last hidden states of a causal LM (default Qwen2.5-0.5B) with all
encoder parameters frozen. The LM head is not loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class FrozenHFEncoder(nn.Module):
    def __init__(
        self,
        model_id: str,
        *,
        max_len: int = 256,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.torch_dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            raise ValueError(f"{model_id} has no hidden_size")
        self.d_model = int(hidden)

    def _forward_chunk(self, texts: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.backbone(**enc, use_cache=False)
            hidden = out.last_hidden_state.float()
        mask = enc["attention_mask"].bool()
        return hidden, mask

    def forward_texts(
        self,
        texts: Sequence[str],
        device: torch.device,
        *,
        chunk_size: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode texts. Wide menus are chunked so a 4 GiB card does not OOM."""
        if not texts:
            hidden = torch.zeros(0, 1, self.d_model, device=device)
            mask = torch.zeros(0, 1, dtype=torch.bool, device=device)
            return hidden, mask
        rows = list(texts)
        if len(rows) <= chunk_size:
            return self._forward_chunk(rows, device)
        parts_h: list[torch.Tensor] = []
        parts_m: list[torch.Tensor] = []
        for i in range(0, len(rows), chunk_size):
            h, m = self._forward_chunk(rows[i : i + chunk_size], device)
            parts_h.append(h)
            parts_m.append(m)
        max_t = max(h.size(1) for h in parts_h)
        dim = parts_h[0].size(-1)
        hidden = parts_h[0].new_zeros(len(rows), max_t, dim)
        mask = parts_m[0].new_zeros(len(rows), max_t, dtype=torch.bool)
        offset = 0
        for h, m in zip(parts_h, parts_m, strict=True):
            n, tlen = h.size(0), h.size(1)
            hidden[offset : offset + n, :tlen] = h
            mask[offset : offset + n, :tlen] = m
            offset += n
        return hidden, mask


LAST_ERROR: str | None = None


def try_load_frozen_encoder(
    model_id: str,
    **kwargs: Any,
) -> tuple[FrozenHFEncoder, dict[str, Any]] | None:
    global LAST_ERROR
    LAST_ERROR = None
    try:
        enc = FrozenHFEncoder(model_id, **kwargs)
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    meta = {
        "model_id": model_id,
        "device": str(enc.device),
        "dtype": str(enc.torch_dtype),
        "d_model": enc.d_model,
        "max_len": enc.max_len,
    }
    return enc, meta
