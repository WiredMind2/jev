"""Hugging Face causal LM wrapper for zero-shot scoring. Optional import."""

from __future__ import annotations

from typing import Any

import torch

from jev.scoring.logprob import LogprobScorer, Reduction

LAST_ERROR: str | None = None


def try_load_hf_logprob_scorer(
    model_id: str,
    *,
    dtype: str = "float16",
    device: str | None = None,
    reduction: Reduction = "sum",
    revision: str | None = None,
) -> tuple[LogprobScorer, dict[str, Any]] | None:
    """Return a logprob scorer or None if transformers/model are unavailable."""
    global LAST_ERROR
    LAST_ERROR = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    if device == "cpu":
        torch_dtype = torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch_dtype,
        )
        model.to(device)
        model.eval()
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    scorer = LogprobScorer(
        model=model,
        tokenizer=tokenizer,
        reduction=reduction,
        model_id=model_id,
        use_cache=True,
    )
    meta = {
        "model_id": model_id,
        "revision": getattr(model.config, "_name_or_path", model_id),
        "device": device,
        "dtype": str(torch_dtype),
    }
    return scorer, meta
