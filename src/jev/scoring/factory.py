"""Construct named scorers for CLI and serving."""

from __future__ import annotations

from pathlib import Path

from jev.scoring.fake import FakeScorer
from jev.scoring.logprob import LogprobScorer
from jev.scoring.protocol import Scorer
from jev.scoring.tiny_lm import ByteTokenizer, seeded_tiny_lm


def build_scorer(
    backend: str,
    *,
    checkpoint: Path | None = None,
    hf_model_id: str = "Qwen/Qwen2.5-0.5B",
    reduction: str = "sum",
    device: str | None = None,
) -> Scorer:
    name = backend.strip().lower()
    if name in {"fake", "fake-overlap"}:
        return FakeScorer()
    if name in {"tiny-logprob", "tiny"}:
        return LogprobScorer(seeded_tiny_lm(0), ByteTokenizer(), model_id="tiny-causal-logprob")
    if name in {"hf-logprob", "hf", "zero-shot"}:
        from jev.scoring.hf_lm import try_load_hf_logprob_scorer

        loaded = try_load_hf_logprob_scorer(
            hf_model_id, device=device, reduction=reduction  # type: ignore[arg-type]
        )
        if loaded is None:
            from jev.scoring.hf_lm import LAST_ERROR

            raise RuntimeError(f"failed to load HF logprob scorer {hf_model_id}: {LAST_ERROR}")
        scorer, _meta = loaded
        return scorer
    if name in {"option-head", "head", "hashing-head"}:
        if checkpoint is None:
            raise ValueError("option-head backend requires --checkpoint")
        from jev.scoring.option_head import OptionHeadScorer, load_checkpoint

        encoder, head, meta = load_checkpoint(checkpoint)
        model_id = str(meta.get("model_id") or "option-attention")
        return OptionHeadScorer(encoder, head, model_id=model_id)
    raise ValueError(
        f"unknown backend {backend!r}; expected fake|tiny-logprob|hf-logprob|option-head"
    )
