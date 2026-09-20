"""Zero-shot continuation log-prob scorer with optional KV reuse.

Reductions: sum, mean, PMI. Cached scores must match naïve re-encoding
of prefix+continuation on the same model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from jev.render import continuations, option_keys, render_prefix
from jev.schema import SystemOneRequest, Usage
from jev.scoring.protocol import ScoredQuestion
from jev.scoring.tiny_lm import ByteTokenizer, TinyCausalLM

Reduction = Literal["sum", "mean", "pmi"]


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _gather_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """logits [T, V], token_ids [T] -> log p(token_t | prefix) for each t."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def score_ids_naive(
    forward: Callable[[torch.Tensor], torch.Tensor],
    prefix_ids: Sequence[int],
    cont_ids: Sequence[int],
    reduction: Reduction,
    device: torch.device | None = None,
) -> float:
    if not cont_ids:
        return 0.0
    device = device or torch.device("cpu")
    full = torch.tensor([list(prefix_ids) + list(cont_ids)], dtype=torch.long, device=device)
    logits = forward(full)
    # token t is predicted from logits at position t-1
    start = len(prefix_ids) - 1
    if start < 0:
        raise ValueError("prefix must contain at least one token")
    slice_logits = logits[0, start : start + len(cont_ids)]
    targets = torch.tensor(list(cont_ids), dtype=torch.long, device=logits.device)
    tok_lp = _gather_token_logprobs(slice_logits, targets)
    return float(_reduce(tok_lp, reduction))


def score_ids_cached(
    forward_cache: Callable,
    prefix_ids: Sequence[int],
    cont_ids: Sequence[int],
    reduction: Reduction,
    unconditional_lp: float | None = None,
    device: torch.device | None = None,
) -> float:
    del unconditional_lp
    if not cont_ids:
        return 0.0
    device = device or torch.device("cpu")
    prefix = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
    prefix_logits, past = forward_cache(prefix, None)
    first_target = torch.tensor([cont_ids[0]], dtype=torch.long, device=prefix_logits.device)
    first_lp = _gather_token_logprobs(prefix_logits[0, -1:, :], first_target.unsqueeze(0))[0]
    lps = [first_lp]
    if len(cont_ids) > 1:
        rest = torch.tensor([list(cont_ids[:-1])], dtype=torch.long, device=prefix_logits.device)
        rest_logits, _ = forward_cache(rest, past)
        rest_targets = torch.tensor(list(cont_ids[1:]), dtype=torch.long, device=prefix_logits.device)
        rest_lp = _gather_token_logprobs(rest_logits[0], rest_targets)
        lps.append(rest_lp)
    tok_lp = torch.cat([p.reshape(-1) for p in lps], dim=0)
    return float(_reduce(tok_lp, reduction))


def _reduce(tok_lp: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "sum":
        return tok_lp.sum()
    if reduction == "mean":
        return tok_lp.mean()
    if reduction == "pmi":
        # PMI vs length-normalized uniform token prior, applied by caller for real PMI.
        return tok_lp.sum()
    raise ValueError(f"unknown reduction {reduction}")


def batched_continuation_logprobs(
    forward_cache: Callable,
    prefix_ids: Sequence[int],
    conts: Sequence[Sequence[int]],
    reduction: Reduction,
    device: torch.device | None = None,
) -> list[float]:
    """Score many continuations after one prefix prefill. Pads the batch."""
    if not conts:
        return []
    device = device or torch.device("cpu")
    prefix = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
    prefix_logits, past = forward_cache(prefix, None)
    # Expand past along batch
    max_len = max(len(c) for c in conts)
    # First-token scores from prefix logits (shared).
    first_ids = torch.tensor([c[0] if c else 0 for c in conts], dtype=torch.long, device=device)
    first_lp = _gather_token_logprobs(
        prefix_logits[0, -1, :].unsqueeze(0).expand(len(conts), -1),
        first_ids,
    )
    # Remaining tokens batched.
    rest_lens = [max(0, len(c) - 1) for c in conts]
    if max_len <= 1:
        tok_lps = first_lp.unsqueeze(-1)
        lengths = torch.tensor([max(1, len(c)) for c in conts], device=device)
        return [
            _reduce_row(tok_lps[i, : lengths[i]], reduction, int(lengths[i].item()))
            for i in range(len(conts))
        ]

    b = len(conts)
    rest = torch.zeros(b, max_len - 1, dtype=torch.long, device=device)
    rest_targets = torch.zeros(b, max_len - 1, dtype=torch.long, device=device)
    rest_mask = torch.zeros(b, max_len - 1, dtype=torch.bool, device=device)
    for i, c in enumerate(conts):
        if len(c) > 1:
            rest[i, : len(c) - 1] = torch.tensor(list(c[:-1]), dtype=torch.long, device=device)
            rest_targets[i, : len(c) - 1] = torch.tensor(list(c[1:]), dtype=torch.long, device=device)
            rest_mask[i, : len(c) - 1] = True
    past_b = _expand_past(past, b)
    rest_logits, _ = forward_cache(rest, past_b)
    rest_lp = _gather_token_logprobs_2d(rest_logits, rest_targets)
    rest_lp = rest_lp.masked_fill(~rest_mask, 0.0)
    rows = []
    for i, c in enumerate(conts):
        parts = [first_lp[i].reshape(-1)]
        if rest_lens[i]:
            parts.append(rest_lp[i, : rest_lens[i]])
        rows.append(torch.cat(parts, dim=0))
    return [_reduce_row(row, reduction, max(1, len(c))) for row, c in zip(rows, conts, strict=True)]


def _gather_token_logprobs_2d(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def _expand_past(past: list[tuple[torch.Tensor, torch.Tensor]], batch: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in past:
        out.append((k.expand(batch, -1, -1, -1).contiguous(), v.expand(batch, -1, -1, -1).contiguous()))
    return out


def _reduce_row(tok_lp: torch.Tensor, reduction: Reduction, length: int) -> float:
    if reduction == "mean":
        return float(tok_lp.sum() / max(1, length))
    return float(tok_lp.sum())


def pmi_adjust(cond_sum: float, uncond_sum: float) -> float:
    return cond_sum - uncond_sum


@dataclass
class LogprobScorer:
    model: TinyCausalLM | torch.nn.Module
    tokenizer: ByteTokenizer | object
    reduction: Reduction = "sum"
    model_id: str = "tiny-causal-logprob"
    use_cache: bool = True
    uncond_prefix: str = "Answer:"

    def _device(self) -> torch.device:
        if isinstance(self.model, torch.nn.Module):
            return _module_device(self.model)
        return torch.device("cpu")

    def _encode(self, text: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode")
        ids = encode(text, add_special_tokens=False)
        if hasattr(ids, "ids"):
            ids = ids.ids
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return list(ids)

    def _forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids, past_key_values=None, use_cache=False)
        if isinstance(out, tuple):
            return out[0]
        return out.logits if hasattr(out, "logits") else out

    def _forward_cache(self, input_ids: torch.Tensor, past: object) -> tuple[torch.Tensor, object]:
        out = self.model(input_ids, past_key_values=past, use_cache=True)
        if isinstance(out, tuple):
            return out[0], out[1]
        return out.logits, getattr(out, "past_key_values", None)

    @torch.no_grad()
    def score_strings(self, prefix: str, option_conts: Sequence[str]) -> list[float]:
        prefix_ids = self._encode(prefix)
        if not prefix_ids:
            prefix_ids = [0]
        cont_ids = [self._encode(c) for c in option_conts]
        red: Reduction = "mean" if self.reduction == "mean" else "sum"
        device = self._device()
        if self.use_cache:
            if isinstance(self.model, TinyCausalLM):
                scores = batched_continuation_logprobs(
                    self._forward_cache, prefix_ids, cont_ids, red, device=device
                )
            else:
                scores = [
                    score_ids_cached(self._forward_cache, prefix_ids, c, red, device=device)
                    for c in cont_ids
                ]
        else:
            scores = [
                score_ids_naive(self._forward_full, prefix_ids, c, red, device=device)
                for c in cont_ids
            ]
        if self.reduction == "pmi":
            uncond_prefix_ids = self._encode(self.uncond_prefix) or [0]
            if isinstance(self.model, TinyCausalLM):
                uncond = batched_continuation_logprobs(
                    self._forward_cache, uncond_prefix_ids, cont_ids, "sum", device=device
                )
            else:
                uncond = [
                    score_ids_cached(self._forward_cache, uncond_prefix_ids, c, "sum", device=device)
                    for c in cont_ids
                ]
            scores = [pmi_adjust(c, u) for c, u in zip(scores, uncond, strict=True)]
        return scores

    def score_request(self, request: SystemOneRequest) -> list[ScoredQuestion]:
        out: list[ScoredQuestion] = []
        for qid, question in request.questions.items():
            prefix = render_prefix(request.state, question)
            conts = continuations(question)
            logits = self.score_strings(prefix, conts)
            n_in = len(self._encode(prefix))
            n_out = sum(len(self._encode(c)) for c in conts)
            out.append(
                ScoredQuestion(
                    question_id=qid,
                    type=question.type,
                    keys=option_keys(question),
                    logits=logits,
                    usage=Usage(input_tokens=n_in, output_tokens=n_out),
                )
            )
        return out


def naive_vs_cached_scores(
    model: TinyCausalLM,
    tokenizer: ByteTokenizer,
    prefix: str,
    conts: Sequence[str],
) -> tuple[list[float], list[float]]:
    naive = LogprobScorer(model, tokenizer, use_cache=False, reduction="sum")
    cached = LogprobScorer(model, tokenizer, use_cache=True, reduction="sum")
    return naive.score_strings(prefix, conts), cached.score_strings(prefix, conts)
