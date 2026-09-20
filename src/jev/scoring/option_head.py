"""Frozen-encoder option-query cross-attention head.

State tokens H_x are encoded once. Each option is a query over those
tokens. The encoder (hashing toy or HF causal LM as encoder) stays frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from jev.render import option_keys, option_texts, render_state
from jev.schema import (
    ChoiceTrainingExample,
    NoulTrainingExample,
    ScoreTrainingExample,
    SystemOneRequest,
    Usage,
    parse_training_example,
)
from jev.scoring.protocol import ScoredQuestion

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
CHECKPOINT_VERSION = 1


def simple_tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return toks or ["empty"]


class HashingEncoder(nn.Module):
    """Frozen deterministic encoder for CPU tests. Not a language model."""

    def __init__(self, d_model: int = 64, vocab_size: int = 4096, max_len: int = 128, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        emb = torch.randn(vocab_size, d_model, generator=g)
        pos = torch.randn(max_len, d_model, generator=g)
        self.register_buffer("emb", emb, persistent=True)
        self.register_buffer("pos", pos, persistent=True)
        for p in self.parameters():
            p.requires_grad_(False)

    def token_ids(self, text: str) -> torch.Tensor:
        ids = []
        for tok in simple_tokenize(text)[: self.max_len]:
            digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
            h = int(digest, 16) % self.vocab_size
            ids.append(h)
        return torch.tensor(ids, dtype=torch.long)

    def forward_texts(self, texts: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.token_ids(t).to(device) for t in texts]
        padded = pad_sequence(seqs, batch_first=True, padding_value=0)
        if padded.size(1) > self.max_len:
            padded = padded[:, : self.max_len]
        b, t = padded.shape
        mask = torch.zeros(b, t, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            n = min(len(s), t)
            mask[i, :n] = True
        pos = self.pos[:t].unsqueeze(0)
        hidden = self.emb[padded.clamp(0, self.vocab_size - 1)] + pos
        hidden = hidden * mask.unsqueeze(-1)
        return hidden, mask


class OptionAttentionHead(nn.Module):
    def __init__(self, d_model: int, rank: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.query = nn.Linear(d_model, rank, bias=False)
        self.key = nn.Linear(d_model, rank, bias=False)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        state_hidden: torch.Tensor,
        state_mask: torch.Tensor,
        option_hidden: torch.Tensor,
        option_mask: torch.Tensor,
        option_batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        state_hidden: [B, T, D]
        option_hidden: [N, S, D] ragged options belonging to batch items
        option_batch_index: [N] mapping option -> state row
        returns logits [N]
        """
        opt_pool = masked_mean(option_hidden, option_mask)
        q = self.query(opt_pool)
        gathered_state = state_hidden[option_batch_index]
        gathered_mask = state_mask[option_batch_index]
        k = self.key(gathered_state)
        scale = math.sqrt(self.rank)
        scores = torch.einsum("nd,ntd->nt", q, k) / scale
        scores = scores.masked_fill(~gathered_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        attended = torch.einsum("nt,ntd->nd", attn, gathered_state)
        logits = self.mlp(torch.cat([attended, opt_pool], dim=-1)).squeeze(-1)
        return logits


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


def example_state_text(example: Any) -> str:
    return render_state(example.state)


def example_option_texts(example: Any) -> list[str]:
    return option_texts(example.question)


def example_gold_index(example: Any) -> int:
    if isinstance(example, ChoiceTrainingExample):
        return list(example.question.criteria).index(example.gold)
    if isinstance(example, ScoreTrainingExample):
        return int(example.gold)
    if isinstance(example, NoulTrainingExample):
        return int(example.gold)
    raise TypeError("unknown example type")


class OptionHeadScorer:
    def __init__(
        self,
        encoder: nn.Module,
        head: OptionAttentionHead,
        device: torch.device | None = None,
        model_id: str = "option-attention-hashing",
    ) -> None:
        self.encoder = encoder
        self.head = head
        inferred = getattr(encoder, "device", None)
        self.device = device or inferred or torch.device("cpu")
        if hasattr(self.encoder, "to"):
            self.encoder.to(self.device)
        self.head.to(self.device)
        self.head.eval()
        self.encoder.eval()
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @torch.no_grad()
    def score_texts(self, state_text: str, option_text_list: Sequence[str]) -> list[float]:
        state_h, state_m = self.encoder.forward_texts([state_text], self.device)
        opt_h, opt_m = self.encoder.forward_texts(list(option_text_list), self.device)
        idx = torch.zeros(len(option_text_list), dtype=torch.long, device=self.device)
        logits = self.head(state_h.float(), state_m, opt_h.float(), opt_m, idx)
        return [float(x) for x in logits.detach().cpu().tolist()]

    def score_request(self, request: SystemOneRequest) -> list[ScoredQuestion]:
        state_text = render_state(request.state)
        out: list[ScoredQuestion] = []
        for qid, question in request.questions.items():
            texts = option_texts(question)
            logits = self.score_texts(state_text, texts)
            out.append(
                ScoredQuestion(
                    question_id=qid,
                    type=question.type,
                    keys=option_keys(question),
                    logits=logits,
                    usage=Usage(input_tokens=len(simple_tokenize(state_text)), output_tokens=len(texts)),
                )
            )
        return out


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 3e-3
    batch_size: int = 16
    rank: int = 32
    d_model: int = 64
    seed: int = 0
    max_steps: int | None = None
    encoder_kind: str = "hashing"
    hf_model_id: str = "Qwen/Qwen2.5-0.5B"
    patience: int = 5
    max_state_tokens: int = 256


def load_jsonl_examples(path: Path) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(parse_training_example(json.loads(line)))
    return rows


def _collate(encoder: nn.Module, batch: Sequence[Any], device: torch.device):
    state_texts = [example_state_text(ex) for ex in batch]
    option_texts_flat: list[str] = []
    option_index: list[int] = []
    gold: list[int] = []
    n_opts: list[int] = []
    for i, ex in enumerate(batch):
        opts = example_option_texts(ex)
        n_opts.append(len(opts))
        gold.append(example_gold_index(ex))
        for t in opts:
            option_texts_flat.append(t)
            option_index.append(i)
    state_h, state_m = encoder.forward_texts(state_texts, device)
    opt_h, opt_m = encoder.forward_texts(option_texts_flat, device)
    idx = torch.tensor(option_index, dtype=torch.long, device=device)
    gold_t = torch.tensor(gold, dtype=torch.long, device=device)
    return state_h.float(), state_m, opt_h.float(), opt_m, idx, gold_t, n_opts


def listwise_ce(logits: torch.Tensor, n_opts: Sequence[int], gold: torch.Tensor) -> torch.Tensor:
    loss = logits.new_zeros(())
    offset = 0
    for i, n in enumerate(n_opts):
        chunk = logits[offset : offset + n]
        offset += n
        loss = loss + torch.nn.functional.cross_entropy(chunk.unsqueeze(0), gold[i].unsqueeze(0))
    return loss / max(1, len(n_opts))


def train_option_head(
    examples: Sequence[Any],
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    encoder: nn.Module | None = None,
    val_examples: Sequence[Any] | None = None,
) -> tuple[nn.Module, OptionAttentionHead, dict[str, float]]:
    """Train only the option head. `val_examples` is for early stopping; never pass calibration."""
    cfg = config or TrainConfig()
    device = device or torch.device("cpu")
    torch.manual_seed(cfg.seed)
    if encoder is None:
        if cfg.encoder_kind == "hf":
            from jev.scoring.hf_encoder import FrozenHFEncoder

            encoder = FrozenHFEncoder(
                cfg.hf_model_id,
                max_len=cfg.max_state_tokens,
                device=device,
            )
        else:
            encoder = HashingEncoder(d_model=cfg.d_model, seed=cfg.seed).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    d_model = int(getattr(encoder, "d_model", cfg.d_model))
    head = OptionAttentionHead(d_model, rank=cfg.rank).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
    history: dict[str, float] = {
        "final_loss": 0.0,
        "best_val_acc": 0.0,
        "stopped_epoch": float(cfg.epochs),
    }
    steps = 0
    best_state = None
    best_val = -1.0
    wait = 0
    order = list(range(len(examples)))
    for epoch in range(cfg.epochs):
        head.train()
        g = torch.Generator().manual_seed(cfg.seed + epoch)
        perm = torch.randperm(len(order), generator=g).tolist()
        for start in range(0, len(examples), cfg.batch_size):
            batch_idx = perm[start : start + cfg.batch_size]
            batch = [examples[i] for i in batch_idx]
            with torch.no_grad():
                state_h, state_m, opt_h, opt_m, idx, gold, n_opts = _collate(encoder, batch, device)
            logits = head(state_h, state_m, opt_h, opt_m, idx)
            loss = listwise_ce(logits, n_opts, gold)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history["final_loss"] = float(loss.detach().cpu())
            steps += 1
            if cfg.max_steps is not None and steps >= cfg.max_steps:
                head.eval()
                history["stopped_epoch"] = float(epoch)
                return encoder, head, history
        if val_examples:
            acc = accuracy_on_examples(encoder, head, val_examples, device=device)
            if acc > best_val:
                best_val = acc
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    history["stopped_epoch"] = float(epoch)
                    break
    if best_state is not None:
        head.load_state_dict(best_state)
    history["best_val_acc"] = float(max(best_val, 0.0))
    head.eval()
    return encoder, head, history


@torch.no_grad()
def accuracy_on_examples(
    encoder: nn.Module,
    head: OptionAttentionHead,
    examples: Sequence[Any],
    device: torch.device | None = None,
    shuffle_state: bool = False,
    seed: int = 0,
) -> float:
    device = device or torch.device("cpu")
    if not examples:
        return 0.0
    states = [example_state_text(ex) for ex in examples]
    if shuffle_state:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(states), generator=g).tolist()
        states = [states[i] for i in perm]
    correct = 0
    for ex, state in zip(examples, states, strict=True):
        opts = example_option_texts(ex)
        gold = example_gold_index(ex)
        state_h, state_m = encoder.forward_texts([state], device)
        opt_h, opt_m = encoder.forward_texts(opts, device)
        idx = torch.zeros(len(opts), dtype=torch.long, device=device)
        logits = head(state_h.float(), state_m, opt_h.float(), opt_m, idx)
        pred = int(logits.argmax().item())
        correct += int(pred == gold)
    return correct / len(examples)


def save_checkpoint(
    path: Path,
    encoder: nn.Module,
    head: OptionAttentionHead,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(extra or {})
    encoder_kind = extra.get("encoder_kind") or (
        "hf" if encoder.__class__.__name__ == "FrozenHFEncoder" else "hashing"
    )
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "encoder_kind": encoder_kind,
        "head": head.state_dict(),
        "d_model": int(getattr(encoder, "d_model", head.d_model)),
        "rank": head.rank,
        "model_id": extra.get("model_id") or getattr(encoder, "model_id", "option-attention"),
        "extra": extra,
    }
    if encoder_kind == "hashing":
        payload["encoder"] = encoder.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path, device: torch.device | None = None
) -> tuple[nn.Module, OptionAttentionHead, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra = dict(payload.get("extra") or {})
    encoder_kind = payload.get("encoder_kind") or extra.get("encoder_kind") or "hashing"
    if device is None:
        device = torch.device(
            "cuda" if encoder_kind == "hf" and torch.cuda.is_available() else "cpu"
        )
    d_model = int(payload["d_model"])
    if encoder_kind == "hf":
        from jev.scoring.hf_encoder import FrozenHFEncoder

        model_id = str(payload.get("model_id") or extra.get("hf_model_id") or "Qwen/Qwen2.5-0.5B")
        encoder = FrozenHFEncoder(model_id, device=device)
    else:
        encoder = HashingEncoder(d_model=d_model)
        encoder.load_state_dict(payload["encoder"])
        encoder.to(device)
    head = OptionAttentionHead(d_model, rank=int(payload["rank"]))
    head.load_state_dict(payload["head"])
    encoder.eval()
    head.to(device).eval()
    meta = {
        "encoder_kind": encoder_kind,
        "model_id": payload.get("model_id"),
        "extra": extra,
    }
    return encoder, head, meta
