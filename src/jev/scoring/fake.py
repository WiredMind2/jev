"""Deterministic fake scorer for CPU tests (no GPU, no HF downloads).

Scores are token-overlap between canonical state text and each option
description. Golden fixtures pin the numeric logits.
"""

from __future__ import annotations

import math
import re

from jev.render import option_keys, option_texts, render_state
from jev.schema import SystemOneRequest, Usage
from jev.scoring.protocol import ScoredQuestion, Scorer

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def overlap_logit(state_tokens: set[str], option_text: str) -> float:
    option_tokens = tokenize(option_text)
    if not option_tokens:
        return 0.0
    hit = len(state_tokens & option_tokens)
    return math.log1p(hit) + 0.05 * hit


class FakeScorer:
    def __init__(self, model_id: str = "fake-overlap-v0") -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def score_request(self, request: SystemOneRequest) -> list[ScoredQuestion]:
        state_tokens = tokenize(render_state(request.state))
        out: list[ScoredQuestion] = []
        for qid, question in request.questions.items():
            keys = option_keys(question)
            texts = option_texts(question)
            logits = [overlap_logit(state_tokens, text) for text in texts]
            n_in = max(1, len(state_tokens))
            n_out = sum(max(1, len(tokenize(t))) for t in texts)
            out.append(
                ScoredQuestion(
                    question_id=qid,
                    type=question.type,
                    keys=keys,
                    logits=logits,
                    usage=Usage(input_tokens=n_in, output_tokens=n_out),
                )
            )
        return out


def default_fake_scorer() -> Scorer:
    return FakeScorer()
