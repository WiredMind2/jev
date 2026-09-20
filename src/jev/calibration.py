"""Post-hoc temperature scaling. Fit only on the calibration split."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from jev.invariants import softmax
from jev.schema import ChoiceTrainingExample, NoulTrainingExample, ScoreTrainingExample
from jev.scoring.option_head import example_gold_index
from jev.scoring.protocol import Scorer


@dataclass
class TemperatureCalibrator:
    temperature: float = 1.0

    def apply(self, logits: Sequence[float]) -> list[float]:
        return softmax(logits, temperature=self.temperature)


def nll_from_logits(logits: Sequence[float], gold: int, temperature: float) -> float:
    probs = softmax(logits, temperature=temperature)
    p = max(probs[gold], 1e-12)
    return -math.log(p)


def fit_temperature(
    pairs: Sequence[tuple[list[float], int]],
    *,
    grid: Sequence[float] | None = None,
) -> TemperatureCalibrator:
    """Minimize NLL over a positive temperature grid. Ranking is preserved."""
    if not pairs:
        return TemperatureCalibrator(1.0)
    candidates = list(grid) if grid is not None else [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    best_t = 1.0
    best_nll = float("inf")
    for t in candidates:
        if t <= 0:
            continue
        nll = sum(nll_from_logits(logits, gold, t) for logits, gold in pairs) / len(pairs)
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return TemperatureCalibrator(temperature=best_t)


def collect_logit_gold(
    scorer: Scorer,
    examples: Sequence[ChoiceTrainingExample | ScoreTrainingExample | NoulTrainingExample],
) -> list[tuple[list[float], int]]:
    from jev.schema import SystemOneRequest

    pairs: list[tuple[list[float], int]] = []
    for ex in examples:
        request = SystemOneRequest(state=ex.state, questions={"q": ex.question})
        scored = scorer.score_request(request)[0]
        pairs.append((list(scored.logits), example_gold_index(ex)))
    return pairs
