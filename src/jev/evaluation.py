"""Metrics for finite-option decisions. Test set is frozen until reporting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from jev.invariants import entropy_confidence, score_expectation, softmax
from jev.schema import ScoreTrainingExample, SystemOneRequest
from jev.scoring.option_head import example_gold_index
from jev.scoring.protocol import Scorer


@dataclass
class MetricReport:
    n: int
    accuracy: float
    nll: float
    brier: float
    ece: float
    shuffled_accuracy: float | None = None
    coverage_at_1pct: float | None = None
    extras: dict[str, float] = field(default_factory=dict)


def _nll(probs: Sequence[float], gold: int) -> float:
    return -math.log(max(probs[gold], 1e-12))


def _brier(probs: Sequence[float], gold: int) -> float:
    acc = 0.0
    for i, p in enumerate(probs):
        target = 1.0 if i == gold else 0.0
        acc += (p - target) ** 2
    return acc


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    if not confidences:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correct, strict=True):
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append((c, y))
    n = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        acc = sum(y for _, y in bucket) / len(bucket)
        conf = sum(c for c, _ in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - conf)
    return ece


def coverage_at_error(
    confidences: Sequence[float],
    correct: Sequence[int],
    *,
    max_error: float = 0.01,
) -> float:
    """Largest coverage whose empirical error is <= max_error when keeping high confidence."""
    if not confidences:
        return 0.0
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    kept_err = 0
    best = 0.0
    for k, i in enumerate(order, start=1):
        kept_err += 0 if correct[i] else 1
        err = kept_err / k
        if err <= max_error:
            best = k / len(order)
    return best


def evaluate_scorer(
    scorer: Scorer,
    examples: Sequence,
    *,
    temperature: float = 1.0,
    shuffled: bool = True,
    shuffle_seed: int = 0,
) -> MetricReport:
    import random

    nlls: list[float] = []
    briers: list[float] = []
    hits: list[int] = []
    confs: list[float] = []
    maes: list[float] = []
    peaked: list[float] = []
    for ex in examples:
        request = SystemOneRequest(state=ex.state, questions={"q": ex.question})
        scored = scorer.score_request(request)[0]
        gold = example_gold_index(ex)
        probs = softmax(scored.logits, temperature=temperature)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        hits.append(int(pred == gold))
        nlls.append(_nll(probs, gold))
        briers.append(_brier(probs, gold))
        # ECE and risk–coverage use predicted-class confidence (p_max).
        confs.append(max(probs))
        peaked.append(entropy_confidence(probs) if len(probs) >= 2 else max(probs))
        if isinstance(ex, ScoreTrainingExample):
            maes.append(abs(score_expectation(probs) - float(gold)))
    shuffled_acc = None
    if shuffled and examples:
        rng = random.Random(shuffle_seed)
        states = [ex.state for ex in examples]
        perm = states[:]
        rng.shuffle(perm)
        sh_hits = 0
        for ex, state in zip(examples, perm, strict=True):
            request = SystemOneRequest(state=state, questions={"q": ex.question})
            scored = scorer.score_request(request)[0]
            gold = example_gold_index(ex)
            pred = max(range(len(scored.logits)), key=lambda i: scored.logits[i])
            sh_hits += int(pred == gold)
        shuffled_acc = sh_hits / len(examples)
    n = max(1, len(examples))
    extras: dict[str, float] = {"mean_entropy_confidence": sum(peaked) / n if peaked else 0.0}
    if maes:
        extras["mae"] = sum(maes) / len(maes)
    return MetricReport(
        n=len(examples),
        accuracy=sum(hits) / n,
        nll=sum(nlls) / n,
        brier=sum(briers) / n,
        ece=expected_calibration_error(confs, hits),
        shuffled_accuracy=shuffled_acc,
        coverage_at_1pct=coverage_at_error(confs, hits, max_error=0.01),
        extras=extras,
    )


def report_as_dict(report: MetricReport) -> dict[str, float | int | None]:
    return {
        "n": report.n,
        "accuracy": report.accuracy,
        "nll": report.nll,
        "brier": report.brier,
        "ece": report.ece,
        "shuffled_accuracy": report.shuffled_accuracy,
        "coverage_at_1pct": report.coverage_at_1pct,
        **report.extras,
    }
