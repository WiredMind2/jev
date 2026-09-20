"""Distribution invariants and System One answer builders.

Choice/Score confidence is normalized entropy (open approximation, not a
TypeSafe-published formula):

    confidence(p) = 1 - H(p) / log(K)

Noul has no separate confidence field; the scalar `noul` is P(true).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from jev.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer

PROB_ATOL = 1e-5


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [float(z) / temperature for z in logits]
    peak = max(scaled)
    exps = [math.exp(z - peak) for z in scaled]
    total = sum(exps)
    if total <= 0:
        raise ValueError("softmax denominator is not positive")
    return [e / total for e in exps]


def entropy(probs: Sequence[float]) -> float:
    acc = 0.0
    for p in probs:
        if p <= 0:
            continue
        acc -= p * math.log(p)
    return acc


def entropy_confidence(probs: Sequence[float]) -> float:
    """Decisiveness in [0, 1]: 0 if uniform, 1 if one-hot. Not calibration."""
    k = len(probs)
    if k < 2:
        raise ValueError("confidence requires at least two outcomes")
    h = entropy(probs)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def assert_prob_simplex(probs: Sequence[float], *, atol: float = PROB_ATOL) -> None:
    if any(p < -atol or p > 1.0 + atol for p in probs):
        raise ValueError("probabilities must lie in [0, 1]")
    total = sum(probs)
    if abs(total - 1.0) > atol:
        raise ValueError(f"probabilities must sum to 1, got {total}")


def argmax_index(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("empty values")
    best = 0
    best_v = values[0]
    for i, v in enumerate(values):
        if v > best_v:
            best = i
            best_v = v
    return best


def score_expectation(probs: Sequence[float]) -> float:
    return sum(i * p for i, p in enumerate(probs))


def noul_probability(true_logit: float, false_logit: float, temperature: float = 1.0) -> float:
    return softmax([false_logit, true_logit], temperature=temperature)[1]


def build_choice_answer(
    keys: Sequence[str],
    logits: Sequence[float],
    *,
    temperature: float = 1.0,
) -> ChoiceAnswer:
    if len(keys) != len(logits):
        raise ValueError("keys and logits length mismatch")
    probs = softmax(logits, temperature=temperature)
    assert_prob_simplex(probs)
    idx = argmax_index(probs)
    mapping = {k: p for k, p in zip(keys, probs, strict=True)}
    return ChoiceAnswer(
        choice=keys[idx],
        probabilities=mapping,
        confidence=entropy_confidence(probs),
    )


def build_score_answer(
    levels: Sequence[object],
    logits: Sequence[float],
    *,
    temperature: float = 1.0,
) -> ScoreAnswer:
    if len(levels) != len(logits):
        raise ValueError("levels and logits length mismatch")
    probs = softmax(logits, temperature=temperature)
    assert_prob_simplex(probs)
    legend = {str(i): level for i, level in enumerate(levels)}
    keyed = {str(i): p for i, p in enumerate(probs)}
    return ScoreAnswer(
        score=score_expectation(probs),
        legend=legend,
        probabilities=keyed,
        confidence=entropy_confidence(probs),
    )


def build_noul_answer(
    true_logit: float,
    false_logit: float,
    *,
    temperature: float = 1.0,
) -> NoulAnswer:
    p_true = noul_probability(true_logit, false_logit, temperature=temperature)
    if not 0.0 - PROB_ATOL <= p_true <= 1.0 + PROB_ATOL:
        raise ValueError("noul probability out of range")
    return NoulAnswer(noul=float(p_true))


def assert_choice_answer(answer: ChoiceAnswer) -> None:
    probs = list(answer.probabilities.values())
    assert_prob_simplex(probs)
    expected = max(answer.probabilities, key=answer.probabilities.get)
    if answer.choice != expected:
        raise ValueError("choice is not argmax of probabilities")
    expected_c = entropy_confidence(probs)
    if abs(answer.confidence - expected_c) > 1e-4:
        raise ValueError("choice confidence is not normalized entropy")


def assert_score_answer(answer: ScoreAnswer) -> None:
    keys = sorted(answer.probabilities, key=lambda k: int(k))
    probs = [answer.probabilities[k] for k in keys]
    assert_prob_simplex(probs)
    expected = score_expectation(probs)
    if abs(answer.score - expected) > 1e-5:
        raise ValueError("score is not the expectation of level indices")
    if set(answer.legend) != set(answer.probabilities):
        raise ValueError("score legend keys must match probability keys")
    expected_c = entropy_confidence(probs)
    if abs(answer.confidence - expected_c) > 1e-4:
        raise ValueError("score confidence is not normalized entropy")


def assert_noul_answer(answer: NoulAnswer) -> None:
    if not 0.0 <= answer.noul <= 1.0:
        raise ValueError("noul must be in [0, 1]")
    extra = getattr(answer, "model_extra", None) or {}
    if "confidence" in extra:
        raise ValueError("noul must not include a separate confidence field")


def keys_and_logits_from_mapping(mapping: Mapping[str, float]) -> tuple[list[str], list[float]]:
    keys = list(mapping)
    return keys, [float(mapping[k]) for k in keys]
