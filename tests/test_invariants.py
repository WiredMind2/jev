
import pytest

from jev.invariants import (
    assert_choice_answer,
    assert_noul_answer,
    assert_score_answer,
    build_choice_answer,
    build_noul_answer,
    build_score_answer,
    entropy_confidence,
    softmax,
)
from jev.schema import NoulAnswer


def test_softmax_sums_to_one() -> None:
    probs = softmax([1.0, 2.0, 3.0])
    assert abs(sum(probs) - 1.0) < 1e-9


def test_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], temperature=0.0)


def test_choice_argmax_and_confidence() -> None:
    ans = build_choice_answer(["a", "b", "c"], [0.0, 5.0, 0.1])
    assert ans.choice == "b"
    assert_choice_answer(ans)
    assert 0.0 <= ans.confidence <= 1.0
    uniform = entropy_confidence([1 / 3, 1 / 3, 1 / 3])
    assert uniform < 1e-9


def test_score_is_expectation() -> None:
    ans = build_score_answer(["low", "mid", "high"], [0.0, 0.0, 10.0])
    assert_score_answer(ans)
    assert ans.score == pytest.approx(2.0, abs=1e-3)
    assert set(ans.legend) == {"0", "1", "2"}


def test_noul_has_no_confidence_field() -> None:
    ans = build_noul_answer(4.0, 0.0)
    assert_noul_answer(ans)
    dumped = ans.model_dump()
    assert "confidence" not in dumped
    assert 0.5 < ans.noul <= 1.0


def test_noul_schema_forbids_confidence() -> None:
    with pytest.raises(Exception):
        NoulAnswer(noul=0.2, confidence=0.9)  # type: ignore[call-arg]
