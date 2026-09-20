from pathlib import Path

import pytest
from pydantic import ValidationError

from jev.canonical import canonical_dumps, sha256_json
from jev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest, parse_request


def test_choice_rejects_one_option() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(type="choice", instructions="x", criteria={"a": "only"})


def test_choice_rejects_256_options() -> None:
    criteria = {f"k{i}": f"d{i}" for i in range(256)}
    with pytest.raises(ValidationError):
        ChoiceQuestion(type="choice", instructions="x", criteria=criteria)


def test_score_rejects_11_levels() -> None:
    with pytest.raises(ValidationError):
        ScoreQuestion(type="score", instructions="x", criteria=["a"] * 11)


def test_request_requires_questions() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest(state="s", questions={})


def test_canonical_sorts_keys() -> None:
    a = canonical_dumps({"b": 1, "a": 2})
    b = canonical_dumps({"a": 2, "b": 1})
    assert a == b == '{"a":2,"b":1}'
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})


def test_example_request_parses() -> None:
    path = Path("examples/systemone.request.json")
    req = parse_request(__import__("json").loads(path.read_text()))
    assert set(req.questions) == {"department", "frustration", "is_urgent"}
    assert isinstance(req.questions["is_urgent"], NoulQuestion)
