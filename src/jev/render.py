"""Render state + question into a shared prefix and option continuations."""

from __future__ import annotations

from typing import Any

from jev.canonical import canonical_dumps
from jev.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, SystemOneRequest


def render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return canonical_dumps(value)


def render_state(state: Any) -> str:
    return render_value(state)


def render_prefix(state: Any, question: Question) -> str:
    state_text = render_state(state)
    instructions = render_value(question.instructions)
    if isinstance(question, ChoiceQuestion):
        option_block = "\n".join(
            f"- {key}: {render_value(desc)}" for key, desc in question.criteria.items()
        )
        return (
            f"State:\n{state_text}\n\n"
            f"Question type: choice\n"
            f"Instructions: {instructions}\n"
            f"Options:\n{option_block}\n\n"
            "Select the single best option key:"
        )
    if isinstance(question, ScoreQuestion):
        level_block = "\n".join(
            f"- {i}: {render_value(level)}" for i, level in enumerate(question.criteria)
        )
        return (
            f"State:\n{state_text}\n\n"
            f"Question type: score\n"
            f"Instructions: {instructions}\n"
            f"Levels (evaluated independently):\n{level_block}\n\n"
            "The matching level is:"
        )
    if isinstance(question, NoulQuestion):
        true_t = "true"
        false_t = "false"
        if question.criteria is not None:
            if question.criteria.true is not None:
                true_t = f"true ({render_value(question.criteria.true)})"
            if question.criteria.false is not None:
                false_t = f"false ({render_value(question.criteria.false)})"
        return (
            f"State:\n{state_text}\n\n"
            f"Question type: noul\n"
            f"Instructions: {instructions}\n"
            f"Decide whether the proposition holds ({false_t} / {true_t}).\n\n"
            "Answer:"
        )
    raise TypeError(f"unsupported question type: {question.type}")


def option_keys(question: Question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria)
    if isinstance(question, ScoreQuestion):
        return [str(i) for i in range(len(question.criteria))]
    if isinstance(question, NoulQuestion):
        return ["false", "true"]
    raise TypeError(f"unsupported question type: {question.type}")


def option_texts(question: Question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return [f"{k}: {render_value(v)}" for k, v in question.criteria.items()]
    if isinstance(question, ScoreQuestion):
        return [f"{i}: {render_value(level)}" for i, level in enumerate(question.criteria)]
    if isinstance(question, NoulQuestion):
        true_t = "true"
        false_t = "false"
        if question.criteria is not None:
            if question.criteria.true is not None:
                true_t = render_value(question.criteria.true)
            if question.criteria.false is not None:
                false_t = render_value(question.criteria.false)
        return [f"false: {false_t}", f"true: {true_t}"]
    raise TypeError("unsupported question type")


def continuations(question: Question) -> list[str]:
    """Short labels scored as causal continuations after the prefix."""
    if isinstance(question, ChoiceQuestion):
        return [f" {key}" for key in question.criteria]
    if isinstance(question, ScoreQuestion):
        return [f" {i}" for i in range(len(question.criteria))]
    if isinstance(question, NoulQuestion):
        return [" false", " true"]
    raise TypeError("unsupported question type")


def render_request_prefixes(request: SystemOneRequest) -> dict[str, str]:
    return {qid: render_prefix(request.state, q) for qid, q in request.questions.items()}
