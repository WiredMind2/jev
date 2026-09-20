"""Request -> scorer logits -> temperature -> typed System One response."""

from __future__ import annotations

from jev.invariants import build_choice_answer, build_noul_answer, build_score_answer
from jev.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from jev.scoring.protocol import ScoredQuestion, Scorer


def answer_from_scored(item: ScoredQuestion, question, temperature: float = 1.0):
    if item.type == "choice" or isinstance(question, ChoiceQuestion):
        return build_choice_answer(item.keys, item.logits, temperature=temperature)
    if item.type == "score" or isinstance(question, ScoreQuestion):
        levels = list(question.criteria) if isinstance(question, ScoreQuestion) else item.keys
        return build_score_answer(levels, item.logits, temperature=temperature)
    if item.type == "noul" or isinstance(question, NoulQuestion):
        keyed = dict(zip(item.keys, item.logits, strict=True))
        return build_noul_answer(keyed["true"], keyed["false"], temperature=temperature)
    raise TypeError(f"unsupported scored type {item.type}")


def merge_usage(parts: list[ScoredQuestion]) -> Usage:
    return Usage(
        input_tokens=sum(p.usage.input_tokens for p in parts),
        output_tokens=sum(p.usage.output_tokens for p in parts),
    )


def respond(
    scorer: Scorer,
    request: SystemOneRequest,
    *,
    temperature: float = 1.0,
    model_id: str | None = None,
) -> SystemOneResponse:
    scored = scorer.score_request(request)
    answers = {}
    for item in scored:
        question = request.questions[item.question_id]
        answers[item.question_id] = answer_from_scored(item, question, temperature=temperature)
    missing = set(request.questions) - set(answers)
    if missing:
        raise ValueError(f"scorer omitted questions: {sorted(missing)}")
    return SystemOneResponse(
        model=model_id or scorer.model_id,
        answers=answers,
        usage=merge_usage(scored),
    )
