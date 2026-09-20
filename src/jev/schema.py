"""Runtime Pydantic models matching schemas/ plus training-row invariants."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonValue = Any
FORMAT_VERSION = "1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChoiceQuestion(_Strict):
    type: Literal["choice"]
    instructions: JsonValue
    criteria: dict[str, JsonValue]

    @field_validator("criteria")
    @classmethod
    def _choice_arity(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        n = len(value)
        if n < 2 or n > 255:
            raise ValueError("choice criteria must contain 2..255 keys")
        if any(not str(k) for k in value):
            raise ValueError("choice keys must be non-empty strings")
        return value


class ScoreQuestion(_Strict):
    type: Literal["score"]
    instructions: JsonValue
    criteria: list[JsonValue]

    @field_validator("criteria")
    @classmethod
    def _score_arity(cls, value: list[JsonValue]) -> list[JsonValue]:
        n = len(value)
        if n < 2 or n > 10:
            raise ValueError("score criteria must contain 2..10 levels")
        return value


class NoulCriteria(_Strict):
    true: JsonValue | None = None
    false: JsonValue | None = None


class NoulQuestion(_Strict):
    type: Literal["noul"]
    instructions: JsonValue
    criteria: NoulCriteria | None = None


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(_Strict):
    model: str | None = None
    state: JsonValue
    questions: dict[str, Question]

    @field_validator("questions")
    @classmethod
    def _nonempty_questions(cls, value: dict[str, Question]) -> dict[str, Question]:
        if not value:
            raise ValueError("questions must contain at least one item")
        return value


class ChoiceAnswer(_Strict):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(_Strict):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, JsonValue]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(_Strict):
    type: Literal["noul"] = "noul"
    noul: float


Answer = Annotated[
    ChoiceAnswer | ScoreAnswer | NoulAnswer,
    Field(discriminator="type"),
]


class Usage(_Strict):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(_Strict):
    model: str
    answers: dict[str, Answer]
    usage: Usage | None = None


class ExampleMetadata(_Strict):
    model_config = ConfigDict(extra="allow")

    domain: str
    group_id: str | None = None
    risk_tier: str | None = None
    split: Literal["train", "validation", "calibration", "test"] | None = None
    source: str | None = None
    candidate_sha256: str | None = None


class ChoiceTrainingExample(_Strict):
    id: str
    type: Literal["choice"] = "choice"
    format_version: str = FORMAT_VERSION
    criteria_version: str
    state: JsonValue
    question: ChoiceQuestion
    gold: str
    metadata: ExampleMetadata

    @model_validator(mode="after")
    def _gold_in_menu(self) -> ChoiceTrainingExample:
        if self.gold not in self.question.criteria:
            raise ValueError(f"gold {self.gold!r} is not present in criteria")
        return self


class ScoreTrainingExample(_Strict):
    id: str
    type: Literal["score"] = "score"
    format_version: str = FORMAT_VERSION
    criteria_version: str
    state: JsonValue
    question: ScoreQuestion
    gold: int
    metadata: ExampleMetadata

    @model_validator(mode="after")
    def _gold_in_range(self) -> ScoreTrainingExample:
        k = len(self.question.criteria)
        if not 0 <= self.gold < k:
            raise ValueError(f"gold {self.gold} out of range for {k} score levels")
        return self


class NoulTrainingExample(_Strict):
    id: str
    type: Literal["noul"] = "noul"
    format_version: str = FORMAT_VERSION
    criteria_version: str
    state: JsonValue
    question: NoulQuestion
    gold: bool
    metadata: ExampleMetadata


TrainingExample = Annotated[
    ChoiceTrainingExample | ScoreTrainingExample | NoulTrainingExample,
    Field(discriminator="type"),
]


def parse_training_example(data: dict[str, Any]) -> (
    ChoiceTrainingExample | ScoreTrainingExample | NoulTrainingExample
):
    kind = data.get("type")
    if kind == "choice":
        return ChoiceTrainingExample.model_validate(data)
    if kind == "score":
        return ScoreTrainingExample.model_validate(data)
    if kind == "noul":
        return NoulTrainingExample.model_validate(data)
    raise ValueError(f"unknown training example type: {kind!r}")


def parse_request(data: dict[str, Any]) -> SystemOneRequest:
    return SystemOneRequest.model_validate(data)
