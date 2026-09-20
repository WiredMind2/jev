"""Scorer protocol: state + questions -> per-question logits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from jev.schema import SystemOneRequest, Usage


@dataclass
class ScoredQuestion:
    question_id: str
    type: str
    keys: list[str]
    logits: list[float]
    usage: Usage = field(default_factory=Usage)


@runtime_checkable
class Scorer(Protocol):
    """Finite-option scorer. Model execution is separate from policy."""

    @property
    def model_id(self) -> str: ...

    def score_request(self, request: SystemOneRequest) -> list[ScoredQuestion]: ...
