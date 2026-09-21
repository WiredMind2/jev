"""Non-LM and JSON-LLM baselines for the v0 report table. Not research substitutes for Qwen."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from jev.evaluation import gold_key
from jev.render import json_continuations, option_keys, render_prefix, render_state
from jev.schema import SystemOneRequest, Usage
from jev.scoring.logprob import LogprobScorer
from jev.scoring.protocol import ScoredQuestion


class MajorityScorer:
    """Class-prior logits from the train split. Practical non-AI floor."""

    def __init__(self, counts: dict[str, int], model_id: str = "majority-prior") -> None:
        self.counts = dict(counts)
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @classmethod
    def fit(cls, examples: Sequence[Any]) -> MajorityScorer:
        counts: Counter[str] = Counter(gold_key(ex) for ex in examples)
        return cls(dict(counts))

    def score_request(self, request: SystemOneRequest) -> list[ScoredQuestion]:
        out: list[ScoredQuestion] = []
        for qid, question in request.questions.items():
            keys = option_keys(question)
            logits = [float(self.counts.get(k, 0)) for k in keys]
            out.append(
                ScoredQuestion(
                    question_id=qid,
                    type=question.type,
                    keys=keys,
                    logits=logits,
                    usage=Usage(input_tokens=1, output_tokens=len(keys)),
                )
            )
        return out


class TfidfLinearScorer:
    """TF-IDF + logistic regression on state text. Encoder-classifier baseline."""

    def __init__(self, vectorizer: Any, clf: Any, model_id: str = "tfidf-linear") -> None:
        self.vectorizer = vectorizer
        self.clf = clf
        self._model_id = model_id
        self.classes_ = [str(c) for c in clf.classes_]

    @property
    def model_id(self) -> str:
        return self._model_id

    @classmethod
    def fit(cls, examples: Sequence[Any]) -> TfidfLinearScorer:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts = [render_state(ex.state) for ex in examples]
        labels = [gold_key(ex) for ex in examples]
        n_classes = len({lab for lab in labels})
        if n_classes < 2:
            raise ValueError(f"tfidf-linear needs at least 2 gold classes, got {n_classes}")
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
        x = vectorizer.fit_transform(texts)
        clf = LogisticRegression(max_iter=200, C=2.0)
        clf.fit(x, labels)
        return cls(vectorizer, clf)

    def score_request(self, request: SystemOneRequest) -> list[ScoredQuestion]:
        import numpy as np

        text = render_state(request.state)
        x = self.vectorizer.transform([text])
        log_proba = self.clf.predict_log_proba(x)[0]
        by_class = {cls: float(p) for cls, p in zip(self.classes_, log_proba, strict=True)}
        floor = float(np.min(log_proba) - 10.0) if len(log_proba) else -20.0
        out: list[ScoredQuestion] = []
        for qid, question in request.questions.items():
            keys = option_keys(question)
            logits = [by_class.get(k, floor) for k in keys]
            out.append(
                ScoredQuestion(
                    question_id=qid,
                    type=question.type,
                    keys=keys,
                    logits=logits,
                    usage=Usage(input_tokens=max(1, len(text.split())), output_tokens=len(keys)),
                )
            )
        return out


class JsonLmScorer:
    """Score JSON object continuations with a causal LM. Local JSON-LLM baseline."""

    def __init__(self, inner: LogprobScorer) -> None:
        self.inner = inner

    @property
    def model_id(self) -> str:
        return f"json-llm:{self.inner.model_id}"

    def score_request(self, request: SystemOneRequest) -> list[ScoredQuestion]:
        out: list[ScoredQuestion] = []
        for qid, question in request.questions.items():
            prefix = render_prefix(request.state, question) + "\nJSON:"
            conts = json_continuations(question)
            logits = self.inner.score_strings(prefix, conts)
            n_in = len(self.inner._encode(prefix))
            n_out = sum(len(self.inner._encode(c)) for c in conts)
            out.append(
                ScoredQuestion(
                    question_id=qid,
                    type=question.type,
                    keys=option_keys(question),
                    logits=logits,
                    usage=Usage(input_tokens=n_in, output_tokens=n_out),
                )
            )
        return out
