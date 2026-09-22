from jev.calibration import fit_temperature, nll_from_logits
from jev.evaluation import coverage_at_error, evaluate_scorer, expected_calibration_error, risk_coverage_curve
from jev.scoring.fake import FakeScorer


def test_temperature_fit_prefers_better_nll() -> None:
    pairs = [([5.0, 0.0], 0), ([4.0, 0.1], 0), ([0.0, 3.0], 1)]
    cal = fit_temperature(pairs)
    assert cal.temperature > 0
    base = sum(nll_from_logits(z, y, 1.0) for z, y in pairs)
    fitted = sum(nll_from_logits(z, y, cal.temperature) for z, y in pairs)
    assert fitted <= base + 1e-9


def test_ece_zero_when_perfect() -> None:
    assert expected_calibration_error([1.0, 1.0, 1.0], [1, 1, 1]) == 0.0


def test_coverage_at_error_keeps_confident_correct() -> None:
    conf = [0.99, 0.98, 0.2, 0.1]
    correct = [1, 1, 0, 0]
    cov = coverage_at_error(conf, correct, max_error=0.01)
    assert cov == 0.5


def test_evaluate_scorer_metrics_keys() -> None:
    from jev.data.synthetic import make_synthetic_choice

    examples = make_synthetic_choice(n=12, seed=4)
    report = evaluate_scorer(FakeScorer(), examples, shuffled=True)
    assert report.n == 12
    assert 0.0 <= report.accuracy <= 1.0
    assert report.nll >= 0.0
    assert report.brier >= 0.0
    assert report.ece >= 0.0
    assert report.shuffled_accuracy is not None
    assert report.coverage_at_1pct is not None
    assert report.risk_coverage
    assert report.risk_coverage[0]["threshold"] == 0.0
    assert "coverage" in report.risk_coverage[0]


def test_evaluate_scorer_progress_ticks() -> None:
    from jev.data.synthetic import make_synthetic_choice

    seen: list[tuple[str, int, int]] = []
    examples = make_synthetic_choice(n=12, seed=5)
    evaluate_scorer(
        FakeScorer(),
        examples,
        shuffled=True,
        progress=lambda stage, i, n: seen.append((stage, i, n)),
    )
    assert ("eval", 12, 12) in seen
    assert ("eval", 1, 12) in seen
    assert ("shuffled", 12, 12) in seen


class _BudgetScorer:
    def __init__(self, inner: FakeScorer, limit: int) -> None:
        self.inner = inner
        self.limit = limit
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def score_request(self, request):
        self.calls += 1
        if self.calls > self.limit:
            raise RuntimeError("interrupt")
        return self.inner.score_request(request)


def test_evaluate_scorer_resumes_from_logit_cache(tmp_path) -> None:
    from jev.data.synthetic import make_synthetic_choice

    examples = make_synthetic_choice(n=12, seed=6)
    cache = tmp_path / "eval.partial.jsonl"
    first = _BudgetScorer(FakeScorer(), limit=4)
    try:
        evaluate_scorer(first, examples, shuffled=False, cache_path=cache)
    except RuntimeError:
        pass
    assert cache.is_file()
    assert first.calls == 5
    second = _BudgetScorer(FakeScorer(), limit=20)
    report = evaluate_scorer(second, examples, shuffled=False, cache_path=cache)
    assert second.calls == 8
    full = evaluate_scorer(FakeScorer(), examples, shuffled=False)
    assert report.accuracy == full.accuracy
    assert report.nll == full.nll


def test_risk_coverage_zero_when_all_correct_and_confident() -> None:
    conf = [1.0, 1.0, 1.0]
    correct = [1, 1, 1]
    curve = risk_coverage_curve(conf, correct)
    full = next(row for row in curve if row["threshold"] == 0.0)
    assert full["coverage"] == 1.0
    assert full["risk"] == 0.0
