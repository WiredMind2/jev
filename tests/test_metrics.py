from jev.calibration import fit_temperature, nll_from_logits
from jev.evaluation import (
    binary_auroc,
    coverage_at_error,
    evaluate_scorer,
    expected_calibration_error,
    risk_coverage_curve,
)
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


def test_risk_coverage_curve_monotone_coverage() -> None:
    conf = [0.9, 0.8, 0.2, 0.1]
    correct = [1, 1, 0, 0]
    curve = risk_coverage_curve(conf, correct, fractions=(0.5, 1.0))
    assert curve[0]["n_kept"] == 2.0
    assert curve[0]["risk"] == 0.0
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["risk"] == 0.5


def test_binary_auroc_separates() -> None:
    assert binary_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert binary_auroc([0.5, 0.5], [1, 1]) is None


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
    curve = report.extras["risk_coverage"]
    assert isinstance(curve, list) and curve
    assert curve[-1]["coverage"] == 1.0
    dumped = __import__("jev.evaluation", fromlist=["report_as_dict"]).report_as_dict(report)
    assert "risk_coverage" in dumped
