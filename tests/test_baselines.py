from jev.data.splits import assign_groups
from jev.data.synthetic import make_synthetic_choice
from jev.evaluation import evaluate_scorer, stratified_sample
from jev.scoring.baselines import MajorityScorer, TfidfLinearScorer


def test_assign_groups_fills_val_and_calib() -> None:
    labeled = assign_groups([f"g{i}" for i in range(8)], seed=0)
    assert set(labeled.values()) >= {"train", "validation", "calibration"}


def test_stratified_sample_caps_and_covers() -> None:
    rows = make_synthetic_choice(n=40, k=4, seed=0)
    sample = stratified_sample(rows, 12, seed=1)
    assert len(sample) == 12
    golds = {r.gold for r in sample}
    assert golds


def test_majority_and_tfidf_on_synthetic() -> None:
    rows = make_synthetic_choice(n=48, k=4, seed=2)
    maj = MajorityScorer.fit(rows)
    tfidf = TfidfLinearScorer.fit(rows)
    maj_rep = evaluate_scorer(maj, rows[:12], shuffled=False)
    tfidf_rep = evaluate_scorer(tfidf, rows[:12], shuffled=False)
    assert 0.0 <= maj_rep.accuracy <= 1.0
    assert tfidf_rep.accuracy >= 0.5


def test_json_llm_tiny_runs() -> None:
    from jev.scoring.baselines import JsonLmScorer
    from jev.scoring.logprob import LogprobScorer
    from jev.scoring.tiny_lm import ByteTokenizer, seeded_tiny_lm

    rows = make_synthetic_choice(n=8, k=4, seed=3)
    inner = LogprobScorer(seeded_tiny_lm(0), ByteTokenizer(), model_id="tiny-causal-logprob")
    scorer = JsonLmScorer(inner)
    report = evaluate_scorer(scorer, rows[:4], shuffled=False)
    assert report.n == 4
    assert 0.0 <= report.accuracy <= 1.0
