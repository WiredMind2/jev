import torch

from jev.scoring.logprob import naive_vs_cached_scores
from jev.scoring.tiny_lm import ByteTokenizer, seeded_tiny_lm


def test_tiny_lm_cached_matches_naive() -> None:
    model = seeded_tiny_lm(0)
    tok = ByteTokenizer()
    prefix = "State: pick azure.\nAnswer:"
    conts = [" azure", " crimson", " ochre"]
    naive, cached = naive_vs_cached_scores(model, tok, prefix, conts)
    assert len(naive) == 3
    for a, b in zip(naive, cached, strict=True):
        assert abs(a - b) < 1e-4


def test_option_order_does_not_change_individual_scores() -> None:
    model = seeded_tiny_lm(1)
    tok = ByteTokenizer()
    prefix = "Answer:"
    conts = [" aa", " bb", " cc"]
    naive, cached = naive_vs_cached_scores(model, tok, prefix, conts)
    naive2, cached2 = naive_vs_cached_scores(model, tok, prefix, list(reversed(conts)))
    assert naive[0] == naive2[-1]
    assert cached[0] == cached2[-1]
    for a, b in zip(naive, cached, strict=True):
        assert abs(a - b) < 1e-4


def test_reductions_are_finite() -> None:
    from jev.scoring.logprob import LogprobScorer

    model = seeded_tiny_lm(2)
    tok = ByteTokenizer()
    for red in ("sum", "mean", "pmi"):
        scorer = LogprobScorer(model, tok, reduction=red, use_cache=True)
        scores = scorer.score_strings("Answer:", [" yes", " no"])
        assert all(torch.isfinite(torch.tensor(s)) for s in scores)
