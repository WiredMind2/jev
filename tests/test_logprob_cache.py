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


def test_chunked_wide_menu_cached_matches_naive() -> None:
    model = seeded_tiny_lm(3)
    tok = ByteTokenizer()
    prefix = "Answer:"
    conts = [f" opt{i:02d}" + ("x" * (i % 5)) for i in range(40)]
    naive, cached = naive_vs_cached_scores(model, tok, prefix, conts)
    assert len(cached) == 40
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


def test_expand_past_hf_style_cache() -> None:
    from jev.scoring.logprob import _expand_past

    class FakeCache:
        def __init__(self) -> None:
            self.keys = torch.zeros(1, 2, 3, 4)
            self.values = torch.zeros(1, 2, 3, 4)

        def batch_repeat_interleave(self, repeats: int) -> None:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    cache = FakeCache()
    out = _expand_past(cache, 4)
    assert out is cache
    assert tuple(cache.keys.shape) == (4, 2, 3, 4)


def test_expand_past_legacy_cache() -> None:
    from jev.scoring.logprob import _expand_past

    class LegacyCache:
        def __init__(self, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
            self.layers = layers

        def to_legacy_cache(self):
            return tuple(self.layers)

        @classmethod
        def from_legacy_cache(cls, expanded):
            return cls(list(expanded))

    k = torch.zeros(1, 2, 3, 4)
    v = torch.zeros(1, 2, 3, 4)
    out = _expand_past(LegacyCache([(k, v)]), 5)
    assert out.layers[0][0].shape[0] == 5


def test_expand_past_tuple_layers() -> None:
    from jev.scoring.logprob import _expand_past

    past = [(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))]
    out = _expand_past(past, 6)
    assert out[0][0].shape[0] == 6
