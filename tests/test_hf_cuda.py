import pytest

from jev.hardware import load_hardware_pin, release_cuda
from jev.scoring.hf_lm import LAST_ERROR, try_load_hf_logprob_scorer
from jev.scoring.logprob import LogprobScorer

pytestmark = [pytest.mark.cuda, pytest.mark.hf]


def test_hf_zero_shot_naive_matches_cached() -> None:
    pin = load_hardware_pin()
    assert pin.zero_shot_model == "Qwen/Qwen2.5-0.5B"
    loaded = try_load_hf_logprob_scorer(pin.zero_shot_model, dtype=pin.dtype, device="cuda")
    if loaded is None:
        pytest.skip(f"HF logprob scorer unavailable: {LAST_ERROR}")
    scorer, meta = loaded
    try:
        prefix = "State: the matching bucket is azure.\nAnswer:"
        conts = [" azure", " crimson"]
        cached = LogprobScorer(
            scorer.model, scorer.tokenizer, use_cache=True, reduction="sum", model_id=scorer.model_id
        )
        naive = LogprobScorer(
            scorer.model, scorer.tokenizer, use_cache=False, reduction="sum", model_id=scorer.model_id
        )
        a = naive.score_strings(prefix, conts)
        b = cached.score_strings(prefix, conts)
        for x, y in zip(a, b, strict=True):
            assert abs(x - y) < 1e-3, (a, b, meta)
    finally:
        del scorer
        release_cuda()
