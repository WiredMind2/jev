import pytest
import torch

from jev.data.synthetic import make_synthetic_choice
from jev.scoring.logprob import naive_vs_cached_scores
from jev.scoring.option_head import TrainConfig, accuracy_on_examples, train_option_head
from jev.scoring.tiny_lm import ByteTokenizer, seeded_tiny_lm

pytestmark = pytest.mark.cuda


def test_cuda_is_available() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).total_memory >= 1 << 20


def test_tiny_lm_naive_matches_cached_on_cuda() -> None:
    model = seeded_tiny_lm(0).cuda()
    tok = ByteTokenizer()
    naive, cached = naive_vs_cached_scores(model, tok, "Answer:", [" yes", " no", " maybe"])
    for a, b in zip(naive, cached, strict=True):
        assert abs(a - b) < 1e-4


def test_hashing_head_trains_on_cuda() -> None:
    rows = make_synthetic_choice(n=48, seed=2)
    device = torch.device("cuda")
    encoder, head, _ = train_option_head(
        rows, TrainConfig(epochs=8, batch_size=8, seed=2, lr=3e-3), device=device
    )
    acc = accuracy_on_examples(encoder, head, rows, device=device)
    assert acc >= 0.8
