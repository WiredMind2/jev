from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def cuda_ready() -> bool:
    return torch.cuda.is_available()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: needs NVIDIA GPU, CUDA PyTorch, and enough VRAM")
    config.addinivalue_line("markers", "hf: downloads Hugging Face models or datasets")
