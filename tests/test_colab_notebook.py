"""Guard the Colab path so v0 cannot drop the hosted-GPU recipe."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "jev_colab_hosted_gpu.ipynb"
COLAB_DOC = ROOT / "docs" / "11-colab.md"
README = ROOT / "README.md"


def _notebook_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload["cells"]:
        src = cell.get("source", [])
        if isinstance(src, list):
            chunks.append("".join(src))
        else:
            chunks.append(str(src))
    return "\n".join(chunks)


def test_colab_docs_and_notebook_exist() -> None:
    assert COLAB_DOC.is_file()
    assert NOTEBOOK.is_file()
    assert "Hosted GPUs" in COLAB_DOC.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    assert "docs/11-colab.md" in readme
    assert "notebooks/jev_colab_hosted_gpu.ipynb" in readme


def test_notebook_invokes_cli_not_a_second_trainer() -> None:
    text = _notebook_source()
    for needle in (
        "python -m jev",
        "data-convert",
        "train-head",
        "calibrate",
        "evaluate",
        "--out",
        "--no-deps",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-3B",
        "HF_TOKEN",
        "google.colab",
        "nvidia-smi",
        "hardware",
        "--encoder",
        "--resume",
        "--temperature-json",
        "--limit",
        "hardware.md",
        "banking77",
        "sst5",
        "boolq",
        "wikispeedia",
        "clinc150",
        "hf-logprob",
        "majority",
        "tfidf-linear",
        "laptop GPU",
        "git HEAD",
        "FETCH_HEAD",
    ):
        assert needle in text, needle
    assert "def train_option_head" not in text
    assert "hf_xxx" not in text.lower()
    assert "hf_token =" not in text.lower()
    # Serving is documented as forbidden, not offered as a cell to run.
    assert "jev serve" not in text or "not" in text.lower()
    # Run All on Colab+CUDA must not skip later public tasks via env defaults of "0".
    assert 'get("RUN_SST5", "0")' not in text
    assert 'get("RUN_BOOLQ", "0")' not in text
    assert 'get("RUN_WIKISPEEDIA", "0")' not in text
    assert 'get("RUN_3B_PROBE", "0")' not in text
    assert "MAX_STEPS" in text
    assert '"wikispeedia": "400"' in text or '"wikispeedia":"400"' in text.replace(" ", "")
