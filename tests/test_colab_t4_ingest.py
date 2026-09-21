"""Ingest Colab Drive eval JSON into reports/colab-t4 without touching v0."""

from __future__ import annotations

import json
from pathlib import Path

from jev.colab_t4 import gpu_label, ingest_colab_t4, render_metrics_md

ROOT = Path(__file__).resolve().parents[1]


def _payload(**extra: float) -> dict:
    base = {
        "accuracy": 0.421,
        "nll": 1.504,
        "brier": 0.512,
        "ece": 0.101,
        "coverage_at_1pct": 0.25,
        "shuffled_accuracy": 0.12,
        "mae": 0.8,
    }
    base.update(extra)
    return base


def test_gpu_label_requires_t4() -> None:
    live = {"live_gpu": {"name": "Tesla T4", "total_mib": 15109}}
    assert gpu_label("", live) == "Colab T4"
    assert "not Colab T4" in gpu_label("NVIDIA GeForce RTX 4050", None)


def test_ingest_fills_qwen_rows_and_leaves_v0_untouched(tmp_path: Path) -> None:
    src = tmp_path / "jev-runs" / "reports"
    dest = tmp_path / "colab-t4"
    v0 = tmp_path / "v0" / "metrics.md"
    (src / "metrics").mkdir(parents=True)
    dest.mkdir()
    v0.parent.mkdir()
    v0.write_text("do not touch 1650 table\n", encoding="utf-8")
    template = (ROOT / "reports" / "colab-t4" / "metrics.md").read_text(encoding="utf-8")
    (dest / "metrics.md").write_text(template, encoding="utf-8")
    (src / "metrics" / "eval-banking77-hf-head.json").write_text(
        json.dumps(_payload()) + "\n", encoding="utf-8"
    )
    (src / "metrics" / "eval-banking77-hf-logprob.json").write_text(
        json.dumps(_payload(accuracy=0.333, shuffled_accuracy=0.05)) + "\n",
        encoding="utf-8",
    )
    (src / "colab-live-hardware.json").write_text(
        json.dumps({"live_gpu": {"name": "Tesla T4", "total_mib": 15109}}) + "\n",
        encoding="utf-8",
    )
    (src / "hardware.md").write_text(
        "# Colab live GPU\n\n- name: Tesla T4\n", encoding="utf-8"
    )
    (src / "model-cards").mkdir()
    (src / "model-cards" / "qwen25-3b-probe.md").write_text(
        "# probe\n\n**Outcome:** fit (2 steps completed)\n", encoding="utf-8"
    )

    result = ingest_colab_t4(src, dest, v0_metrics=v0)
    assert result["gpu"] == "Colab T4"
    assert result["filled"]["banking77"] == ["hf-head", "hf-logprob"]
    md = (dest / "metrics.md").read_text(encoding="utf-8")
    assert "| Frozen Qwen2.5-0.5B option head | 0.421 | 1.504 | 0.512 | 0.101 | 0.25 | 0.120 |" in md
    assert "| Zero-shot Qwen2.5-0.5B logprob | 0.333 |" in md
    assert "fit (2 steps completed)" in md
    assert v0.read_text(encoding="utf-8") == "do not touch 1650 table\n"
    assert (dest / "metrics" / "eval-banking77-hf-head.json").is_file()
    assert (dest / "hardware.md").is_file()


def test_ingest_colab_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from jev.cli import app

    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    (dest / "metrics.md").write_text(
        (ROOT / "reports" / "colab-t4" / "metrics.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    v0 = tmp_path / "v0.md"
    v0.write_text("pin\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["ingest-colab", str(src), "--dest", str(dest), "--v0-metrics", str(v0)],
    )
    assert result.exit_code == 0, result.stdout
    assert "unknown GPU" in result.stdout or "gpu" in result.stdout.lower()


def test_ingest_refuses_v0_dest(tmp_path: Path) -> None:
    dest = tmp_path / "reports" / "v0" / "sneak"
    dest.mkdir(parents=True)
    try:
        ingest_colab_t4(tmp_path, dest)
    except ValueError as exc:
        assert "reports/v0" in str(exc)
    else:
        raise AssertionError("expected refusal")


def test_render_does_not_claim_t4_for_empty_gpu() -> None:
    template = (ROOT / "reports" / "colab-t4" / "metrics.md").read_text(encoding="utf-8")
    text = render_metrics_md(template, rows={}, gpu="unknown GPU")
    assert "Qwen rows below are from a live Colab T4 session" not in text
    assert "| Frozen Qwen2.5-0.5B option head |  |" in text or "| Frozen Qwen2.5-0.5B option head |" in text
