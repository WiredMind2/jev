from pathlib import Path

from typer.testing import CliRunner

from jev.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_convert_train_calibrate_evaluate_synthetic(tmp_path: Path) -> None:
    out = tmp_path / "data"
    result = runner.invoke(app, ["data-convert", "synthetic", "--out", str(out), "--n", "128"])
    assert result.exit_code == 0, result.stdout
    train = out / "synthetic" / "jsonl" / "train.jsonl"
    val = out / "synthetic" / "jsonl" / "validation.jsonl"
    calib = out / "synthetic" / "jsonl" / "calibration.jsonl"
    test = out / "synthetic" / "jsonl" / "test.jsonl"
    assert train.exists() and test.exists() and calib.exists()
    ckpt = tmp_path / "head.pt"
    result = runner.invoke(
        app,
        [
            "train-head",
            str(train),
            "--out",
            str(ckpt),
            "--val-jsonl",
            str(val),
            "--epochs",
            "12",
            "--encoder",
            "hashing",
        ],
    )
    assert result.exit_code == 0, result.stdout
    result = runner.invoke(app, ["calibrate", str(calib), "--checkpoint", str(ckpt)])
    assert result.exit_code == 0, result.stdout
    assert "temperature" in result.stdout
    temp_json = tmp_path / "t.json"
    result = runner.invoke(
        app,
        [
            "calibrate",
            str(calib),
            "--backend",
            "tiny-logprob",
            "--out",
            str(temp_json),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert temp_json.exists()
    assert "calibration" in temp_json.read_text(encoding="utf-8")
    result = runner.invoke(
        app,
        ["evaluate", str(test), "--backend", "option-head", "--checkpoint", str(ckpt)],
    )
    assert result.exit_code == 0, result.stdout
    assert "accuracy" in result.stdout
    assert "risk_coverage" in result.stdout
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(test),
            "--backend",
            "tiny-logprob",
            "--temperature-json",
            str(temp_json),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = result.stdout.strip().splitlines()[-1]
    import json

    parsed = json.loads(payload)
    written = json.loads(temp_json.read_text(encoding="utf-8"))
    assert parsed["temperature"] == written["temperature"]
    result = runner.invoke(
        app,
        ["evaluate", str(test), "--backend", "fake"],
    )
    assert result.exit_code == 0, result.stdout
    progress_text = (result.stderr or "") + result.output
    assert "ETA" in progress_text
    assert "eval" in progress_text


def test_cli_train_head_prints_progress(tmp_path: Path) -> None:
    out = tmp_path / "data"
    result = runner.invoke(app, ["data-convert", "synthetic", "--out", str(out), "--n", "64"])
    assert result.exit_code == 0, result.stdout
    train = out / "synthetic" / "jsonl" / "train.jsonl"
    ckpt = tmp_path / "head.pt"
    result = runner.invoke(
        app,
        [
            "train-head",
            str(train),
            "--out",
            str(ckpt),
            "--encoder",
            "hashing",
            "--epochs",
            "1",
            "--max-steps",
            "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    progress_text = (result.stderr or "") + result.output
    assert "train-head" in progress_text
    assert "ETA" in progress_text


def test_cli_validate_and_score_example() -> None:
    result = runner.invoke(app, ["validate", "examples/systemone.request.json"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["score", "examples/systemone.request.json", "--backend", "fake"])
    assert result.exit_code == 0
    assert "department" in result.stdout
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Qwen/Qwen2.5-0.5B" in result.stdout
    assert "GTX 1650" in result.stdout or "GeForce GTX 1650" in result.stdout
