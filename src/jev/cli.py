"""Typer CLI for the research package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import uvicorn

from jev import __version__
from jev.calibration import collect_logit_gold, fit_temperature, load_temperature_json
from jev.canonical import canonical_dumps
from jev.colab_t4 import ingest_colab_t4
from jev.data.convert import freeze_and_write
from jev.data.manifest import criteria_path
from jev.data.synthetic import make_synthetic_choice, split_synthetic
from jev.evaluation import evaluate_scorer, report_as_dict, stratified_sample
from jev.hardware import load_hardware_pin
from jev.pipeline import respond
from jev.progress import StageBars, TaskProgress
from jev.schema import parse_request
from jev.scoring.factory import build_scorer
from jev.scoring.option_head import (
    OptionHeadScorer,
    TrainConfig,
    accuracy_on_examples,
    load_checkpoint,
    load_jsonl_examples,
    train_option_head,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _root() -> None:
    """Open Jev-like research CLI. Not TypeSafe Jev."""


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def hardware() -> None:
    """Print the pinned GPU and Qwen2.5-0.5B reduction."""
    typer.echo(canonical_dumps(load_hardware_pin().as_dict()))


@app.command()
def validate(
    path: Path = typer.Argument(..., help="JSON request or training example"),
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "questions" in payload:
        req = parse_request(payload)
        typer.echo(f"request ok: {len(req.questions)} questions")
        return
    from jev.schema import parse_training_example

    ex = parse_training_example(payload)
    typer.echo(f"example ok: {ex.id} type={ex.type}")


@app.command("data-convert")
def data_convert(
    dataset: str = typer.Argument(..., help="synthetic|banking77|sst5|boolq|clinc150|wikispeedia"),
    out: Path = typer.Option(Path("data"), help="Output parent directory"),
    n: int = typer.Option(256, help="Synthetic row count"),
    fixture: Path | None = typer.Option(None, help="Optional local JSON/JSONL fixture"),
) -> None:
    dest = out / dataset
    dest.mkdir(parents=True, exist_ok=True)
    if dataset == "synthetic":
        rows = make_synthetic_choice(n=n)
        splits = split_synthetic(rows)
        manifest = freeze_and_write(
            dataset="synthetic",
            primitive="choice",
            criteria_file=criteria_path("synthetic"),
            converter="jev.data.synthetic",
            license_name="generated-in-repo",
            source="jev.data.synthetic.make_synthetic_choice",
            split_rule="group_id hashed; ~10% groups held out as test",
            examples_by_split=splits,
            out_dir=dest,
        )
        typer.echo(str(manifest))
        return
    from jev.data.registry import convert_named

    path = convert_named(dataset, dest, fixture=fixture)
    typer.echo(str(path))


@app.command("score")
def score_cmd(
    request_path: Path = typer.Argument(..., exists=True),
    backend: str = typer.Option("fake", help="fake|tiny-logprob|hf-logprob|option-head"),
    checkpoint: Path | None = typer.Option(None, help="Required for option-head"),
    temperature: float = typer.Option(1.0),
) -> None:
    request = parse_request(json.loads(request_path.read_text(encoding="utf-8")))
    scorer = build_scorer(backend, checkpoint=checkpoint)
    response = respond(scorer, request, temperature=temperature)
    typer.echo(canonical_dumps(response.model_dump(mode="json")))


@app.command("train-head")
def train_head_cmd(
    train_jsonl: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("runs/synthetic-head.pt")),
    val_jsonl: Path | None = typer.Option(None, help="Validation JSONL for early stopping only"),
    epochs: int = typer.Option(12),
    encoder: str = typer.Option("hashing", help="hashing|hf"),
    model_id: str = typer.Option("Qwen/Qwen2.5-0.5B"),
    device: str | None = typer.Option(None),
    batch_size: int = typer.Option(16),
    max_steps: int | None = typer.Option(None),
    resume: Path | None = typer.Option(None, help="Checkpoint to continue; encoder stays frozen"),
    save_every: int = typer.Option(50, help="Write --out every N steps; 0 disables mid-run saves"),
) -> None:
    import torch

    examples = load_jsonl_examples(train_jsonl)
    val = load_jsonl_examples(val_jsonl) if val_jsonl else None
    if device is None:
        torch_device = torch.device("cuda" if encoder == "hf" and torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    cfg = TrainConfig(
        epochs=epochs,
        encoder_kind=encoder,
        hf_model_id=model_id,
        batch_size=batch_size,
        max_steps=max_steps,
        save_every=save_every or None,
        checkpoint_path=out,
        resume_path=resume,
    )
    train_bar = TaskProgress("train-head", 1, stream=sys.stderr, unit="step")

    def _train_progress(step: int, total: int) -> None:
        train_bar.set(step, total=total)

    trained_encoder, head, history = train_option_head(
        examples, cfg, device=torch_device, val_examples=val, progress=_train_progress
    )
    train_bar.close()
    if encoder == "hf":
        typer.echo(
            f"skip train-acc/train-shuffled n={len(examples)}; "
            "frozen test evaluate --limit is the comparison",
            err=True,
        )
        acc = None
        shuf = None
    else:
        acc_bar = TaskProgress("train-acc", len(examples), stream=sys.stderr, unit="ex")
        acc = accuracy_on_examples(
            trained_encoder,
            head,
            examples,
            device=torch_device,
            progress=lambda i, n: acc_bar.set(i, total=n),
        )
        acc_bar.close()
        shuf_bar = TaskProgress("train-shuffled", len(examples), stream=sys.stderr, unit="ex")
        shuf = accuracy_on_examples(
            trained_encoder,
            head,
            examples,
            device=torch_device,
            shuffle_state=True,
            seed=1,
            progress=lambda i, n: shuf_bar.set(i, total=n),
        )
        shuf_bar.close()
    typer.echo(
        canonical_dumps(
            {
                "checkpoint": str(out),
                "train_acc": acc,
                "shuffled_acc": shuf,
                "encoder_kind": encoder,
                "model_id": model_id if encoder == "hf" else "option-attention-hashing",
                **history,
            }
        )
    )


@app.command("calibrate")
def calibrate_cmd(
    jsonl: Path = typer.Argument(..., exists=True, help="Calibration split only"),
    checkpoint: Path | None = typer.Option(None, help="Required for option-head"),
    backend: str = typer.Option("option-head", help="option-head|hf-logprob|tiny-logprob|fake"),
    out: Path | None = typer.Option(None, help="Write temperature JSON"),
    limit: int = typer.Option(0, help="Stratified sample of the calibration split; 0 keeps all"),
    seed: int = typer.Option(0),
) -> None:
    examples = load_jsonl_examples(jsonl)
    if limit:
        examples = stratified_sample(examples, limit, seed=seed)
    name = backend.strip().lower()
    if name in {"option-head", "head", "hashing-head"}:
        if checkpoint is None:
            raise typer.BadParameter("option-head calibrate requires --checkpoint")
        encoder, head, meta = load_checkpoint(checkpoint)
        scorer = OptionHeadScorer(encoder, head, model_id=str(meta.get("model_id") or "option-attention"))
    else:
        scorer = build_scorer(backend, checkpoint=checkpoint)
    bars = StageBars(stream=sys.stderr, unit="ex")
    try:
        pairs = collect_logit_gold(scorer, examples, progress=bars)
    finally:
        bars.close()
    cal = fit_temperature(pairs)
    payload = {
        "temperature": cal.temperature,
        "n": len(pairs),
        "split": "calibration",
        "backend": name,
        "model_id": scorer.model_id,
    }
    text = canonical_dumps(payload)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)


@app.command("evaluate")
def evaluate_cmd(
    jsonl: Path = typer.Argument(..., exists=True),
    backend: str = typer.Option(
        "option-head", help="fake|tiny-logprob|hf-logprob|option-head|majority|tfidf-linear|json-llm"
    ),
    checkpoint: Path | None = typer.Option(None),
    train_jsonl: Path | None = typer.Option(None, help="Train JSONL for majority/tfidf-linear"),
    temperature: float = typer.Option(1.0),
    temperature_json: Path | None = typer.Option(
        None, help="Calibrate JSON with {temperature, split}; overrides --temperature"
    ),
    limit: int = typer.Option(0, help="Stratified sample size; 0 keeps all rows"),
    seed: int = typer.Option(0),
    shuffle: bool = typer.Option(True, help="Also score shuffled-context control"),
    out: Path | None = typer.Option(None, help="Write metrics JSON"),
) -> None:
    examples = load_jsonl_examples(jsonl)
    if limit:
        examples = stratified_sample(examples, limit, seed=seed)
    if temperature_json is not None:
        temperature = load_temperature_json(temperature_json)
    scorer = build_scorer(backend, checkpoint=checkpoint, train_jsonl=train_jsonl)
    bars = StageBars(stream=sys.stderr, unit="ex")
    try:
        report = evaluate_scorer(
            scorer,
            examples,
            temperature=temperature,
            shuffled=shuffle,
            progress=bars,
        )
    finally:
        bars.close()
    payload = {
        "backend": backend,
        "temperature": temperature,
        "model_id": scorer.model_id,
        "limit": limit or None,
        **report_as_dict(report),
    }
    text = canonical_dumps(payload)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)


@app.command("ingest-colab")
def ingest_colab_cmd(
    src: Path = typer.Argument(..., help="Drive jev-runs dump (reports/ and metrics/)"),
    dest: Path = typer.Option(Path("reports/colab-t4"), help="Git T4 report dir; never reports/v0"),
    v0_metrics: Path = typer.Option(Path("reports/v0/metrics.md"), help="Guard file that must not change"),
) -> None:
    """Copy Colab eval JSON into reports/colab-t4. Does not mix into the 1650 table."""
    typer.echo(canonical_dumps(ingest_colab_t4(src, dest, v0_metrics=v0_metrics)))


@app.command("serve")
def serve_cmd(
    host: str = "127.0.0.1",
    port: int = 8000,
    backend: str = "fake",
    checkpoint: Path | None = typer.Option(None),
) -> None:
    from jev.api import create_app

    scorer = build_scorer(backend, checkpoint=checkpoint)
    uvicorn.run(create_app(scorer), host=host, port=port)


if __name__ == "__main__":
    app()
