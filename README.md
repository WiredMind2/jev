# Open notes on Jev-like decision models

This repository collects public information about TypeSafe's **Jev**, the first
"System One" model, and turns it into a working plan for an **open, equivalent
decision engine**. Equivalent here means the same *task interface* and the same
*measurable properties*—typed finite-option distributions, shared-state
batching, and calibrated confidence—not a clone of TypeSafe's unpublished
weights, architecture, or RLCD recipe.

Jev is best understood as a **closed, non-autoregressive decision system**, not
as a chat LLM with a JSON schema.

```text
state + {typed question_i}  -->  {typed decision_i, distribution_i, confidence_i}
```

## Status of this project

| Claim | Status |
|---|---|
| Document the public API contract | Schemas and `docs/02-api-contract.md` |
| Choose public train/eval datasets | Catalog in `docs/10-datasets.md`; converters in `src/jev/data/` |
| Reconstruct a serving baseline (cached option scoring) | Implemented: FastAPI `POST /v1/systemone`, Qwen2.5-0.5B cached logprob |
| Train a variable-menu decision head | Implemented: frozen encoder + option-attention head; hashing trainer proof + Qwen frozen head |
| v0 report on frozen synthetic test | [reports/v0](reports/v0/metrics.md) (CLI `train-head --encoder hf`: zero-shot 0.75 vs underfit frozen head 0.083; hashing 1.00 is not the LM comparison) |
| Reproduce TypeSafe's private architecture | **Not possible from public information** |
| Reproduce RLCD as TypeSafe trains it | **Not possible from public information** |

Hardware pin: **Qwen/Qwen2.5-0.5B fp16** for both zero-shot and the frozen head. Qwen2.5-3B does not fit on the 4 GiB GTX 1650 used here (`configs/hardware.yaml`). Later frozen-head runs that need more VRAM (BANKING77, Wikispeedia, a 3B encoder) can use a free Colab T4; that is a separate pin, not a rewrite of the v0 table. See [Hosted GPUs](docs/11-colab.md) and the runnable notebook [notebooks/jev_colab_hosted_gpu.ipynb](notebooks/jev_colab_hosted_gpu.ipynb) ([Open in Colab](https://colab.research.google.com/github/WiredMind2/jev/blob/main/notebooks/jev_colab_hosted_gpu.ipynb)).

### Research package

Python ≥3.11, install from this repo:

```text
python -m pip install -e ".[dev]"
jev --help
```

CLI: `validate`, `data-convert`, `score`, `train-head`, `calibrate`, `evaluate`, `serve`, `hardware`.

CPU tests: `pytest -m "not cuda and not hf"`. CUDA gates: `pytest -m cuda`. Do not use system Python 3.9; use `.venv`. Hosted-GPU training (Colab T4, not serving): [docs/11-colab.md](docs/11-colab.md), [notebook](notebooks/jev_colab_hosted_gpu.ipynb), v0 gate in [reports/v0/colab.md](reports/v0/colab.md), T4 report tree [reports/colab-t4](reports/colab-t4/README.md). Local CPU proof of that CLI sequence: `scripts/colab_cpu_smoke.sh`.

This is independent research. It is **not** TypeSafe Jev and **not** RLCD.

TypeSafe has not published parameter counts, topology, tokenizer, pretraining
data, the RLCD reward, or open weights. Treat every architectural diagram in
this repo as a **research hypothesis** unless a source is marked as a public
fact.

## How to read these notes

Each technical claim is tagged:

- **Fact** — stated by TypeSafe in docs, the launch post, or the HTTP API.
- **Measurement** — reported by TypeSafe or a third party on a named eval; not
  independently reproduced here.
- **Inference** — a design that would explain the public behavior. Useful for
  building an open model; not TypeSafe's disclosed implementation.
- **Open project** — observed in community code (`open-jev`, `jevlike`, etc.).

Start here:

1. [Public record](docs/01-public-record.md) — confirmed facts and unknowns
2. [API contract](docs/02-api-contract.md) — `POST /v1/systemone` reconstructed
3. [Systems design](docs/03-systems-design.md) — shared-state scoring hypotheses
4. [Training](docs/04-training.md) — supervised path and open RLCD analogues
5. [Evaluation](docs/05-evaluation.md) — metrics, ablations, report table
6. [Implementation plan](docs/06-implementation.md) — four stages to a working engine
7. [Open equivalents](docs/07-open-equivalents.md) — projects to study and fork
8. [Limitations](docs/08-limitations.md) — jaggedness that any replica must handle
9. [Bibliography](docs/09-bibliography.md) — sources with retrieval dates
10. [Datasets](docs/10-datasets.md) — public corpora mapped to Choice / Score / Noul
11. [Hosted GPUs](docs/11-colab.md) — Colab for training/eval that does not fit the 1650; not serving. Notebook: [notebooks/jev_colab_hosted_gpu.ipynb](notebooks/jev_colab_hosted_gpu.ipynb)

JSON Schemas for the public request/response and a training-row format live in
[`schemas/`](schemas/).

## Bottom line

You can replicate the useful **Jev pattern** now:

```text
state
  -> finite typed questions
  -> batched candidate logits
  -> calibrated distributions
  -> deterministic policy
```

The strongest first implementation is:

1. Cached, batched zero-shot option likelihoods (the `open-jev` approach).
2. A learned variable-option decision head (the `jevlike` approach).
3. Train on the v0 mix in [Datasets](docs/10-datasets.md): synthetic menus,
   BANKING77, SST-5, BoolQ, then Wikispeedia next-click.
4. Calibration, abstention, leakage controls, and a policy layer as the actual
   research contribution.
5. Measurement against a JSON-generating LLM *and* a task-specific encoder
   classifier.

The opportunity is not guessing Jev's parameter count. It is building a
transparent decision system with evidence for when its probabilities should be
trusted.

## Non-goals

- No claim of affiliation with TypeSafe AI.
- No claim that an open model here *is* Jev.
- No guarantee of factual correctness from typed outputs. Type-safety means
  "cannot emit an undeclared enum variant," not "cannot pick the wrong one."
- A decision model should recommend and route. It should not be the sole
  authority for irreversible writes, payments, or authorization.

## License

Documentation and schemas in this repository are MIT-licensed. TypeSafe's
product, docs, and trademarks remain theirs. Linked open-source projects keep
their own licenses.
