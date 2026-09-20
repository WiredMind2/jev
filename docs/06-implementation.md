# Implementation plan

Technically serious path. Do not begin by training a transformer from
scratch.

## Stage 0 — specify the decision contract

Implement a schema that makes a bad output *shape* impossible. See
[API contract](02-api-contract.md) and [`../schemas/`](../schemas/).

Hard constraints:

- At least two choices; cap at 255.
- Score levels in `[2, 10]`.
- Canonicalize object serialization before caching.
- Opaque stable option IDs; long text lives in descriptions.
- Always support `unknown` / `insufficient_information` / `defer` where
  the question can fail closed.
- Model output never sets authorization boundaries.

## Stage 1 — zero-shot serving baseline

Local causal model, candidate continuation scoring, prefix cache.

| Hardware / goal | Starting class |
|---|---|
| Apple Silicon, rapid iteration | Gemma 3 4B or Qwen 3–4B via MLX |
| Consumer NVIDIA GPU | Qwen 2.5/3 3B–8B via Transformers, vLLM, or SGLang |
| CPU / low-memory proof of concept | Small quantized Qwen/Gemma or an embedding baseline |
| Production experimental server | vLLM or SGLang with batched prefill |

```python
prefix = render_state_and_question(state, question)
cache = model.prefill(prefix)
candidate_scores = model.score_continuations_batched(
    cache=cache,
    continuations=[encode(label) for label in labels],
)
probabilities = softmax(normalize(candidate_scores))
```

Test at least: sum / mean / PMI norms; option-order permutations; several
prompt templates; option-label randomization; JSON field-order
randomization; adversarial state.

Concrete reference: `daseinlabs/open-jev`, including its check that cached
batched scoring matches naive re-encoding.

Do not invent a private ticket corpus in week 1. Convert **BANKING77**
(Choice) using the recipe in [Datasets](10-datasets.md). Keep a
`jevlike` synthetic run as a unit test of the trainer. Add SST-5 and
BoolQ before mixing domains.

## Stage 2 — evaluation harness

See [Evaluation](05-evaluation.md). Ship the harness before the learned
head. Include a JSON-LLM baseline and an embedding/linear baseline on day
one so later gains are comparable.

## Stage 3 — train a decision head

Freeze a pretrained encoder. Train only an option-conditioned scoring
head.

```text
State text → frozen encoder → H_state ∈ R[T, d]
Question + option text → frozen encoder → u_option ∈ R[d]
u_option → low-rank query projection
query × H_state → cross-attention over state tokens
attended state + option representation → MLP scalar logit
masked softmax over the option set
```

Loss: listwise cross-entropy on the gold option index.

Fork/adapt `jevlike`. Compare the head to zero-shot causal scoring on the
exact same held-out test set.

## Stage 4 — calibration and deferral

Fit temperature scaling, then isotonic / Dirichlet. Add explicit `defer`.
Report selective error at fixed coverage and coverage at fixed error.
Add injection-bearing inputs and option-order tests to CI.

## Stage 5 — shared-state, multi-question optimization

Batch 5–50 questions against the same state. Profile prefill versus
scoring. Implement option masks and compact numeric output. Quantize or
compile only after correctness and calibration are stable.

## Six-week schedule

### Week 1 — interface and zero-shot baseline

- One finite-option abstraction for Choice, Score, Noul.
- Local logprob scorer with batching and prefix caching.
- Convert BANKING77 + synthetic menus to the training schema; freeze
  intent descriptions in a hashed file.
- Run zero-shot logprob on a 300-row stratified BANKING77 slice (same
  protocol as `jev-eval`) so later heads have a number to beat.

### Week 2 — evaluation harness

- Group-aware splits.
- Accuracy, NLL, Brier, ECE, risk–coverage, order sensitivity,
  shuffled-context.
- JSON-LLM and embedding/linear baselines.

### Week 3 — learned option scorer

- Frozen encoder + low-rank option-query head.
- Train on BANKING77; eval vs zero-shot on the same test set.
- If that works, start Wikispeedia next-click (variable `N`,
  target-disjoint split). Synthetic 98% is not evidence.

### Week 4 — calibration and deferral

- Temperature, then isotonic/Dirichlet.
- `defer` / `insufficient_information`.
- Policy tests and injection-bearing inputs.

### Week 5 — multi-question amortization

- 5–50 questions, one state.
- Masks, dynamic batching, profiling.

### Week 6 — freeze and report

- Freeze datasets and splits.
- Robustness suite.
- Model card, data card, calibration plots, failure analysis.
- Document non-goals: no factual-correctness guarantee, no replacement
  for authorization, no claim of reproducing Jev/RLCD.

## Suggested first code layout (when implementation starts)

```text
src/jev/
  schema.py          # pydantic models matching schemas/
  render.py          # state + question → prefix / option texts
  scorer_logprob.py  # Stage 1
  scorer_head.py     # Stage 3
  calibrate.py
  serve.py           # FastAPI /v1/systemone
  data/
    convert.py       # BANKING77 / SST-5 / BoolQ / Wikispeedia → JSONL
  eval/
    metrics.py
    splits.py
    ablations.py
```

This repository currently contains documentation, schemas, a Python package
(`src/jev`), CPU/CUDA tests, and a v0 report under `reports/v0/`.
