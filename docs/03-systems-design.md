# Systems design

Everything in this file is **inference** unless marked otherwise. It is
grounded in TypeSafe's public claims and in behavior that open projects
already reproduce. It is not TypeSafe's disclosed implementation.

## Why a normal LLM call is the wrong cost model

A request has one state `x` and `m` questions `q_i = (instructions, C_i)`.

A naive design renders `m` prompts `p_i = render(x, q_i)` and calls a
generator `m` times. That re-encodes `x` every time. When `x` is long and
each question is short, prefill dominates.

A Jev-like implementation should:

1. Encode state once.
2. Cache token states or the decoder KV cache.
3. Attach each question / option block to that prefix.
4. Score all candidates in one batched forward pass, or a small number of
   them.
5. Return structured tensors, not sampled completion text.

```text
H_x, KV_x = fθ_prefix(x)

s_{i,j} = gθ(KV_x, q_i, c_{i,j})

p_{i,j} = exp(s_{i,j} / τ_i) / Σ_k exp(s_{i,k} / τ_i)
```

The main speed benefit does **not** require a new foundation-model
architecture. Cache reuse plus batched candidate scoring already removes
redundant prefill and avoids generated output.

**Open project.** `daseinlabs/open-jev` implements this with Gemma 3 4B
via MLX. On their M5 Pro measurement, a cached 202-token context with
eight options is 0.17 s median versus 0.68 s when the context is
re-encoded per candidate.

TypeSafe's own docs are consistent with this cost model: state is ingested
once; extra questions cost extra tokens, not extra prefills of `x`.

## Route A — continuation log-probability scorer

For a causal decoder, score candidate `c_j` as its conditional log
likelihood after a prompt `p`:

```text
s_j = Σ_t log Pθ(c_{j,t} | p, c_{j,<t})
P(c_j | p, C) = softmax_j(s_j)
```

No answer tokens are sampled.

Raw summed log likelihood favors short candidates. Three normalizations:

| Method | Score | Strength | Weakness |
|---|---|---|---|
| Sum | `Σ log p(c_t \| …)` | Proper sequence likelihood | Short-answer bias |
| Mean | `(1/L) Σ log p(c_t \| …)` | Reduces length bias | Not a sequence probability |
| PMI | `log P(c\|p) − log P(c)` | Discounts generic options | Extra unconditional pass |

**Open project.** `open-jev` exposes all three. Its authors note that
softmaxed LM likelihoods are **not** calibrated correctness
probabilities. On TypeSafe's published quickstart ticket, Gemma 3 4B
zero-shot agreed on the department (`technical`) but returned
confidence 1.00, scored frustration at the top bin, and flipped urgency
(`noul` 0.005 vs Jev 0.999).

Use short opaque labels (`A`, `B`, `C`) only if definitions live fully in
the prefix. Otherwise score meaningful names. Compare both: lexical priors
and tokenization often dominate.

## Route B — dedicated compatibility head

A more faithful research design does not score option text with the LM
head:

1. Encode state into context token embeddings `H_x ∈ R^{T×d}`.
2. Encode every option into a vector `u_j ∈ R^d`.
3. Let each option attend to the state.
4. Map the option-specific summary to one scalar logit.
5. Masked softmax across the current option set.

```text
q_j = W_q u_j

α_{j,t} = softmax_t( q_jᵀ W_k H_{x,t} / √d_k )

r_j = Σ_t α_{j,t} W_v H_{x,t}

s_j = (W_o u_j)ᵀ r_j / √d  +  b_j

p_j = softmax_j(s_j)
```

This supports variable-size menus and a single forward decision pass.

**Open project.** This is the independently released `jevlike` design. It
trains from JSONL `{context, options, label}` with either a scratch byte
encoder or a frozen Hugging Face encoder.

**Fact that supports independent option scoring.** TypeSafe's Score page
says every level is evaluated separately and that the model does not see
the level number or its neighbours. That is exactly the "each candidate is
a query against shared state" pattern, not an autoregressive ranking over
a numbered rubric.

## A plausible general model family

```text
Structured state / text
        │
        ▼
Shared encoder or decoder-prefix transformer
        │
        ├── token states Hx
        └── pooled global state zx
        │
Question encoder ───► question vector qi
Option encoder   ───► option vectors ui,1 ... ui,ni
        │
        ▼
Question-conditioned, option-query cross-attention
        │
        ▼
One scalar logit per option
        │
        ▼
Masked softmax over valid options
        │
        ├── Choice probabilities
        ├── Boolean probability (Noul = 2-way Choice)
        └── Ordinal score distribution
```

For a prototype, freeze an open encoder (Qwen, Gemma, ModernBERT, e5,
DeBERTa-family). For high-throughput production, an encoder /
cross-attention architecture is a better fit than a causal decoder:
there is no causal-token constraint, and every decision slot can be
scored in parallel.

Unify the three primitives as finite-option distributions:

- Choice: arbitrary named labels
- Noul: `{false, true}`
- Score: ordered rubric levels, independently encoded

That shares data loaders, batching, metrics, calibration, and serving.

## High cardinality

**Fact.** Direct Choice stops at 255. Above that, TypeSafe independently
scores candidates and then makes an explicit choice.

**Inference.** Stage 1 is a cheap pointwise scorer (or the same head
without a menu-wide softmax). Stage 2 is a Choice over a shortlist.
Log the shortlist; leakage and prior bias show up here first.

## Multi-question amortization

For `M` questions with `N_i` options:

```text
Cost_naive  ≈ Σ_i Encode(x, q_i, C_i)
Cost_shared ≈ Encode(x) + Σ_i Score(q_i, C_i | H_x)
```

This amortization—not a mysterious semantic capability—is where the
clearest speedup should appear when state is long and many decisions
share it.

After a single-question model works:

1. Serialize and encode state once.
2. Encode each question/options bundle.
3. Batch all option-query heads.
4. Use attention masks for variable `N_i`.
5. Return every typed answer in one call.
6. Profile tokenization, prefill, option encoding, the scoring head,
   calibration, and JSON/network overhead separately.

## Serving sketch

```text
FastAPI / TypeScript client
        │
POST /v1/systemone
        │
Schema validation and canonicalization
        │
Shared state encoder cache
        │
Decision-head batch inference
        │
Temperature / Dirichlet calibration
        │
Policy layer: threshold, defer, audit event
        │
Typed JSON response
```

Recommended stack for an open engine:

- Training: PyTorch, Transformers, Accelerate
- Data: JSONL + DuckDB/Parquet
- Serving: FastAPI, Pydantic, Uvicorn; later vLLM/SGLang
- Apple Silicon experiments: MLX, following `open-jev`
- Metrics: torchmetrics / sklearn plus custom reliability and
  risk–coverage curves
- Tracing: log state hash, schema version, candidate-set hash, raw
  logits, calibrated probabilities, decision, downstream outcome

Expose the API as a tool only for **read-only decisions**. Keep execution
tools behind deterministic authorization and explicit confirmation.
