# v0 methods

Open reconstruction of the public System One *interface* (Choice / Score / Noul).
Not TypeSafe Jev, not RLCD, not a claimed clone of unpublished weights.

Retrieved hardware and models: 20 September 2026.

## Decision contract

- Request/response: `POST /v1/systemone` as reconstructed in `docs/02-api-contract.md`.
- Training rows: `schemas/training-example.schema.json`.
- Choice: 2–255 named options; Score: 2–10 ordered levels; Noul: P(true) with no separate confidence field.
- Open confidence (Choice/Score) is normalized entropy, documented as an open approximation, not TypeSafe's unpublished formula.
- Object state is canonical JSON (sorted keys) before hashing or prefix caching.

## Scorers

1. **FakeScorer** — token overlap. CPU tests and local serving only.
2. **TinyCausalLM logprob** — byte LM with KV cache. Tests/dev only. Not the research comparison.
3. **HF zero-shot continuation scorer** — `Qwen/Qwen2.5-0.5B` fp16. Prefill once, score option continuations from the KV cache. Reductions: sum / mean / PMI. Cached scores are checked against naïve re-encoding of prefix+continuation.
4. **Option-attention head** — frozen encoder hidden states `H_x`; each option is a low-rank query over state tokens; MLP scalar logit; listwise CE. Two encoders:
   - hashing toy encoder (CPU trainer proof, not a language model)
   - frozen `Qwen/Qwen2.5-0.5B` (`AutoModel`, no LM head)

Hashing is not a substitute for the Qwen comparison. The research comparison on the frozen synthetic test is zero-shot Qwen logprob vs the frozen-Qwen option head.

## Calibration

Temperature scaling `p = softmax(z / T)` is fit by grid NLL **only** on the calibration split. Validation is reserved for early stopping. Test is frozen until reporting.

## Splits

Four splits: train / validation / calibration / test. Group-aware where the dataset requires it (`docs/10-datasets.md`). Calibration is never passed to `train_option_head`.

## Serving

`jev serve` (FastAPI + Uvicorn): `POST /v1/systemone`, `GET /health`, `GET /healthz`, `GET /readyz`, `GET /v1/models`. Policy and authorization stay outside the model.

## Hardware pin

See [hardware.md](hardware.md) and `configs/hardware.yaml`. Both zero-shot and frozen-head roles use Qwen2.5-0.5B because Qwen2.5-3B fp16 does not fit on a 4 GiB GTX 1650. Later BANKING77 / Wikispeedia / 3B frozen-head runs can use a Colab T4 as a separate pin ([Hosted GPUs](../../docs/11-colab.md)); those numbers are not this v0 table.
