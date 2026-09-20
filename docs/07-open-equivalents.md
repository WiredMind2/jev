# Open equivalents

Independent experiments. **None of these is a verified reproduction of
TypeSafe's private model, sampler, or RLCD.** Study them for interface,
systems tricks, and training baselines.

## `open-jev` — zero-shot log-probability scorer

- https://github.com/daseinlabs/open-jev

Fastest path to the **systems** reconstruction. Local Gemma 3 4B through
MLX: prefill shared context once, expand the KV cache across options,
score each candidate as a continuation, softmax. Implements
`POST /v1/systemone` with Choice / Score / Noul.

Useful for:

- prefix / KV-cache reuse
- batched candidate continuation scoring
- length and PMI normalization
- API compatibility
- demonstrating why raw option logprobs are uncalibrated

It is not a training recipe for a general calibrated foundation model.
A later path in the same repo trains `jevlike`'s cross-attention head on
frozen Gemma features.

## `jevlike` — trainable option-attention scorer

- https://github.com/vinnylarouge/jevlike

Strongest public **model-side** reconstruction. Variable-menu classifier
from JSONL. Scratch byte encoder or frozen HF encoder (example:
Qwen2.5-0.5B). Reports top-1 / top-3, ECE, shuffled-context controls.
CPU / CUDA / MPS. Separate visual game experiments exist in-tree; they
are not required for a text System One clone.

Natural project to fork for Stage 3 in [Implementation](06-implementation.md).

## Catalog

- https://github.com/cobanov/awesome-jev — source-reviewed ecosystem list
  (official docs, SDKs, community clients, evals, reproductions). Review
  notes dated 19 September 2026.

Other reproductions listed there (treat as experiments):

| Project | What it explores |
|---|---|
| [LitJev](https://github.com/zhengxuyu/litjev) | Qwen → `/v1/systemone`, no extra training |
| [kev](https://github.com/jaredpalmer/kev) | Qwen2.5-0.5B adapter + decision head, public weights |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | Small parallel-decision model + training pipeline |
| [openjev-sglang](https://github.com/ekzhang/openjev-sglang) | Prefill-only serving of open models |
| [Simple Jev](https://github.com/featherless-ai/simple-jev) | Logit readout without a trained classifier head |
| [Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev) | ModernBERT + calibrated uncertainty |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | Typed option readout from frozen open models |
| [PlayJev](https://github.com/OmniJev/PlayJev) | Open 0.8B vision-language move picker |

## Official TypeSafe surface (not open weights)

Use these to stay aligned with the hosted contract, not to train:

- Docs: https://docs.typesafe.ai/
- Launch post: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Python SDK: https://github.com/typesafe-ai/typesafe-sdk-python
- JS SDK: https://github.com/typesafe-ai/typesafe-sdk-js
- LLM adapter for comparisons: https://github.com/typesafe-ai/system-one-adapter-python
- Agent skill: https://github.com/typesafe-ai/skills

Provider listings (same hosted model, different envelopes): Cloudflare
Workers AI `typesafe/jev`, OpenRouter `typesafe/jev-1.13`, Vercel AI
Gateway `typesafe-ai/jev`.
