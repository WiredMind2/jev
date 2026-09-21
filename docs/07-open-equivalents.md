# Open equivalents

Independent experiments. **None of these is a verified reproduction of
TypeSafe's private model, sampler, or RLCD.** Study them for interface,
systems tricks, and training baselines.

## `laya` — open-weight System One with published RLCD

- Hub: https://huggingface.co/convaiinnovations/laya
- Code / benchmarks: https://github.com/NandhaKishorM/laya (`research` branch)
- PyPI: `pip install laya` (≥0.3.3)
- Demo: https://huggingface.co/spaces/convaiinnovations/laya-demo
- Write-up: https://laya.convaiinnovations.com/

Strongest public **end-to-end** System One stack as of this note: same
`choice` / `score` / `noul` primitives over a shared state, single forward
pass, Apache 2.0 weights, and an explicit open RLCD recipe (proper scoring
rules + REINFORCE / GRPO-style updates). Not TypeSafe Jev; treat numbers
as Laya's own claims unless re-run here.

Three checkpoints under one Hub repo (subfolder download):

| Checkpoint | Encoder | Params | Context | Role |
|---|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 | English base |
| `…/laya` + `subfolder=multilingual` | mmBERT-base | 322M | 1024 (enc. to 8k) | 100+ languages |
| `…/laya` + `subfolder=typed-decisions` | ModernBERT-large | 421M | 1024 | Fine-tuned on four typed workflows |

Architecture (their disclosure): fully fine-tuned bidirectional encoder +
a small decision head (2 transformer layers, option-marker scorer at
`[MASK]` tokens, act/escalate head). All questions in one call share one
forward pass. A `Router` picks English vs multilingual vs typed-decisions
from script / language signals before inference — needed because the
English checkpoint stays highly confident while collapsing on non-Latin
scripts.

### Why it matters for this repo

| This project’s stage | Laya counterpart |
|---|---|
| Stage 1 zero-shot logprob (Qwen) | Different route: bidirectional encoder, not causal continuation scoring |
| Stage 3 option-attention head | Closest peer: option markers + variable menus at request time |
| Stage 4 temperature / ECE | They ship over-confident; fit per-(type, #options) temperatures (ECE 0.466→0.081 claimed) |
| Stage 5 open RLCD | First public recipe that *uses the RLCD name*: Gaussian logit noise, log+spherical (+ RPS for ordinal), REINFORCE with group-mean baseline; TD(λ=1) on multi-turn prefixes |
| Serving / API | SDK `agent.predict(state, questions)` — map against our `POST /v1/systemone` schemas |

Reported strengths (third-party Jev numbers mixed in; verify before
citing as fact): ~33 ms / question on T4; strong on AG News / DAIR /
typed-decisions after fine-tune; multilingual routing. Known ceilings they
state honestly: base checkpoints near chance on typed-decisions zero-shot
(~0.35); Banking77-scale menus (~77 options) collapse under default
`head_max_len` (raise budget or hierarchical choice); ordinal `score`
weakest (SST-5 ~0.37).

Practical next steps here:

1. Add Laya as an external baseline row in the v0 eval table (BANKING77,
   SST-5, BoolQ, AG News / DAIR if fixtures exist).
2. Read their fine-tune notebook before inventing a Stage 5 reward —
   `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`.
3. Do **not** swap this repo’s Qwen 0.5B pin for ModernBERT without a new
   hardware note; Laya’s 322–421M encoders need more VRAM than the GTX
   1650 v0 pin.

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

## Evaluation harnesses (use their tasks, not only their code)

- [4esv/jev-eval](https://github.com/4esv/jev-eval) — BANKING77 Choice,
  SST-5 Score, IMDB Noul; 300 stratified items; Jev vs OpenRouter.
- [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks)
  — calibration, selective coverage, BTZSC slices (AG News, BANKING77,
  DAIR Emotion).
- [baibizhe/jev-decision-benchmarks](https://github.com/baibizhe/jev-decision-benchmarks)
  — MetaTool, When2Call, BFCL V4 action/abstention.

Dataset mapping for those tasks: [Datasets](10-datasets.md).

## Catalog

- https://github.com/cobanov/awesome-jev — source-reviewed ecosystem list
  (official docs, SDKs, community clients, evals, reproductions). Review
  notes dated 19 September 2026.

Other reproductions listed there (treat as experiments):

| Project | What it explores |
|---|---|
| [**Laya**](https://github.com/NandhaKishorM/laya) | Open-weight System One + published RLCD (see section above) |
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
