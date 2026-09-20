# Public record

What is actually known about Jev as of 19 September 2026, versus what
this repository infers in order to build an open equivalent.

## Product identity

**Fact.** TypeSafe AI (San Francisco) came out of stealth on 15 September
2026 with Jev as its first public "System One" model. Founder Diogo
Almeida previously worked on instruction-following methods at OpenAI
(InstructGPT / related RLHF work). The company reports a $40M seed round
(third-party coverage; not independently verified here).

**Fact.** The names are intentional:

- *System One* — Kahneman's fast, intuitive judgment, as opposed to slow
  System 2 reasoning. TypeSafe's claim is that this class can be made
  *more* reliable than unconstrained generation for bounded decisions.
- *Jev* — William Stanley Jevons. The bet is Jevons' paradox: cheaper
  intelligence is used more, not less.

**Fact.** Jev currently accepts **text only**: a string, a JSON object, or
an array of text. Images, audio, and video are not supported on the hosted
model. English is the primary training language.

## Public contract

**Fact.** One shared `state`, many typed questions, one structured answer
per question, evaluated independently and in parallel:

```text
state + {typed question_i}_{i=1..m}
    -->  {typed decision_i, distribution_i, confidence_i}_{i=1..m}
```

Three primitives:

| Type | Input | Output | Typical use |
|---|---|---|---|
| `choice` | Named finite criteria, 2–255 options | Selected key, distribution over keys, confidence | Route, classify, pick an action |
| `score` | Ordered rubric, 2–10 levels | Expected level index, per-level probabilities, confidence, legend | Priority, severity, quality, risk |
| `noul` | Binary proposition; optional `{true, false}` | Probability the proposition is true. **No** separate confidence field | Guardrails, eligibility, verification |

**Fact.** Question ids are not sent to the model. Option keys *and*
descriptions are. Score levels are evaluated separately; the model is not
shown the level number or its neighbours. The continuous score is the
probability-weighted mean of those indices.

**Fact.** Hosted model `jev-1.13.0`. Aliases `jev-latest` and
`jev-preview` both pointed at it on 19 September 2026. Pin the versioned
id once you tune thresholds. `GET /v1/models` lists aliases.

**Fact.** Context: 64k tokens per request for state + all questions
combined; 32k for state plus the longest single question. Choice
cardinality cap 255; above that TypeSafe uses a two-stage procedure
(independent scoring, then an explicit choice).

**Fact.** Pricing published for early access: **$0.042 per million input
tokens**, output unmetered. Rate limits on the models page (subject to
change): 250,000 tokens/s and 1,200 requests/min. End-to-end latency
claimed 70–500 ms.

**Fact.** TypeSafe does not fine-tune or LoRA-adapt Jev on customer data.
The same weights serve every account. Domain adaptation is via `state`,
`instructions`, and `criteria`. TypeSafe says it does not train on
customer requests/responses.

**Fact.** The stack TypeSafe *names* but does not specify:

- a new model architecture
- a parallel sampler
- Reinforcement Learning for Calibrated Decisions (RLCD)

## What "cannot hallucinate" means

**Fact.** Possible outputs are defined in advance. A successful response
cannot contain an undeclared option, a malformed field, or generated
prose. TypeSafe plots a 0% type-error rate and says the number is
structural, not empirical.

That is **type-safety**, not semantic correctness. The model can still
pick the wrong allowed option, misread evidence, or assign an
unjustifiably high probability. A schema without `unknown` /
`insufficient_information` forces probability onto the closest listed
key.

## What Jev is likely optimizing

The intended operating point differs from a chat / code LLM:

| Dimension | Chat / code LLM | Jev-style decision model |
|---|---|---|
| Primary output | Variable-length token sequence | Finite categorical / binary / ordinal distribution |
| Decoding | Autoregressive | Parallel or batched scoring over a known output space |
| Failure mode | Invalid JSON, invented fields | Wrong selection among allowed outcomes |
| Core objective | Helpful text or rewardable trajectory | Accurate, calibrated bounded decisions |
| Best workloads | Synthesis, explanation, coding | Routing, ranking, verification, control |
| Program integration | Parse and validate generated content | Branch on typed output |
| "No hallucination" | Not normally guaranteed | Cannot emit a value outside the declared type |

## Published evaluations

**Measurement (vendor).** TypeSafe's workflow evals use the average of two
external frontier models (they name GPT-6 Astra and Claude Fable 5.1) as
the reference probability, not human ground truth. Four workflows:
security incident response, agent-trace observability, invoice processing,
customer service. Headline homepage numbers: up to **193.6× faster** and
**444.6× cheaper**. TypeSafe says these may sit at the high end of
real-world gains and that its capability team created the workflows.

Secondary summaries of those evals (not re-run here) report Jev at about
67.8% agreement, $0.0004 and 0.4 s per case, matching GPT "Terra" / Claude
Sonnet 5 on agreement while trailing GPT "Sol" (74.1%) and Claude Opus 5
(73.1%). Invoice processing was the widest gap in that writeup.

**Measurement (third party, small).** Independent early tests reported in
the week after launch include: Every's evals team (777 judgments, <0.7 s,
~$0.0025); NearHere listing moderation (96% vs 86% Gemini Flash-Lite);
a spam-email zero-shot study tying a TF-IDF logistic regression after
criteria were revised on 1,000 labeled emails. Treat these as leads, not
as a scientific benchmark.

**Fact.** TypeSafe's Doom demo uses structured game state as data, not
pixels. Wikiracing is the high-cardinality demo.

## Confirmed unknowns

| Area | Public status |
|---|---|
| Parameter count | Not disclosed |
| Base architecture | Not disclosed |
| Dense vs MoE | Not disclosed |
| Encoder-only vs decoder-derived vs bidirectional | Not disclosed |
| Layers / width / heads | Not disclosed |
| Tokenizer | Not disclosed |
| Pretraining corpus | Not disclosed |
| Decision-training corpus | Not disclosed |
| RLCD reward function | Not disclosed |
| RLCD optimization algorithm | Not disclosed |
| Calibration method and ECE / Brier tables | Not disclosed |
| Official open weights | No |
| Official source code | No |
| Self-hosted deployment | No |
| Paper sufficient for reproduction | No |

So "reimplement Jev" in this repository means: **implement an open
Jev-like decision engine with the same task interface and measurable
properties.**
