# Hosted Jev 1.13 limitations

An open replica that ignores these will overfit the happy path and fail
the same way the commercial model does—or worse. Source: TypeSafe
[Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13),
last reviewed there 17 September 2026. **Fact** that TypeSafe documents
these; not a claim we re-measured them.

## Failure modes to put in the eval suite

| Mode | What goes wrong | What the system should do |
|---|---|---|
| Literal reading | Scoping words, negations, implied conditions are taken at face value | Write the exact condition; put boundary cases in criteria; split implied judgments |
| Math and numbers | Unreliable counting, hex/RGB nearness, interpolating a real number from Score | Count, convert, and compute in code; ask the model only the semantic residue |
| Date and time | Dates are text, not ordered quantities | Extract closed-set parts with Choice (include "not stated"); compare in code |
| Indirection | Double negatives, property-of-a-property, multi-hop | Flatten hops; name the relevant state fields |
| Large irrelevant state | Accuracy falls as distractors grow ("context rot") | Filter in retrieval/code first; optionally Noul-filter passages |
| Adversarial content | Injected instructions and self-serving framing move the answer | Precise criteria; hostile test cases before deploy |
| Contradictory instruction vs criteria | e.g. Noul where `true` means no | Align the two; prefer phrasing a person would read the same way |
| Structural invariants | `P(A)` and `1 − P(not A)` need not match; Noul ≠ yes/no Choice | Ask one question one way; do not transfer thresholds across phrasings |
| Generation | Chaining Choices to emit text is slow and weak | Regex or a generative model proposes candidates; Jev-like model picks |

## Invariants you must **not** assume

TypeSafe's own example: the same ticket, "Is the customer asking for a
refund?" as a Noul versus as a yes/no Choice, produced numbers that do
not line up. A question and its negation as two Nouls need not sum to 1.

A Choice is a *relative* distribution over the listed options. Each Noul
is an *absolute* probability and can be low for every option at once.
Skill-suggestion style pipelines use both: Choice to pick, Nouls to
decide whether to suggest anything.

## Design rules that follow

1. Atomic questions. One snap judgment each.
2. Compose in code: weights, thresholds, authorization.
3. Prefer speculative fan-out (ask questions you might not need) over
   sequential agent loops, except when the next option set truly depends
   on the previous answer.
4. Keep high-impact actions behind deterministic checks and a human
   fallback.
5. Pin model version once thresholds are tuned; aliases move.

## What an open model should add, not copy blindly

Hosted Jev is text-only and English-first. An open engine can choose to
support other languages or modalities, but then those are *extensions*,
and they need their own calibration plots. Do not silently mix a vision
encoder into a text decision head and report the numbers as Jev-like.
