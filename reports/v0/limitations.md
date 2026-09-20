# v0 limitations

This is research v0. It is not TypeSafe Jev and does not reproduce RLCD.

## What this run does not claim

- No factual-correctness guarantee from typed outputs.
- No authorization: model output never sets access boundaries.
- No match to hosted Jev 0.78 BANKING77 (jev-eval, n=300). That is a later bar.
- Frozen Qwen option head underfit on 64 synthetic rows. Zero-shot beat it on the same frozen test. Do not read the hashing 100% as an LM result.
- JSON-LLM and encoder-classifier baselines from `docs/05-evaluation.md` were not run.
- Multi-question amortization (stage 5) is specified, not profiled.
- Isotonic / Dirichlet calibration was not fit; temperature scaling only.
- Latency, P95, and cost per 1k decisions were not measured.

## Data limits

- `google/boolq` omits Wikipedia titles. Grouping falls back to a passage hash. Fixture BoolQ still groups by title.
- BANKING77 was loaded from `mteb/banking77` (9993/3076) after `PolyAI/banking77` failed unauthenticated. Counts differ slightly from Casanueva's 10003/3080.
- Wikispeedia SNAP archives are not vendored. Conversion is proven on a committed fixture that includes `<` back-clicks and a 260-degree hub (capped at 255, gold kept).
- Public intent/sentiment sets are likely in LM pretraining. Shuffled-context is the contamination control used here.

## Jaggedness (hosted Jev, not re-measured)

TypeSafe documents literal reading, bad math, dates-as-text, context rot, and injection sensitivity. This replica inherits those failure modes wherever it uses an LM backbone. Arithmetic, identity, and authorization stay in code.

## Confidence vs calibration

Normalized-entropy `confidence` is decisiveness. ECE uses predicted-class `p_max`. TypeSafe's product confidence formula is unpublished. Do not mix those numbers in one threshold.
