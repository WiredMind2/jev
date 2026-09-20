# v0 limitations

This is research v0. It is not TypeSafe Jev and does not reproduce RLCD.

## What this run does not claim

- No factual-correctness guarantee from typed outputs.
- No authorization: model output never sets access boundaries.
- No match to hosted Jev 0.78 BANKING77 (jev-eval, n=300). That is a later bar.
- Frozen Qwen option head underfit on the CLI `train-head --encoder hf` run (full synthetic train, batch 1, GTX 1650): test acc 0.083. Zero-shot beat it on the same frozen test (0.75). Fixture frozen heads: SST-5 0.20, BoolQ 0.50, BANKING77 0.013 (chance), CLINC150 0.154, Wikispeedia 0.50. Do not read the hashing 100% as an LM result.
- BANKING77 fixture frozen-head **fits in 4 GiB** (completed run peak ≈ 1176 MiB after encoder-text cache; earlier uncached attempt used 1198 MiB for ~80 min). Official full-split 77-way / Wikispeedia / 3B jobs stay on Colab T4.
- JSON-LM and TF-IDF baselines **were** run on the frozen synthetic test (`eval-json-llm-synthetic-test.json`, `eval-tfidf-synthetic-test.json`). Official BANKING77/Wikispeedia Qwen scoring is a Colab T4 job. Fixture BANKING77 zero-shot **was** scored on this 1650 (n=154, acc 0.786).
- Multi-question amortization (stage 5) is specified, not profiled.
- Isotonic / Dirichlet calibration was not fit; temperature scaling only.
- Latency, P95, and cost per 1k decisions were not measured.
- No live Colab / T4 measurement from the v0 1650 box. The hosted-GPU
  path is specified in `docs/11-colab.md` and
  `notebooks/jev_colab_hosted_gpu.ipynb`; the gate and what was (not)
  executed are in [colab.md](colab.md). BANKING77 / Wikispeedia / 3B
  numbers require a T4 run filed as a **new** hardware note, not an
  edit of this 1650 pin.

## Data limits

- Original BoolQ JSONL (`storage.googleapis.com/boolq/train.jsonl`) returned **HTTP 403 Forbidden** from this host. Conversion therefore used `google/boolq`, which omits Wikipedia titles. Grouping falls back to a passage hash. Fixture BoolQ still groups by title.
- BANKING77 was loaded from `mteb/banking77` after `PolyAI/banking77` failed unauthenticated. Counts differ slightly from Casanueva's 10003/3080.
- Wikispeedia SNAP **was** downloaded and converted (see `reports/v0/manifests/wikispeedia-snap.json`). The committed fixture still proves `<` stack replay and the 255-cap-keeping-gold rule without vendoring the dump.
- Public intent/sentiment sets are likely in LM pretraining. Shuffled-context is the contamination control used here.

## Jaggedness (hosted Jev, not re-measured)

TypeSafe documents literal reading, bad math, dates-as-text, context rot, and injection sensitivity. This replica inherits those failure modes wherever it uses an LM backbone. Arithmetic, identity, and authorization stay in code.

## Confidence vs calibration

Normalized-entropy `confidence` is decisiveness. ECE uses predicted-class `p_max`. TypeSafe's product confidence formula is unpublished. Do not mix those numbers in one threshold.
