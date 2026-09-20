# Contributing

This repository is independent research documentation. It is not affiliated
with TypeSafe AI.

Useful contributions:

- Closing an unknown in `docs/01-public-record.md` with a **primary**
  source (TypeSafe docs, launch post, SDK, or a paper they publish).
- Measurements with grouped splits, ECE / Brier, shuffled-context
  controls, and a pinned model/schema version.
- Schema fixes that match the public HTTP contract more closely.
- Dataset conversions that follow `docs/10-datasets.md` (license, grouped
  split, frozen criteria text, no raw corpus in git).
- Implementation PRs that follow `docs/06-implementation.md` without
  claiming to be Jev.
- Colab / hosted-GPU runs that follow `docs/11-colab.md` and
  `notebooks/jev_colab_hosted_gpu.ipynb`: invoke the CLI, persist
  checkpoints off the VM, and ship a new hardware note rather than
  rewriting the v0 1650 pin. Optional `HF_TOKEN` belongs in Colab
  Secrets, never in cells.

Please tag claims as Fact, Measurement, Inference, or Open project, the
same way the existing docs do.

Do not add TypeSafe trademarks to logos or README badges in a way that
implies official status. Do not commit API keys, customer data, or
copyrighted TypeSafe eval dumps.
