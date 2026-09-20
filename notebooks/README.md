# Notebooks

Research v0 only. This is not TypeSafe Jev, not RLCD, and not a place
to run `jev serve`.

## Hosted GPU (Google Colab)

Authoritative recipe: [`docs/11-colab.md`](../docs/11-colab.md).

Runnable notebook: [`jev_colab_hosted_gpu.ipynb`](jev_colab_hosted_gpu.ipynb)

[Open in Colab](https://colab.research.google.com/github/WiredMind2/jev/blob/main/notebooks/jev_colab_hosted_gpu.ipynb)

The notebook only provisions the runtime and invokes `python -m jev`.
It does not reimplement `train_option_head`. Cells write JSON under a
`jev-runs/reports/` tree that matches `reports/v0` metric keys. Do not
paste those rows into [`reports/v0/metrics.md`](../reports/v0/metrics.md)
without a **new** hardware note: Colab is a different pin from the
4 GiB GTX 1650 / Qwen2.5-0.5B table.

| Path | Machine |
|---|---|
| Hashing-head synthetic smoke | CPU (local or Colab). Proves the CLI. |
| v0 0.5B synthetic numbers already in `reports/v0/` | Leave on the 1650. Do not replace. |
| Frozen-head BANKING77 / SST-5 / BoolQ / Wikispeedia | Colab T4 (or any ≥12 GiB NVIDIA). |
| Frozen Qwen2.5-3B encoder | Colab T4. Does not fit 4 GiB. |
| 7B/8B fp16 | Not on free T4. |

Optional Hugging Face auth: Colab Secrets key `HF_TOKEN`. Never paste a
token into a cell. Qwen2.5-0.5B is typically ungated.

Local CPU proof of the same CLI sequence:
`scripts/colab_cpu_smoke.sh`.
