# Colab T4 / hosted-GPU public-split metrics

Do not mix these rows into [`../v0/metrics.md`](../v0/metrics.md).

Comparison protocol: same frozen official JSONL; temperature fit on the
calibration split; stratified `--limit 300 --seed 0` of the frozen test.
Majority and TF-IDF are **CPU** on that slice (hardware-independent).
Qwen rows below are from a live Colab T4 session. Empty cells remain unused,
not invented.

GPU pin for these Qwen rows: **Colab T4** (see [`hardware.md`](hardware.md)).
Do not run those jobs on the laptop GPU. `jev hardware` still prints the
GTX 1650 repo pin.

## BANKING77 (Choice, 77)

| Model | Accuracy | NLL | Brier | ECE | Coverage at 1% error | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.013 | 24.026 | 1.950 | 0.975 | 0.00 | 0.013 |
| TF-IDF + logistic | 0.870 | 1.009 | 0.377 | 0.387 | 0.42 | 0.020 |
| Frozen Qwen2.5-0.5B option head | 0.013 | 4.358 | 0.987 | 0.006 | 0.00 | 0.013 |
| Zero-shot Qwen2.5-0.5B logprob | 0.143 | 4.134 | 0.976 | 0.104 | 0.01 | 0.023 |

Sources: [`metrics/eval-banking77-majority.json`](metrics/eval-banking77-majority.json), [`metrics/eval-banking77-tfidf.json`](metrics/eval-banking77-tfidf.json), [`metrics/eval-banking77-hf-head.json`](metrics/eval-banking77-hf-head.json), [`metrics/eval-banking77-hf-logprob.json`](metrics/eval-banking77-hf-logprob.json). Head T=5.0, zero-shot T=3.0, n=300. GPU: Colab T4.

## SST-5 (Score, 5)

| Model | Accuracy | NLL | Brier | ECE | MAE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.200 | 22.105 | 1.600 | 0.800 | 1.400 | 0.200 |
| TF-IDF + logistic | 0.383 | 1.417 | 0.730 | 0.044 | 0.932 | 0.180 |
| Frozen Qwen2.5-0.5B option head | 0.200 | 1.611 | 0.801 | 0.019 | 1.204 | 0.200 |
| Zero-shot Qwen2.5-0.5B logprob | 0.247 | 1.539 | 0.773 | 0.101 | 1.098 | 0.203 |

Sources: [`metrics/eval-sst5-majority.json`](metrics/eval-sst5-majority.json), [`metrics/eval-sst5-tfidf.json`](metrics/eval-sst5-tfidf.json), [`metrics/eval-sst5-hf-head.json`](metrics/eval-sst5-hf-head.json), [`metrics/eval-sst5-hf-logprob.json`](metrics/eval-sst5-hf-logprob.json). Head T=5.0, zero-shot T=1.5, n=300. GPU: Colab T4.

## BoolQ (Noul)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority | 0.500 | 13.816 | 1.000 | 0.500 | 0.500 |
| TF-IDF + logistic | 0.570 | 0.716 | 0.514 | 0.087 | 0.497 |
| Frozen Qwen2.5-0.5B option head | 0.500 | 0.696 | 0.503 | 0.036 | 0.500 |
| Zero-shot Qwen2.5-0.5B logprob | 0.537 | 0.693 | 0.500 | 0.013 | 0.523 |

Sources: [`metrics/eval-boolq-majority.json`](metrics/eval-boolq-majority.json), [`metrics/eval-boolq-tfidf.json`](metrics/eval-boolq-tfidf.json), [`metrics/eval-boolq-hf-head.json`](metrics/eval-boolq-hf-head.json), [`metrics/eval-boolq-hf-logprob.json`](metrics/eval-boolq-hf-logprob.json). Head T=1.5, zero-shot T=2.0, n=300. GPU: Colab T4.

## Wikispeedia next-click (variable Choice)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority | 0.003 | 27.539 | 1.993 | 0.997 | 0.003 |
| TF-IDF + logistic |  |  |  |  |  |
| Frozen Qwen2.5-0.5B option head | 0.027 | 3.738 | 0.967 | 0.033 | 0.027 |
| Zero-shot Qwen2.5-0.5B logprob | 0.063 | 3.649 | 0.960 | 0.009 | 0.013 |

Sources: [`metrics/eval-wikispeedia-majority.json`](metrics/eval-wikispeedia-majority.json), [`metrics/eval-wikispeedia-hf-head.json`](metrics/eval-wikispeedia-hf-head.json), [`metrics/eval-wikispeedia-hf-logprob.json`](metrics/eval-wikispeedia-hf-logprob.json). Head T=5.0, zero-shot T=5.0, n=300. GPU: Colab T4. TF-IDF skipped: official train is 159k rows over ~4.6k gold articles; a dense 20k-feature multinomial coefficient matrix does not fit a responsible CPU first pass.

## CLINC150 + out_of_scope (Choice, 151)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority | 0.117 | 24.407 | 1.767 | 0.883 | 0.117 |
| TF-IDF + logistic | 0.837 | 1.121 | 0.367 | 0.308 | 0.030 |
| Frozen Qwen2.5-0.5B option head | 0.007 | 5.074 | 0.994 | 0.011 | 0.007 |
| Zero-shot Qwen2.5-0.5B logprob | 0.047 | 4.481 | 0.978 | 0.030 | 0.007 |

Sources: [`metrics/eval-clinc150-majority.json`](metrics/eval-clinc150-majority.json), [`metrics/eval-clinc150-tfidf.json`](metrics/eval-clinc150-tfidf.json), [`metrics/eval-clinc150-hf-head.json`](metrics/eval-clinc150-hf-head.json), [`metrics/eval-clinc150-hf-logprob.json`](metrics/eval-clinc150-hf-logprob.json). Head T=5.0, zero-shot T=2.0, n=300. GPU: Colab T4. CLINC gold uses named intents from the Hugging Face ClassLabel (`oos` → `out_of_scope`).

## 3B VRAM probe

| Item | Result |
|---|---|
| Qwen2.5-3B `--max-steps` 2 batch 1 | fit (2 steps completed) |
