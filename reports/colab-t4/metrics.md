# Colab T4 / hosted-GPU public-split metrics

Do not mix these rows into [`../v0/metrics.md`](../v0/metrics.md).

Comparison protocol: same frozen official JSONL; temperature fit on the
calibration split; stratified `--limit 300 --seed 0` of the frozen test.
Majority and TF-IDF are **CPU** on that slice (hardware-independent).
Qwen rows are filled only after a live CUDA run; empty cells are unused,
not invented.

GPU pin for remaining Qwen rows: **Colab T4** (see [`hardware.md`](hardware.md)).
Do not run those jobs on the laptop GPU. `jev hardware` still prints the
GTX 1650 repo pin.

## BANKING77 (Choice, 77)

| Model | Accuracy | NLL | Brier | ECE | Coverage at 1% error | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.013 | 24.026 | 1.950 | 0.975 | 0.00 | 0.013 |
| TF-IDF + logistic | 0.870 | 1.009 | 0.377 | 0.387 | 0.42 | 0.020 |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob |  |  |  |  |  |  |

Sources: [`metrics/eval-banking77-majority.json`](metrics/eval-banking77-majority.json), [`metrics/eval-banking77-tfidf.json`](metrics/eval-banking77-tfidf.json).

## SST-5 (Score, 5)

| Model | Accuracy | NLL | Brier | ECE | MAE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.200 | 22.105 | 1.600 | 0.800 | 1.400 | 0.200 |
| TF-IDF + logistic | 0.383 | 1.417 | 0.730 | 0.044 | 0.932 | 0.180 |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob | 0.243 | 1.539 | 0.773 | 0.104 | 1.098 | 0.197 |

Sources: [`metrics/eval-sst5-majority.json`](metrics/eval-sst5-majority.json), [`metrics/eval-sst5-tfidf.json`](metrics/eval-sst5-tfidf.json), [`metrics/eval-sst5-hf-logprob.json`](metrics/eval-sst5-hf-logprob.json). Zero-shot T=1.5 on calibration n=300. GPU: RTX 4050, not Colab T4.

## BoolQ (Noul)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority | 0.500 | 13.816 | 1.000 | 0.500 | 0.500 |
| TF-IDF + logistic | 0.570 | 0.716 | 0.514 | 0.087 | 0.497 |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob | 0.537 | 0.693 | 0.500 | 0.013 | 0.517 |

Sources: [`metrics/eval-boolq-majority.json`](metrics/eval-boolq-majority.json), [`metrics/eval-boolq-tfidf.json`](metrics/eval-boolq-tfidf.json), [`metrics/eval-boolq-hf-logprob.json`](metrics/eval-boolq-hf-logprob.json). Zero-shot T=2.0 on calibration n=300. GPU: RTX 4050 (see hardware.md), not Colab T4.

## Wikispeedia next-click (variable Choice)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority | 0.003 | 27.539 | 1.993 | 0.997 | 0.003 |
| TF-IDF + logistic |  |  |  |  |  |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob |  |  |  |  |  |

Sources: [`metrics/eval-wikispeedia-majority.json`](metrics/eval-wikispeedia-majority.json). TF-IDF skipped: official train is 159k rows over ~4.6k gold articles; a dense 20k-feature multinomial coefficient matrix does not fit a responsible CPU first pass.

## CLINC150 + out_of_scope (Choice, 151)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority | 0.117 | 24.407 | 1.767 | 0.883 | 0.117 |
| TF-IDF + logistic | 0.837 | 1.121 | 0.367 | 0.308 | 0.030 |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob |  |  |  |  |  |

Sources: [`metrics/eval-clinc150-majority.json`](metrics/eval-clinc150-majority.json), [`metrics/eval-clinc150-tfidf.json`](metrics/eval-clinc150-tfidf.json). CLINC gold uses named intents from the Hugging Face ClassLabel (`oos` → `out_of_scope`).

## 3B VRAM probe

| Item | Result |
|---|---|
| Qwen2.5-3B `--max-steps` 2 batch 1 | not executed |
