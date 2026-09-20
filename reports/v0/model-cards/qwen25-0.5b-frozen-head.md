# Model card — frozen Qwen2.5-0.5B option-attention head

`AutoModel` last hidden states, all encoder parameters frozen. Trainable: low-rank option query + MLP logit.

- **Checkpoint:** encoder `Qwen/Qwen2.5-0.5B` fp16; head saved without encoder weights (`runs/synthetic-hf-head.pt`, gitignored).
- **CLI:** `jev train-head TRAIN.jsonl --val-jsonl VAL.jsonl --encoder hf --model-id Qwen/Qwen2.5-0.5B --batch-size 1 --epochs 12` then `jev calibrate CAL.jsonl` then `jev evaluate TEST.jsonl --out …`.
- **This run (GTX 1650 4 GiB):** full synthetic train (172 rows), batch 1, early stop epoch 7 on validation (not calibration). train_acc 0.273 = shuffled_acc 0.273. Frozen test n=12 acc **0.083** = shuffled 0.083. T=5.0 on 48 calibration rows. **No OOM.**
- **Interpretation:** underfit. Zero-shot Qwen logprob on the same frozen test (acc 0.75) is the stronger Qwen result. BANKING77 / Wikispeedia / 3B encoder belong on a Colab T4 with a new hardware note.
- **Evidence:** `reports/v0/metrics/eval-hf-head-synthetic-test.json`, `calibrate-hf-head.json`, `verification-commands.txt`.
