# Model card — frozen Qwen2.5-0.5B option-attention head

`AutoModel` last hidden states, all encoder parameters frozen. Trainable: low-rank option query + MLP logit.

- **Checkpoint:** encoder `Qwen/Qwen2.5-0.5B` fp16; head saved without encoder weights (`runs/synthetic-hf-head.pt`, gitignored).
- **CLI:** `jev train-head TRAIN.jsonl --val-jsonl VAL.jsonl --encoder hf --model-id Qwen/Qwen2.5-0.5B --batch-size 1 --epochs 12` then `jev calibrate CAL.jsonl` then `jev evaluate TEST.jsonl --out …`.
- **This run (GTX 1650 4 GiB):** full synthetic train (172 rows), batch 1, early stop epoch 7 on validation (not calibration). train_acc 0.273 = shuffled_acc 0.273. Frozen test n=12 acc **0.083** = shuffled 0.083. T=5.0 on 48 calibration rows. **No OOM.**
- **Fixture SST-5:** train 30 / calib 5 / test 15. Test acc 0.200 = shuffled 0.200, T=5.0. Peak ≈ 1138 MiB. **No OOM.**
- **Fixture BoolQ:** train 38 / calib 3 / test 8. Test acc 0.500 = shuffled 0.500, T=2.0. **No OOM.**
- **Fixture BANKING77:** 77-way encoding fits (1198 MiB). 12-epoch train did not finish in ~80 min; stopped; not OOM (`eval-hf-head-banking77-fixture-test-timeout.json`). Official BANKING77 frozen-head is Colab T4.
- **Interpretation:** underfit on synthetic, SST-5, and BoolQ. Zero-shot Qwen logprob is the stronger Qwen result on every completed frozen test. BANKING77 / Wikispeedia / 3B encoder full jobs belong on a Colab T4 with a new hardware note.
- **Evidence:** `reports/v0/metrics/eval-hf-head-synthetic-test.json`, `eval-hf-head-sst5-fixture-test.json`, `eval-hf-head-boolq-fixture-test.json`, `calibrate-hf-head.json`, `verification-commands.txt`.
