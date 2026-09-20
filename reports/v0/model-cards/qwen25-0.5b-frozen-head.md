# Model card — frozen Qwen2.5-0.5B option-attention head

`AutoModel` last hidden states, all encoder parameters frozen. Trainable: low-rank option query + MLP logit.

- **Checkpoint:** encoder `Qwen/Qwen2.5-0.5B` fp16; head saved without encoder weights (`runs/synthetic-hf-head.pt`, gitignored).
- **CLI:** `jev train-head TRAIN.jsonl --val-jsonl VAL.jsonl --encoder hf --model-id Qwen/Qwen2.5-0.5B --batch-size 1 --epochs 12` then `jev calibrate CAL.jsonl` then `jev evaluate TEST.jsonl --out …`.
- **This run (GTX 1650 4 GiB):** full synthetic train (172 rows), batch 1, early stop epoch 7 on validation (not calibration). train_acc 0.273 = shuffled_acc 0.273. Frozen test n=12 acc **0.083** = shuffled 0.083. T=5.0 on 48 calibration rows. **No OOM.**
- **Fixture SST-5:** train 30 / calib 5 / test 15. Test acc 0.200 = shuffled 0.200, T=5.0. Peak ≈ 1138 MiB. **No OOM.**
- **Fixture BoolQ:** train 38 / calib 3 / test 8. Test acc 0.500 = shuffled 0.500, T=2.0. **No OOM.**
- **Fixture BANKING77:** 77-way menu cached once. Test n=154 acc **0.013** = shuffled 0.013 (chance), T=5.0 on 77 calibration rows. Peak ≈ 1176 MiB. **Completed, no OOM.** An earlier uncached 12-epoch attempt was stopped at ~80 min (`eval-hf-head-banking77-fixture-test-timeout.json`).
- **Fixture CLINC150:** 151-way. Test n=26 acc 0.154 shuffled 0.077, T=1.0. Peak ≈ 1194 MiB. **No OOM.**
- **Fixture Wikispeedia:** variable-N. Test n=4 acc 0.500 = shuffled 0.500, T=3.0. Peak ≈ 1208 MiB. **No OOM.**
- **Interpretation:** underfit on every frozen-Qwen-head run here. Zero-shot Qwen logprob is the stronger Qwen result on every completed frozen test. Official full BANKING77 / Wikispeedia / 3B encoder jobs belong on a Colab T4 with a new hardware note.
- **Evidence:** `reports/v0/metrics/eval-hf-head-synthetic-test.json`, `eval-hf-head-*-fixture-test.json`, `calibrate-hf-head.json`, `calibrate-hf-head-*.json`, `verification-commands.txt`.
