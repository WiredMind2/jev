# Laya architecture (for training our own)

**Open project.** Reverse-engineered from ConvAI Innovations’
[`laya`](https://github.com/NandhaKishorM/laya) package (`research`
branch, PyPI `laya>=0.3.3`), Hub configs under
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya),
and the published fine-tune notebook
[`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/research/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
Retrieved 21 September 2026.

This is **not** TypeSafe Jev and **not** a claim that we reproduce their
private RLCD. It is a concrete, Apache-2.0 recipe we can reimplement and
train on **Google Colab** (or Kaggle) when we need GPU. Local GTX 1650
stays the v0 measurement pin; see [Hosted GPUs](11-colab.md).

High-level catalog entry: [Open equivalents](07-open-equivalents.md).

---

## 1. What to copy vs invent

| Piece | Status in Laya | Our plan |
|---|---|---|
| `choice` / `score` / `noul` over shared state | Same System One surface as our schemas | Keep our `POST /v1/systemone` contract |
| Bidirectional encoder + option-marker head | Fully disclosed in `laya/common.py` | Implement as a second scorer family |
| Sequence packing (`[MASK]` per option) | Disclosed | Reuse verbatim first, then ablate |
| Soft-target CE + proper-scoring RL (their “RLCD”) | Fine-tune notebook is complete | Port to Colab; do not invent a parallel Stage 5 first |
| Base English / multilingual pretraining mix | **Not** fully published | Start from `answerdotai/ModernBERT-large` or their released checkpoint |
| Temperature buckets | Config + LBFGS fit code | Fit on our held-out JSONL the same way |

Practical fork path: treat `laya.load("convaiinnovations/laya")` as a
**baseline oracle**, then reimplement `DecisionModel` + `build_sequence`
+ the notebook’s `train_ddp.py` loop inside this repo so we own the
trainer, checkpoints, and eval.

---

## 2. Checkpoint family and shipped configs

All three live under one Hub repo; `subfolder=` downloads only that tree.

| Name | Encoder | Params (claimed) | `max_len` | `head_max_len` | Notes from `rl_agent_config.json` |
|---|---|---|---|---|---|
| English (repo root) | `answerdotai/ModernBERT-large` | 421M | 512 | 192 | Fitted temps; `amp_dtype: bf16`; training meta: 7313 updates, 1 epoch, ~2 h |
| Multilingual | `jhu-clsp/mmBERT-base` | 322M | 1024 | 256 | Temps all `1.0` (none fitted); 15987 updates, 4 epochs, ~5 h |
| Typed-decisions | ModernBERT-large | 421M | 1024 | 256 | Fine-tuned from English; `gradient_checkpointing: true`; `max_tokens_per_batch: 4096` |

Shared config fields (all checkpoints):

```json
{
  "head_layers": 2,
  "max_prefixes": 6,
  "act_costs": { "escalate": 0.5 },
  "cost_wrong_act": 3.0
}
```

English temperature vector (choice, score, noul) and option-count
buckets (example):

```text
temperature = [1.637, 1.251, 1.983]
temperature_by_options =
  choice:2 → 1.906
  choice:3-5 → 1.760
  choice:6-10 → 1.000
  choice:11+ → 0.101   # note: near-zero temp → very sharp softmax
  score:3-5 → 1.251
  noul:2 → 1.983
```

Bucket key format: `"{qtype}:{size}"` where size is `2`, `3-5`, `6-10`,
or `11+` (`temp_bucket` in `laya/common.py`).

---

## 3. Input sequence format

Every question becomes **one packed sequence**. Format from
`build_sequence`:

```text
[CLS]
  "{qtype} question: {instructions}"     # truncated into option budget
[SEP]
  [MASK] {option_0 text truncated to ≤48 tokens}
  [MASK] {option_1 text}
  ...
[SEP]
  {serialized state}                     # fills remaining room
[SEP]
```

Length budget:

- Total length ≤ `max_len`.
- Option block + question head share `head_max_len` tokens.
- If options do not fit: each option is cut to
  `per = max(4, (head_max_len - 16) // n_options)`, then the instruction
  head is truncated so the option block still fits.
- State uses the leftover tokens (`max_len - len(head) - 1`). Default
  truncates **right** (`st[:room]`); `truncate_left=True` keeps the
  suffix.

Option text rendering (`render_options`):

| Type | Rendered strings |
|---|---|
| `choice` | `key` or `key: description` per criteria entry |
| `score` | `level i: {criterion}` for each rubric level |
| `noul` | always two options: `false: …`, `true: …` (defaults if criteria omit them) |

State serialization: strings pass through; `dict` / `list` become
`json.dumps(..., ensure_ascii=False)`.

**Marker positions** are the absolute token indices of each `[MASK]`.
Those positions are what the scorer reads — not pooled option spans.

Critical capacity limit: at Banking77 scale (~77 options) with
`head_max_len=256`, each option gets ~3–4 tokens. Labels become
indistinguishable. Mitigations they recommend (and we should adopt):

1. Raise `head_max_len` / `max_len` at runtime, or
2. Hierarchical coarse→fine choice.

---

## 4. Model architecture (`DecisionModel`)

Source of truth: `laya/common.py` class `DecisionModel`.

```text
input_ids, attention_mask
        │
        ▼
HF bidirectional encoder (ModernBERT / mmBERT)
  → last_hidden_state H ∈ R[B, T, d]
        │
        + type_emb[qtype] broadcast to every token   # Embedding(3, d)
        │
        ▼
optional TransformerEncoder head (head_layers=2)
  Pre-LN TransformerEncoderLayer(d, nhead=d//64, ff=4d, dropout=0.1)
  key_padding_mask = ~attention_mask
        │
        ├──────────────────────────────┐
        ▼                              ▼
gather H at marker_pos            CLS token h[:,0]
  m_j = H[marker_j]               + feats from logits:
        │                           [top1, top1−top2, H(p)/log k, k/255]
        ▼                              │
scorer: LN → Linear(d,d) → GELU → Linear(d,1)
  → logit s_j per marker               ▼
mask invalid markers with −1e4    act_head: Linear(d+4,256)→GELU→Linear(256, n_act)
        │
        ▼
return (logits [B,K], act_logits [B, n_act])
```

Details:

- `d = encoder.config.hidden_size` (ModernBERT-large: 1024; mmBERT-base:
  typically 768 — confirm from encoder config when loading).
- `nhead = max(1, d // 64)`.
- `scorer` is shared across all options and question types; type identity
  enters only via `type_emb`.
- `detach_encoder=True` is supported for frozen-encoder ablations
  (forward path exists; fine-tune notebook does **not** use it — encoder
  is trained).
- Encoder loaded with `attn_implementation="sdpa"`. At inference they
  force `encoder.config.reference_compile = False` so ModernBERT does
  not `torch.compile` (bad for tiny batches; can hang).

Parameter groups in the fine-tune notebook:

| Group | LR |
|---|---|
| `encoder.*` | `2.5e-5` |
| everything else (head, type_emb, scorer, act_head) | `1.0e-4` |

Weight decay `0.01`, AdamW, cosine anneal to `1e-6`, grad clip `1.0`,
fp16 autocast + GradScaler, gradient checkpointing on the encoder.

---

## 5. Inference (`Agent.system_one` / `predict`)

For a request with questions `{qid → {type, instructions, criteria}}`:

1. Convert each question to internal `{t, ins, crit}`.
2. `build_sequence` → `(ids, markers)` per question.
3. Collate **all questions in one call** into a batch
   (`collate_items`): pad sequences, pad marker slots to `kmax`,
   `marker_mask` marks valid options.
4. One forward: `logits, act = model(...)`.
5. Per question `r`:
   - `t_scale = temperature_by_options[bucket] or temperature[qtype]`
   - `p = softmax(logits[r,:k] / t_scale)`
   - Decode by type:

| Type | Response fields |
|---|---|
| `choice` | `choice=argmax`, `probabilities` keyed by criteria keys, `confidence` |
| `score` | `score=E[level]`, `legend`, `probabilities` over `"0"…"K-1"`, `confidence` |
| `noul` | `noul=p[true]`, `confidence=max(p, 1−p)` |

`confidence` for choice/score is **normalized Shannon entropy**:

```text
confidence = 1 − H(p) / log(k)     # clipped to [0,1]
```

Each answer also carries `action.act_probability = softmax(act)[0]`
(probability of the “act” class vs escalate — see §7).

Token usage reported as `input_tokens = attention_mask.sum()`,
`output_tokens = 0` (non-autoregressive).

Device policy: CUDA → MPS → CPU; OOM falls back to CPU with a warning.
AMP dtype from config (`bf16` preferred) but forced to fp16 on pre-Ampere
GPUs and fp32 on CPU/MPS.

---

## 6. Act / escalate head

Separate 2-way head on `[CLS]` plus distribution features. Config:

```text
act_costs = { "escalate": 0.5 }
cost_wrong_act = 3.0
n_act = len(act_costs) + 1   # → 2 logits: act vs escalate
```

Features concatenated with CLS:

```text
[ p_top1, p_top1 − p_top2,  H(p)/log(k),  k/255 ]
```

where `p = softmax(logits.detach())` (stop-grad from scorer into act
features).

The published fine-tune loss multiplies `act` by **zero**
(`+ 0.0 * act.sum()`), so the typed-decisions fine-tune does **not**
supervise this head. Treat it as optional / inherited from base
pretraining until we design an act/escalate label.

---

## 7. Their “RLCD” training recipe (fine-tune)

This is the loop in `train_ddp.py` from the Kaggle notebook — the only
complete public training recipe. Name overlap with TypeSafe’s RLCD is
**theirs**; do not equate the algorithms.

### 7.1 Data

Dataset: [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)

| Split | Cases | Decisions (5 q / case) |
|---|---|---|
| train | 1,200 | 6,000 |
| test | 400 | 2,000 |

Workflows (300 train / 100 test each):
`agent_trace_observability`, `customer_service`,
`invoice_processing`, `security_incidents`.

Each row: `state`, `questions`, `gold` as JSON strings. Gold includes
**soft** `probabilities` (teacher ensemble), not only argmax labels.
Training builds one sequence per question with:

```text
target = gold probabilities over options   # renormalized
label  = argmax(target)
```

So the objective matches distributions, not one-hot only.

### 7.2 Hyperparameters (typed-decisions fine-tune)

| Knob | Value |
|---|---|
| GPUs | 2× T4, DDP (`torchrun --nproc_per_node=2`) |
| Epochs | 4 |
| Micro-batch | 8 sequences / GPU |
| Grad accum | 4 → effective batch **64** (= 8 × 2 × 4) |
| Group size G | 4 (GRPO-style baseline samples) |
| σ exploration | 0.4 → 0.1 linear across epochs |
| Loss | `L_rl + 1.0 * L_ce` |
| Proper reward | `w_sph=0.75`, `w_rps=1.0` |
| Context override | `max_len=1024`, `head_max_len=256` |
| Checkpointing | encoder gradient checkpointing on |

Wall-clock: their Hub meta reports ~2 h for the published typed-decisions
checkpoint (`hours: 1.96`); marketing text says ~4–5 h — plan a Colab
session that can checkpoint mid-run.

### 7.3 Forward + proper scoring reward

From `proper_reward(q, target, qtype, mask)`:

```text
log_score = Σ_j target_j · log q_j          # floor log at −9.21
spherical = (target · q) / ||q||₂
r = log_score + w_sph · spherical

if qtype == score:
  rps = Σ_j (CDF_q(j) − CDF_t(j))² / (K − 1)
  r = r − w_rps · rps
```

`q` here is a **noisy** softmax over logits (exploration), not the
policy’s clean distribution.

### 7.4 Exploration and policy gradient

Per micro-batch logits `z₀ ∈ R[B,K]` (masked):

```text
ε ~ N(0, σ²) elementwise, then zero-mean within each option set:
  ε ← (ε − mean_valid(ε)) · mask

z_g = z₀.detach() + ε_g          # g = 1..G
q_g = softmax(z_g)

r_g = proper_reward(q_g, target, ...)
adv_g = (r_g − mean_g(r)) / (std_g(r) + 1e-6)

# Gaussian score-function log-density of the noise (not of categorical samples):
log π(ε|σ) = − ||z_g − z₀||² / (2 σ²)     # summed over valid options

L_rl = − mean(adv · log π)
L_ce = − mean( Σ target · log_softmax(z₀) )
L = (L_rl + L_ce) / grad_accum
```

Interpretation: GRPO-style group baseline over Gaussian logit
perturbations, plus full soft cross-entropy to the teacher distribution.
This is **not** REINFORCE over discrete option samples; the categorical
decision is always `softmax(z₀)` at eval time.

`td_lambda_targets` exists in `common.py` for multi-turn `noul`
trajectories (`ep_group` / `ep_step` in the batch). The typed-decisions
notebook does not use it — single-step soft targets only.

### 7.5 Post-train temperature fit

After training, rank 0:

1. Run ~400 stratified items (`all_items[::15][:400]`).
2. Collect raw logits per `(qtype, target)`.
3. Fit one scalar `T` per qtype with LBFGS minimizing
   `−mean(target · log_softmax(z / T))`, clamp `T ∈ [0.1, 10]`.
4. Write `temperature` into `rl_agent_config.json`.

For production they also recommend fitting **per option-count bucket**
(as shipped on the English checkpoint). Code path:
`temperature_by_options[temp_bucket(qtype, k)]`.

---

## 8. On-disk checkpoint layout

What `Agent` expects under a model directory:

```text
model_dir/
  rl_agent_config.json     # required
  model.safetensors        # full DecisionModel state_dict
  encoder/                 # HF encoder config (+ weights if bundled)
  tokenizer/               # HF tokenizer files
```

Hub bundle layout:

```text
convaiinnovations/laya/           # English at root
  multilingual/                   # subfolder
  typed-decisions/                # subfolder
```

`laya.load(repo, subfolder=...)` uses Hub `allow_patterns` so only one
subfolder downloads.

Compatibility checks on load: config must include `encoder`,
`head_layers`; weight keys must include prefixes
`encoder.`, `type_emb.`, `scorer.`, `act_head.`.

---

## 9. Router (serving concern, not required to train)

`Router` picks english / multilingual / typed-decisions **before** the
forward pass using Unicode script + lightweight language guess
(`laya/lang.py`). Reason: English ModernBERT stays high-confidence while
collapsing on non-Latin scripts (Khmer 0.000 acc @ 0.95 confidence in
their report). Confidence gating cannot fix that.

For our first Colab train we can ignore routing and train a single
English ModernBERT (or start from their English weights).

---

## 10. How this differs from our v0 option-attention head

| | This repo v0 (`option_head.py`) | Laya `DecisionModel` |
|---|---|---|
| Encoder | Frozen Qwen2.5-0.5B (causal as encoder) or hashing toy | Fully fine-tuned ModernBERT / mmBERT |
| Option scoring | Option text → query; cross-attn over **state tokens only** | Options and state in **one** sequence; score at `[MASK]` positions |
| Menus | Variable; separate option encoding | Variable; capacity limited by `head_max_len` |
| Loss (v0) | Listwise CE on gold index | Soft CE + proper-scoring RL on teacher distributions |
| Calibration | Separate `jev calibrate` | Built-in temperature (+ buckets) |
| Hardware pin | 1650 / Qwen 0.5B | T4-class; 322–421M encoder |

Both are Route B in [Systems design](03-systems-design.md) (dedicated
decision head, not causal continuation logprobs). Laya is closer to a
**masked option marker** design; ours is closer to **jevlike**
cross-attention. Worth implementing Laya’s packing as a second head so
we can A/B on the same JSONL.

---

## 11. Colab plan to train *our* version

GPU work goes on Colab (user decision). Do not mix Colab metrics into
`reports/v0` without a new hardware note ([Hosted GPUs](11-colab.md)).

### Phase A — reproduce their fine-tune (oracle)

1. Colab T4 (or Kaggle 2×T4 if available).
2. `pip install "laya>=0.3.3" datasets transformers safetensors`.
3. Run their notebook against `LocalLLaMA/typed-decisions` **or** a
   trimmed single-GPU port of `train_ddp.py` (set `world_size=1`,
   halve micro-batch if OOM).
4. Eval on the official test split with the same metrics table
   (accuracy, soft acc, Brier, ECE, score MAE).
5. Save `runs/laya-typed-decisions-repro/` + JSON metrics under
   `reports/laya-repro/` (new pin, not v0).

Success = we can load the result with `laya.Agent(path)` and beat the
majority baseline; matching 0.766 is optional.

### Phase B — reimplement inside this repo

1. Add `src/jev/scoring/laya_marker.py` (or similar) porting
   `DecisionModel`, `build_sequence`, `proper_reward`,
   `confidence_from_probs` under our license/headers.
2. CLI: `jev train-marker` parallel to `jev train-head`.
3. Data: convert our BANKING77 / SST-5 / BoolQ / synthetic JSONL into
   soft-target rows (one-hot is fine to start; soft teacher later).
4. Start encoder from `answerdotai/ModernBERT-large` **or**
   `convaiinnovations/laya` weights (Apache 2.0). Freezing the encoder
   first is the cheaper ablation (`detach_encoder=True` / stop-grad).
5. Train on Colab; calibrate temperatures; evaluate with existing
   `jev evaluate` once the scorer adapter speaks our response schema.

### Phase C — map to our API

Adapter from Laya-style answers → our
`schemas/systemone.response.schema.json` (field names already nearly
align: `choice` / `score` / `noul` / `probabilities` / `confidence`).

Serve only outside Colab (`jev serve` on a real host).

### Suggested first Colab cell budget

| Step | Approx. VRAM | Notes |
|---|---|---|
| Inference-only `laya.load` English | ~1.5–2 GB fp16 | Sanity + latency |
| Fine-tune head-only (encoder frozen) | fits free T4 | Fast ablation |
| Full encoder + head, `max_len=512` | T4 | Match English config |
| Full `max_len=1024` + grad ckpt | T4 tight / 2×T4 better | Match typed-decisions |

Always `USE_TF=0` when importing transformers if TensorFlow is present
(their note: abseil deadlock).

---

## 12. Unknowns and honesty checks

- **Base pretraining mix** for English / multilingual checkpoints is not
  in the public fine-tune notebook. Starting from their released
  weights is the honest shortcut; training ModernBERT from scratch on
  AG News/BoolQ/etc. is a separate research track.
- Their arXiv cites ([2503.23303](https://arxiv.org/abs/2503.23303)
  SalesRLAgent, [2510.01237](https://arxiv.org/abs/2510.01237)
  confidence routing) are **prior vertical work**, not the Laya
  `DecisionModel` paper. Prefer the GitHub sources above for
  implementation.
- Jev numbers in their tables are third-party; they did not call the
  TypeSafe API. Use them as orientation only.
- Notebook cell text that says “~4 to 6 minutes” contradicts Hub
  `hours: 1.96` and README “4–5 hours” — trust Hub meta + plan for
  hours, not minutes.
- `choice:11+` temperature `≈0.10` on the English checkpoint is extreme;
  verify behavior on high-cardinality menus before copying blindly.

---

## 13. Source index

| Artifact | URL |
|---|---|
| Package / architecture | https://github.com/NandhaKishorM/laya/tree/research/laya |
| Fine-tune notebook | https://github.com/NandhaKishorM/laya/blob/research/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb |
| Benchmark harness | https://github.com/NandhaKishorM/laya/tree/research/research |
| Hub English config | https://huggingface.co/convaiinnovations/laya/blob/main/rl_agent_config.json |
| Hub typed-decisions config | https://huggingface.co/convaiinnovations/laya/blob/main/typed-decisions/rl_agent_config.json |
| Hub multilingual config | https://huggingface.co/convaiinnovations/laya/blob/main/multilingual/rl_agent_config.json |
| Train/eval dataset | https://huggingface.co/datasets/LocalLLaMA/typed-decisions |
| Product write-up | https://laya.convaiinnovations.com/ |
| PyPI | https://pypi.org/project/laya/ |
