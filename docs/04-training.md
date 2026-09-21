# Training

TypeSafe names its method **Reinforcement Learning for Calibrated
Decisions (RLCD)** and describes the objective as: do not generate text;
return decisions whose probabilities match outcomes. The reward, data,
policy parameterization, and optimizer are **not public**. Do not assume
this is PPO, DPO, or Facebook's unrelated RLCD paper.

This file is a practical open analogue, ordered so that each stage is
debuggable before the next.

## Data

Public sources, licenses, conversions, and the v0 mix (synthetic →
BANKING77 → SST-5 → BoolQ → Wikispeedia) are in
[Datasets](10-datasets.md). This section is only the on-disk row shape.

A generic labeled choice record:

```json
{
  "id": "case_004281",
  "state": {
    "user_request": "I was charged twice after checkout.",
    "account_age_days": 82,
    "has_active_order": true
  },
  "question": {
    "instructions": "Choose the correct primary resolution route.",
    "criteria": {
      "duplicate_charge": "The customer was charged more than once.",
      "refund_request": "The customer explicitly requests a refund.",
      "technical_issue": "A product or integration malfunction occurred.",
      "insufficient_information": "There is not enough information to route reliably."
    }
  },
  "gold": "duplicate_charge",
  "metadata": {
    "domain": "support",
    "group_id": "customer_1807",
    "risk_tier": "medium"
  }
}
```

The compact `jevlike` training row is enough to start:

```json
{"context": "The customer needs a refund.", "options": ["refund", "sales", "technical support"], "label": 0}
```

Rules:

- Every option the model will see at prediction time must be in the row.
- Variable `N` per row, minimum 2.
- Split by `group_id` (user, document, website, session), not by random
  rows. Adjacent records leak.
- Hold out a calibration split that is never used for early stopping.
- Include `defer` / `insufficient_information` as a first-class label
  where the gold is genuinely unknown.

See [`../schemas/training-example.schema.json`](../schemas/training-example.schema.json)
and the conversion recipes in [Datasets](10-datasets.md).

Do not train the first head on AG News or IMDB. They are easy, often
already in encoder pretraining, and they do not stress variable menus.
Carve a calibration split from the official train set (stratified);
never tune criteria text on the frozen test split.

## Where to train

The loss and split rules in this file do not change with the GPU.
The box does.

| Work | Machine |
|---|---|
| Hashing-head proofs, CPU tests, converters | Local / CI |
| v0 0.5B synthetic numbers already in `reports/v0/` | The pinned 4 GiB GTX 1650. Leave them. |
| Frozen-head on BANKING77, SST-5, BoolQ, Wikispeedia | **Google Colab T4** (or any ≥12 GiB NVIDIA). Recipe in [Hosted GPUs](11-colab.md). |
| Frozen 3B encoder (the plan's starting class) | Colab T4. Does not fit the 1650. |
| `jev serve` | Local process. Not a Colab notebook. |

Call `jev train-head` / `calibrate` / `evaluate` in both places. A Colab
cell should only provision CUDA, mount Drive, and invoke the CLI. Write
checkpoints off the ephemeral VM. Record the live GPU; `jev hardware`
still prints the repo's 1650 pin.

Do not paste Colab metrics into the v0 table without a new hardware
note and model card.

## Stage 1 — listwise supervised baseline

For labeled `(x, C, y)`:

```text
L_CE = -log pθ(y | x, C)
```

For Score, use the same loss over levels, then report

```text
expected_score = Σ softmax(level_logits)_k · k
```

For Noul, either two-class CE or
`binary_cross_entropy_with_logits(s_true - s_false, target)`.

Prefer the unified finite-option view so Choice / Score / Noul share
code.

Freeze a pretrained encoder at first. Train only the option-query
cross-attention head (`jevlike` default path, including
`Qwen/Qwen2.5-0.5B` in their example).

**Open project measurement.** On target-disjoint Wikispeedia next-click,
`jevlike` reports ~26% top-1 with frozen Qwen2.5-0.5B plus scorer versus
~8% shuffled/random controls; a small from-scratch model reached ~29% on
40k clicks. These are project-specific, not Jev-equivalence.

`open-jev` then trains the same head on frozen Gemma hidden states
(context tokens + masked-mean option vectors). On a synthetic split they
report test top-1 0.970 / ECE 0.027, with shuffled-context top-1 0.258
(chance at ~4.5 options). If shuffled accuracy does not collapse, the
head is reading option priors.

## Stage 2 — proper scoring rules

Accuracy is not the training target for a decision engine. Add a
distributional loss:

```text
L_Brier = Σ_j (pθ,j − 1[j=y])²
```

NLL is also a proper scoring rule. Proper rules incentivize honest
probabilities in the idealized i.i.d. setting.

Optional calibration regularizer:

```text
L = L_CE + λ L_cal
```

where `L_cal` penalizes disagreement between confidence and empirical
accuracy on a batch. Direct ECE optimization is unstable; start with
NLL + temperature scaling instead.

## Stage 3 — post-hoc calibration (do this before RL)

Split: train / validation / **calibration** / test.

### Temperature scaling

```text
p = softmax(z / T)
```

Fit `T > 0` on the calibration split by minimizing NLL. Preserves
ranking. First thing to try.

### Vector scaling

```text
p = softmax(Wz + b)
```

More flexible, easier to overfit.

### Isotonic regression

Monotonic map from raw confidence to empirical correctness. Needs enough
calibration data.

### Dirichlet calibration

Often the serious multiclass baseline when a single temperature is not
enough.

Fit Noul calibration separately from Choice/Score. Fit per domain if the
deployment mix is heterogeneous.

Remember: TypeSafe's product `confidence` is a peakedness statistic.
Your **calibrated p_max** and that statistic are different numbers. Log
both.

## Stage 4 — selective prediction

Add an explicit `defer` candidate, or an abstention head `a(x)`.

A simple automation utility:

```text
R =
  +1              if correct automated decision
  -λ_wrong        if wrong automated decision
  -λ_defer        if defer to a stronger model or human
```

Automate only when expected utility beats the fallback. This is closer to
the operational value of calibrated decisions than raw accuracy.

Example policy (application code, not the model):

```python
p = calibrated_probabilities(logits)
best = argmax(p)

if best == DEFER:
    return escalate()
if p[best] < threshold_for(best, state):
    return escalate()
if risk_class(best, state) == "high":
    return require_human_approval()
return execute_permitted_action(best)
```

Routing might tolerate 0.90. An irreversible tool call might still be
unsafe at 0.995. The threshold is a product risk decision.

## Stage 5 — what an open RLCD might look like

Only after stages 1–4 are measured.

If you have logged workflows with contexts, action sets, and outcomes:

- contextual bandits or conservative offline RL
- propensity scores or another defense against confounding
- a hard split between *recommendation* and *execution*

A plausible reward for calibrated decisions, **not claimed to be
TypeSafe's**:

```text
r = 1[ŷ = y] − μ · (p_ŷ − 1[ŷ = y])² − ν · 1[automate ∧ ŷ ≠ y]
```

That is: correctness, plus a proper scoring penalty, plus extra cost for
confident automation errors.

**Open project.** [Laya](07-open-equivalents.md#laya--open-weight-system-one-with-published-rlcd)
publishes a concrete recipe under the same *name* RLCD: policy emits a
distribution; exploration adds zero-mean Gaussian noise to logits; reward
is a strictly proper scoring rule (log + spherical, plus ranked
probability score for ordinal `score`); updates are REINFORCE with a
group-mean baseline (GRPO-style); multi-turn uses TD(λ=1.0) over prefix
slices. Reproduce from their Kaggle fine-tune notebook before inventing a
parallel Stage 5. Still independent research — not evidence of TypeSafe’s
optimizer.

Work already in the literature on rewarding doubt / calibrated confidence
is also relevant as prior art.

Do not start here. Supervised listwise learning + held-out calibration +
abstention is cheaper, more credible, and easier to debug.

## What not to train

**Fact.** TypeSafe's jaggedness page says Jev is not a calculator, does
not count reliably, reads dates as text, and is a poor generator even if
you chain Choices. Keep arithmetic, identity checks, and authorization in
code. Do not spend capacity teaching the model to be a worse Python.

Train the model on *semantic compatibility between state and a named
option*. Everything that is a closed-form check should stay out of the
loss.
