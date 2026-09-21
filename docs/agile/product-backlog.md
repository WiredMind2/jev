# Product backlog

**Product:** Jev research package  
**Product goal:** An open, measurable finite-option decision engine with the same *task interface* as public System One (Choice / Score / Noul), not a clone of TypeSafe Jev or RLCD.

## Epics

- **EPIC-01 Colab T4 v0 close** — Compare cached zero-shot and frozen option-head on frozen public splits using a Colab T4, without rewriting the GTX 1650 `reports/v0` pin. Stories: US-001, US-002, US-003, US-004, US-005, US-006.
- **EPIC-02 Later calibration and serving** — Icebox stages from `docs/06-implementation.md` weeks 4–6. Stories: US-010, US-011, US-012.

## Backlog stories

None Ready outside the current sprint. Pull from Icebox only after the T4 comparison report exists.

## Icebox

### US-010: Fit isotonic or Dirichlet after temperature
**Epic:** EPIC-02 Later calibration and serving  
**As a** researcher, **I want** a second-stage calibrator on the calibration split **so that** a single temperature is not the only reliability tool.  
**Points:** 5  
**Status:** Needs refinement  
**Acceptance criteria:**
- [ ] Given logits from a frozen test, when I fit isotonic or Dirichlet on calibration only, then test NLL is reported next to temperature-only NLL
- [ ] Given the trainer, when early stopping runs, then the calibration split is still unused

### US-011: Injection-bearing inputs in CI
**Epic:** EPIC-02 Later calibration and serving  
**As a** researcher, **I want** option-order and injection ablations in CPU tests **so that** a prompt-injection state cannot silently pass CI.  
**Points:** 5  
**Status:** Needs refinement  
**Acceptance criteria:**
- [ ] Given an injection-bearing state, when I score a frozen menu, then the eval JSON records the ablation name
- [ ] Given option-order permutation, when I score the same gold, then CI fails if accuracy is unchanged while shuffled-context is also unchanged (leakage)

### US-012: Profile shared-state multi-question scoring
**Epic:** EPIC-02 Later calibration and serving  
**As a** researcher, **I want** 5–50 questions against one state profiled **so that** prefill versus scoring cost is measured before any serving claim.  
**Points:** 8  
**Status:** Needs refinement  
**Acceptance criteria:**
- [ ] Given one state and 5–50 questions, when I score them, then a report logs prefill time versus continuation time
- [ ] Given Colab or local CUDA, when I profile, then `jev serve` is not the measurement vehicle
