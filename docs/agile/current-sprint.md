# Current sprint

**Sprint:** Sprint 1  
**Dates:** 2026-09-21 — 2026-10-05  
**Sprint goal:** A researcher can train, resume, calibrate, and evaluate frozen-head versus zero-shot on frozen public splits from the Colab notebook, then file a T4 report without touching the 1650 v0 table.  
**Committed:** US-001, US-002, US-003, US-004, US-005, US-006 (23 points)

## Board

**To Do**

-

**In Progress**

- US-006

**Done**

- US-001
- US-002
- US-003
- US-004
- US-005

## Committed stories

### US-001: Resume option-head training after a disconnect
**Epic:** EPIC-01 Colab T4 v0 close  
**As a** researcher, **I want** `jev train-head` to save mid-run and `--resume` **so that** a Colab idle timeout does not throw away Wikispeedia or CLINC150 work.  
**Points:** 5  
**Status:** Ready  
**Acceptance criteria:**
- [x] Given a hashing-head synthetic train interrupted after N steps, when I `--resume` the checkpoint, then training continues from a saved step greater than zero
- [x] Given `--val-jsonl`, when I resume, then early stopping still uses validation and not the calibration split

### US-002: Calibrate zero-shot and head from one CLI
**Epic:** EPIC-01 Colab T4 v0 close  
**As a** researcher, **I want** `jev calibrate --backend` and `evaluate --temperature-json` **so that** both scorers share the dedicated calibration split without hand-copying T.  
**Points:** 3  
**Status:** Ready  
**Acceptance criteria:**
- [x] Given a calibration JSONL and `--backend tiny-logprob`, when I calibrate, then a JSON file contains `temperature` and `split: calibration`
- [x] Given that JSON, when I `evaluate --temperature-json`, then the eval payload uses that temperature

### US-003: Report a risk-coverage curve
**Epic:** EPIC-01 Colab T4 v0 close  
**As a** researcher, **I want** eval JSON to include a risk-coverage curve **so that** coverage-at-1% error is not the only deferral number.  
**Points:** 2  
**Status:** Ready  
**Acceptance criteria:**
- [x] Given an `evaluate_scorer` report, when I serialize it, then `risk_coverage` is a list of threshold/coverage/risk objects and `coverage_at_1pct` remains present
- [x] Given a perfectly confident correct set, when I compute the curve, then risk is 0 at full coverage

### US-004: Drive public-task jobs from the Colab notebook
**Epic:** EPIC-01 Colab T4 v0 close  
**As a** researcher, **I want** notebook cells for BANKING77, SST-5, BoolQ, Wikispeedia, CLINC150, zero-shot `--limit`, and a live hardware note **so that** I do not reimplement the trainer in cells.  
**Points:** 5  
**Status:** Ready  
**Acceptance criteria:**
- [x] Given the notebook source, when tests scan it, then it invokes `python -m jev` for `data-convert`, `train-head`, `calibrate`, `evaluate` on those dataset names and does not define `train_option_head`
- [x] Given the notebook, when I look for secrets, then there is no hardcoded Hugging Face token

### US-005: File honest v0 manifests and GPU-run evidence
**Epic:** EPIC-01 Colab T4 v0 close  
**As a** researcher, **I want** `reports/v0/manifests/` and a real `gpu-run.json` **so that** metrics.md does not cite missing files.  
**Points:** 3  
**Status:** Ready  
**Acceptance criteria:**
- [x] Given `reports/v0/metrics.md`, when I follow the manifest and gpu-run links, then those files exist in git
- [x] Given those manifests, when JSONL is absent from git, then notes say converted corpora stay gitignored and split counts come from the recorded conversion

### US-006: File a Colab T4 comparison report
**Epic:** EPIC-01 Colab T4 v0 close  
**As a** researcher, **I want** `reports/colab-t4/` with hardware, metrics, and model cards **so that** T4 numbers never overwrite the 1650 synthetic table.  
**Points:** 5  
**Status:** Ready  
**Acceptance criteria:**
- [ ] Given a live T4 run, when artifacts are ingested, then `reports/colab-t4/metrics.md` labels the GPU as Colab T4 and `reports/v0/metrics.md` is unchanged in its 1650 rows
- [ ] Given BANKING77, SST-5, BoolQ, Wikispeedia, and CLINC150, when the report is filed, then each has eval JSON with accuracy, NLL, Brier, ECE, shuffled-context, and risk-coverage for head and zero-shot on the same frozen comparison slice
