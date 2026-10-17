# H22 Kaggle Batch V3 — Raw Retained Partial Evidence

Date: 2026-10-17 (Asia/Riyadh). Kaggle notebook: https://www.kaggle.com/code/abdulazizshehri/hala-h22-test-batch-replication-20261017 (private).

**Status: COMPLETED_WITH_BLOCKERS — NOT qualified full-triplet evidence.**
Kaggle notebook worker ended COMPLETE, but the internal frozen TEST precision triplet did not all produce measurements.

- FP16: BLOCKED before any task inference due to an incompatible preinstalled `torchao 0.10.0` with `peft 0.20.0` when loading the existing LoRA. Its outcome is **unknown**, not 0/8.
- INT8: measured 0/8 (English 0/4; Arabic 0/4).
- NF4: measured 2/8 (English 2/4; Arabic 0/4).
- All attempted profiles report Tesla T4 `GPU-0712fe0b-f3ad-30ac-e420-9ad77e59fdbe`.
- Fixed scientific source: `07591f109b6683c9302fcf890259ddf0bde91494`; no retraining or data/tuning change.
- Preexisting interactive INT8 lost outcome is still **UNKNOWN** and unrelated to this NEW measured INT8 outcome.
- No precision adopted and no model, production, cost, security or Aladdin integration claim is justified.

## Immutable content digests

| File | SHA256 |
| --- | --- |
| `assigned-hardware.json` | `23c19ef2fad0e3e705b50cfb40e64298919d36d0abf1e336102b955ca6dae3d4` |
| `batch-status.json` | `717bfa3d4f9e2109efc470b3537673d1520aa832c454182c0d6b49c891e55700` |
| `profile-0-fp16-test-evidence.zip` | `d2c0fec245d57af458383c0064b3d714c1d3b3b788082b72c75e40e732ce44f2` |
| `profile-1-int8-test-evidence.zip` | `3c22036a2492230c7b33f3a79c0860dda5d636546866c6f710e143ad44ebcb` |
| `profile-2-nf4-test-evidence.zip` | `06c11e85ed20e56c6ea45c33c1ab6d9012add30e474f0f88c4072b917697501f` |
| `audit-v3-partial.json` | `5affaf0e3b0588bdaa10a77f07e40b6623df97bc9c71c971bb0aafc72bf4e4f8` |

An independent standard-library byte-level audit successfully compared retained ZIP and raw result digests, ZIP CRC, assigned-hardware records, model identity, source SHA, adapter manifest, 8/4/4 episode counts, bilingual outcomes, and explicit blocked FP16 error. The audit does **not** independently attest Kaggle hardware identity or prove real-world task success.

The normal H22 all-three-measured verifier must reject this collection. Retention does not authorize bypassing it. Preserve this original failure and run a newly qualified full triplet only after a separately reviewed package-compatibility correction.
