# Hala H22 — private Kaggle V4 completed TEST triplet

Date: 2026-10-17 (Asia/Riyadh). Kaggle notebook: https://www.kaggle.com/code/abdulazizshehri/hala-h22-test-batch-replication-20261017 (private).

**Status:** Kaggle Saved Version 4 worker `COMPLETE`; H22 internal status `completed_unqualified`. All three fixed synthetic eight-case TEST profiles *finished and were measured*. This is not production, representative task, Arabic, precision-equivalence, hardware-attestation, energy/cost or Orcel native-integration qualification.

| Fresh V4 precision | Verified complete-task count | English | Arabic | Internal status |
|---|---:|---:|---:|---|
| FP16 | 2/8 | 2/4 | 0/4 | finished/measured |
| INT8 | 0/8 | 0/4 | 0/4 | finished/measured |
| NF4 (4-bit) | 2/8 | 2/4 | 0/4 | finished/measured |

- All three V4 profiles record one reported Tesla T4 UUID: `GPU-a00cc5ee-b73b-753b-16e2-cdf6589573f1`. The audit checks consistent reported identity but does **not** independently attest the physical machine.
- Frozen scientific source: `07591f109b6683c9302fcf890259ddf0bde91494`.
- Frozen H22 source SHA256: `98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b`.
- Retained H19 data and H21 trained adapter unchanged. No retraining or new checkpoint was published.
- V4 had the previously reviewed environment-only repair: remove an incompatible preinstalled optional `torchao` distribution before the experiment. V1–V3 remain separate historical evidence; no results were overwritten or silently reused.
- These eight dependent, synthetic, one-click tests cannot establish general computer-use quality. Arabic **0/4 across all three** is a significant development issue, not proof of a general Arabic failure rate.
- No model/precision is adopted. No Orcel source was edited, no paid compute was activated, and founder merge/release decisions remain open.

## Independently retained files (SHA256)

| File | SHA256 |
|---|---|
| `assigned-hardware.json` | `9bf3a6f96ef13c471219984623a69c2265a55911b3dec09e8234cc9ebcccc4a4` |
| `batch-status.json` | `74a5723d58b0cdea6d2f88b6619f2abb9d075a877ce341c26f6689972ffe60f0` |
| `profile-0-fp16-test-evidence.zip` | `c696c335b69d3c20fe81f54e9f8058384021a62d1322aa4777c68c994a82deb0` |
| `profile-1-int8-test-evidence.zip` | `ff6927cca88c841ab02a737a335655a2562f52bcf84713fe53395c275b5978e5` |
| `profile-2-nf4-test-evidence.zip` | `018490f7a68285b65dfccecb1b755d447408ee1a7203cea51a2d594cbdde4e70` |
| `verified-h22-v4.json` | `c825dcbecce38d6f3bfbd1ad1bd8405f854a3a8a1597dbe89a9acc294dcf7499` |

The actual source-tested `tools/verify_h22_batch.py` independent CPU verifier returned success and `all_three_profile_archives_verified=true`; it checked complete three-profile raw ZIP/CRC, byte hashes, exact source and model revision, adapter/data/provenance, bilingual counts, reported GPU UUID, operator source bytes, and original earlier-loss labels. All such success establishes is preserved evidence **consistency**, not independent proof of a trusted GPU or quality improvement.

## Next work / gates

Review the actual wrong-target traces and Arabic/RTL error cases without modifying this frozen TEST or doing TEST-driven retraining. Build and freeze a **new** independently split development dataset, obtain stronger bilingual grounding and planning baselines, and evaluate on independent families plus representative closed-loop tasks. Keep H23 actual inference cancellation/recovery qualification separately blocked until explicit source/receipt/hardware requirements are met. Further GPU experiments require a justified prospective protocol and free-resource verification; avoid repeating this V4 TEST as optimization feedback.
