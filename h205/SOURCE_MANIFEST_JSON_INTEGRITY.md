# H205 — fail-closed source-manifest JSON provenance

**Research-only, no data admission.** Stacked on H204 Draft #225 at immutable parent `935ead167e2e4f7b11dfd3504017e2c3c902808c`.

## Genuine source metadata integrity gap

H198's original-app prospect census and H204's second-pass old SHA collision screen previously used default `json.loads` for new owner source manifests. Python silently retains the last of duplicate JSON keys, including keys that become identical after decoding `\\u` escapes. For example, a single source case could carry contradictory `train_eligible` values while the effective parsed value appeared false. `NaN`, `Infinity`, and `-Infinity` are also accepted by Python's permissive JSON parser despite being invalid JSON.

H205 copy-first extends H198's existing bounded, redacted source-only parser and reuses the **same strict decoder** in H204's second read. Every nested object rejects repeated decoded keys; all nonstandard constants fail closed. Errors do not print owner paths, source case IDs, instructions, digest values or raw JSON.

## Independent synthetic negative controls

`tests/test_h204_training_readiness_gap.py` exercises contradictory top-level and case-level flags, escaped Unicode key aliases, nonstandard numeric constants, and an independent H204 second-pass rejection with the H198 census stubbed. Negative controls verify that the historical metadata scan is never entered after the first-pass rejection.

Owner Windows Abdulaziz source-only worktree `e824d7f8dbeb8953904a57f5dd72b07c256d6489`:
- H204 synthetic: 13/13 normal, 13/13 `python -O`.
- Inherited H198 source metadata: 14/14 normal.
- Git state clean before tests. This is source-only validation, **not** new native browser QA or model training.

Hosted Python 3.11/3.12 CI must be checked independently at the final exact Draft head. No existing successful native Pickr/interact/medium-zoom proof is claimed anew.

## Scientific limits

All 20 original GUI source prospects remain **quarantined**. The owner-only partial six-phase historical census has 130 known old PNG digests but no recorded old decoded-RGBA digests; complete prior Git/fork/media provenance is **unproven**. H199's official TRAIN allowlist remains empty. This decoder repair does not qualify TRAIN, DEV or OOD, infer model generalization, access old consumed H173, read private media/gold, run a model/GPU, merge or publish anything.
