# H18 adaptation readiness

H18 prepares a prospective single shared Mind/Type multitask LoRA experiment. It
does not train, download weights, submit jobs, or retain a learned quality adapter.
Readiness is blocked by incompatible/missing data, not by a missing launch switch.

`experiments/adaptation/readiness-audit.json` is a deterministic audit of checked-in
bytes. It records source, protocol/contract, candidate, H14/H15 and media hashes;
rights evidence; split/family/language counts; and explicit hard stops. The audit
verifies all candidate media and original source identities and reconstructs each
original prompt and first-action label. This is an existing-artifact inventory,
not a general dataset validator or a license determination for outside material.
The audit first pins the reviewed historical source, 58-row examples and media
manifest, H14 tasks/freeze/raw measured result, and H15 report/independent verification
record to source-controlled SHA-256 identities. The H15 independent record also binds
the original candidate manifest and examples. Each image must retain its canonical
`images/<original-task-id>.png` identity and the digest in that pinned media catalog.
Updating paired hashes cannot admit replacement screenshots or rewritten measurements.
Score recomputation remains a consistency check after historical identity verification.
New data or evidence requires a separately reviewed snapshot; this audit cannot import it.

The original source has 180 synthetic tasks: train 116, dev 32, test 32. Its candidate
contains 58 English train rows across eight workflow families. The converter excluded
58 Arabic train records because image shaping was unqualified. Completions are
20 semantic_invoke, 20 click and 18 set_value actions. The click labels use window
pixel coordinates, image_scale and element_ref_hint. All 58 prompts include observed
controls and incidental fixture IDs in their goal. These properties differ from
image-plus-semantic-goal normalized point prediction and from bound Mind plans.
No compatible normalized Type, Mind plan, bilingual semantic, or independent adaptation
dev/test set is present. Original CC0 synthetic rights do not cure these mismatches.

H14 raw outputs rescore to 30/64 hits: English 16/32 and Arabic 14/32. Transit arrivals
is 1/8, warehouse stock 2/8 and document versions 2/8; calendar dates and seat locations
are 7/8 each. These narrow, correlated authored fixtures suggest semantic/numeric
selection weaknesses; they do not establish an Arabic-specific cause or general
desktop performance. H14 tasks and families remain excluded from training.

H15 independently verified real two-step LoRA update/save/reload equivalence. The
two English microbatches establish engineering behavior only. The runtime expired;
there is no retained learned quality adapter and no demonstrated quality improvement
or Arabic remediation. The audit checks the H15 report's hash against the stored
independent verification record; it does not rerun GPU work or recover checkpoints.

`experiments/adaptation/shared-multitask.json` is a prospective specification accepted
only in its disabled form. It names one pinned base and one prospective multitask
adapter, requires reviewed rights/compatible bilingual data/independent splits, and
has no datasets or runnable job path. Unsafe overrides and unknown readiness claims
fail. The detailed admission and comparison protocol is in
`specs/H18-adaptation-readiness.md`. Separate adapters need later empirical justification.

## Reproduce

From the repository root, with the existing bundled Python and Pillow test dependency:

```powershell
rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -m tools.audit_adaptation_data --output experiments/adaptation/readiness-audit.json
rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -m unittest tests.test_adaptation -v
```

The audit's underlying Python exit2 means valid evidence with blocked readiness;
RTK may expose any nonzero child exit as exit1. Exit0 is reserved for readiness and
is unreachable in this preparation-only implementation. Invalid input raises an
error and does not produce a replacement report. Remove stale output before treating
a failed audit as a fresh report.

Tests were first run red (missing training.adaptation), then implemented green.
Focused coverage includes tampering with source-bound rows despite recomputed hashes,
rights/media, split/language/H14 relabeling, H14 score rewriting and config bypasses.
The initial full CPU suite passed 145 tests in 19.569 seconds. Independent review then
reproduced coherent image/catalog substitution, fabricated raw H14 outputs and altered
metadata/H15 claims. Six additional regression tests ran red (nine failing cases), then
passed after the historical identity and strict claim checks were added. The revised
full CPU suite passed 151 tests in 14.188 seconds. Independent Jev,
PStack and Graft review remain parent-owned gates; CPU tests establish no quality claim.
