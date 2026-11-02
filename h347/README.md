# H347 — checkpoint identity uses the admitted manifest bytes

**Status: engineering-only Draft; Hala release remains NO-GO.**

## Confirmed defect

H346's trainer validates the exact bytes of an H199-admitted manifest
before reading training data. Previously it then reopened the same path
much later with `sha256_file(manifest_path)` to derive checkpoint identity.
A changed file on disk between the two reads could cause a checkpoint
to claim the SHA-256 identity of **unadmitted replacement bytes** rather
than the actual originally approved, parsed manifest. The late second
file scan is unnecessary and unbounded.

## Fix and test

`load_json(..., require_gui_admission=True, with_admitted_digest=True)`
now returns the same parsed JSON and its exact officially admitted
SHA-256 from one bounded read. `train()` uses that digest to build
`identity["manifest_sha256"]`, instead of reopening the manifest path.
Ordinary JSON callers and H346 callers without `with_admitted_digest`
are compatible; requesting a digest without admission is rejected.

An invented-only regression replaces the manifest after parsing inside
`load_authoring_records`, captures the real `train(..., resume=True)`
identity at H82 recovery and proves that the identity remains the
approved original digest, even though the on-disk path changed.
All H199 approvals in tests are temporary monkeypatches only,
with no real GPU import, optimizer update, model weights or training.

Owner-local Windows qualification:
`python -m unittest tests.test_h346_official_gui_training_admission tests.test_training tests.test_training_integrity tests.test_h199_native_action_sft tests.test_h82_orphan_optimizer_recovery -q`
— **66 tests run; 65 pass, one skipped.** Hosted exact-head CI remains
a separate required qualification; the previous H346 56/56 applies to
its own exact head only.

## Research gates still blocked

Production H199 `ADMITTED_TRAIN_MANIFEST_SHAS` stays empty.
Independent rendered-image ML rights and title review, lawful
source-family-disjoint TRAIN/DEV, new unexposed blinded OOD,
genuine pinned adapted-vs-base native English/Arabic Qwen gains,
safety evidence and independent Goldfish review remain unavailable.
Do not merge protected main, deploy, publish weights or use paid GPUs.
