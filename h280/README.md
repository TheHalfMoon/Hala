# H280 — closed private Qwen GUI evidence validation

This is a **source-only, standard-library-only, no-model-data** verifier
for the bounded sanitized aggregates from two **previously exposed**
H279 original GUI research families. It does not download private
artifacts and does not qualify a checkpoint, adapt a model, admit
TRAIN/DEV samples, evaluate blind OOD, or measure improvements.

## Why

H279 Stockroom FREE owner-private T4 **V1 and V2** each completed and
returned real Qwen3-VL image-conditioned **1/2 source-owned task-state
success**, two real native clicks and two strict constrained actions.
The source-owned original scene identities and input model were
previously exposed. V2's strict relative-to-pixel order repair did
**not** change these two results; neither V1 nor V2 proves trained
Hala improvement. Separate original Queueboard H279 experiment
initial V1 ended ERROR without a safe aggregate, and later corrected
kernel version 3 was **QUEUED** at last verified check; do not invent
accuracy for a failed run. H280 eliminates manual ad-hoc report
inspection and standardizes a non-leaking fail-closed receipt check.

## Exact source contracts

The verifier has two explicit sources:

- `stockroom`: report basename
  `hala-h279-private-relative-native-aggregate.json`, pinned source
  SHA `53d7fb25e113f8f2cd337bb4aebeafac026d1a22`, two preregistered
  original Arabic scenes, required exact two model-time grammar calls,
  genuine native action dispatch, bounded original-source target/state
  outcome. It does **not** infer a normalized model point from any
  source-owned DOM fixture.
- `queueboard`: report basename
  `hala-h279-private-queueboard-native-aggregate.json`, pinned source
  commit `c61409e0b0178227180f0d8e05c67f5a210d01f9` and exact
  Git blob `3cf452e84a75b017905bc14b04a8571411929891`, two
  preregistered English and Arabic real GUI tasks, strict kind,
  action and real source-owned task-state fields.

Both pin `Qwen/Qwen3-VL-2B-Instruct`, exact Hugging Face revision
`89644892e4d85e24eaac8bacfd4f463576704203`, original T4 free
attestation, absence of optimization, no official TRAIN/DEV admission,
no blind OOD, no weights/media publication, and no learned gain claim.

The verifier:

1. Reads **only** an explicitly named local JSON file of 100–4096
   bytes using Python standard library. It must not be a symlink.
2. Rejects duplicate keys (including escaped aliases), unexpected
   nested keys, non-UTF8, invalid JSON and NaN/Infinity.
3. Verifies exact model/source provenance, hardware, two real
   generations and model-time grammar calls, exact scene order and
   closed schema with strictly typed booleans (not 0/1 integers).
4. Rejects logically contradictory outcomes (e.g. claiming success
   without an executed click, scores exceeding reported totals).
5. Emits **only** names of preregistered source cases, format labels,
   action kind and Boolean safe aggregate counters; never raw
   predictions, image pixels, target coordinates, screenshot, model
   weights or unknown JSON content.

Example, **only if the owner-private kernel is actually COMPLETE**,
and **only after downloading the single allowed report basename via
Kaggle `--file-pattern`** into a new short-lived D: scratch location:

```powershell
python -m h280.verify_private_native_aggregate --family queueboard --report "D:\\<owner-temp>\\hala-h279-private-queueboard-native-aggregate.json"
```

Treat all generated output as *research-only evidence*. An untrusted
JSON can be made to satisfy these checks: verifying content fields is
**not** equivalent to authenticating Kaggle execution. Always
independently check the owner-private Kaggle job status, exact submitted
source/version and GPU logs without exposing private raw predictions.

Use `python -m unittest discover -s tests -p test_h280_closed_private_evidence.py -v`.
Test cases are invented synthetic data and can demonstrate rejection,
**not** real Qwen inference. Source-only GitHub actions CI
and Windows 9/9 regression qualification are separate from real model
quality. No private aggregates, model media or weights should ever be
committed. Remain DRAFT and do not merge/release.
