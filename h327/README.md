# H327 — One-shot original RailKit TRAIN-candidate handoff

## Source-backed defect

H325 validates 16 original browser scenes and 32 decoded RGB frames
before invoking a trusted consumer, but its legacy API then invokes the
consumer **32 times**. If a consumer fails on call 17, earlier deliveries
may already have happened. This is a partial-batch contamination risk,
not an observed private training incident.

## Opt-in contract

`produce_two_step_train_candidates(..., role=ROLE,
trusted_batch_sink=consumer)` stages the same H325 source-owned
validated rows in private memory, and calls `consumer` **exactly once**
with an immutable tuple of 32 immutable
`(png_bytes, task_text, native_label)` tuples. The consumer must return
the literal `True`; any other response or exception prevents a
successful return. Source pin, role, scene state, native browser action,
fresh RGB and distinctness checks remain the existing H325 checks.
The legacy per-row `trusted_sink` path remains available and unchanged;
specifying both consumers or neither fails closed.

**One call is not an external transaction.** A consumer that writes to
a database, object store or filesystem must itself guarantee an atomic
commit or rollback, and must not use the data without separately proven
rights, independent review and official admission. This module cannot
verify or undo side effects inside the consumer. Use only a trusted
transient in-memory consumer for current research; do not persist images,
task gold, coordinates or predictions.

## Qualification

- Invented-only adversarial tests in
  `tests/test_h327_railkit_one_shot_candidate_handoff.py` cover
  full-batch delivery, zero prevalidation calls, duplicate refusal,
  mutually exclusive consumers, literal acknowledgment, exceptions,
  and unchanged legacy delivery.
- H325 opt-in original Chromium and Edge browser qualification now also
  runs the one-shot path over all 16 original scenes, checking a single
  in-memory consumer call and no official TRAIN admission.
- Existing free GitHub Actions CPU and H325 browser workflows qualify
  this code at the exact PR head; do not infer a pass until terminal.

## Scientific and rights boundaries

H325 RailKit is an original source claim with no external media or
network assets; this code change does **not** independently certify
authorship, source history, legal rights or custody. H199 official
TRAIN admission remains empty. RailKit and BeaconBoard are publicly
exposed source families, not independent blind OOD. Historical H291/H292
paired BASE 6/8 vs ADAPTED 6/8 proves no measured learned gain;
H324 pretrained BeaconBoard 1/2 is a tiny exposed baseline, not an
adapter result. No Qwen inference, optimizer, Kaggle, model weights,
private data or release is involved in H327.
