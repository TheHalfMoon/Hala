# H306 — closed native browser transition evidence

## Concrete integrity gap in H305

H305 correctly distinguished model/policy turns from native browser
dispatches, but its arithmetic-only validator could still accept
impossible combinations: a native type without a preceding focus click,
a first-stage type, or simultaneous source refusal flags. These are
**forged evidence tuples**, not observations of a genuine Qwen run.

## Bounded source-only repair

- Require one of five realizable two-stage paths: first abstain,
  wrong-focus refusal, second abstain, wrong-code write refusal, or
  actual focus followed by native type.
- Reject type dispatch without focus, source focus refusal after the
  second model turn, simultaneous refusal flags, and non-boolean flags.
- At the H298 integration boundary, require a boolean source-owned task
  result, a recorded focus refusal after any non-abstaining first-only
  turn, and both native browser dispatches before claiming task success.
- Preserve the H305 historical `native_actions` policy-turn field and
  separate `native_dispatched_actions` counter; do not alter any
  frozen prior private GPU receipt or H300 result.
- Test with invented counters only; no raw predictions, private
  screenshots, coordinates, model weights, or gold.

## Scientific limits

This improves the *truthfulness of source-owned evaluation receipts*,
not Qwen competence. No new GPU, LoRA training, officially admitted
TRAIN/DEV, independent blind OOD, or model release. H292 paired
BASE 6/8 vs ADAPTED 6/8 still shows **zero learned improvement**.
H173 sealed OOD and H293/H294 safety blocks remain untouched.
