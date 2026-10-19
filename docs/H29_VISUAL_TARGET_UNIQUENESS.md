# H29: reject visually ambiguous point proposals even when one target is excluded

Status: **CPU security repair, DRAFT, no model qualification, no native action or release**. Hala only, stacked on H28/H28b Draft PR #11 at exact `3399b7fcb3637a058b38838bfcca2c004255a626`.

## Reproduced issue / independent impact boundary

The optional H16 candidate path decodes a model's normalized `[x,y]` point, then maps it onto the `eligible` observed accessibility targets. A target overlapping ANY declared redaction is excluded from `eligible`, but its other unredacted points remain part of the original accessibility geometry. If an excluded element overlaps the chosen point and exactly one eligible element also contains the point, the previous filter treated the point as unambiguous. A masked target could therefore disappear from the *ambiguity test* even while the point was inside its accessible rectangle.

Concrete synthetic reproduction on Windows:
- `Save` target `[10,20,100,30]`; second target `[40,30,35,30]`.
- Redaction `[45,55,3,3]` hits only the second target at a different location (not the chosen point).
- Synthetic engine emits normalized `[63,59]`, decoded as pixel `[50,35]`, inside **both** target rectangles.
- Before repair, optional candidate `Service.infer("v2_action", ...)` returned an **unqualified** click proposal `point=[50,35]` attributed to `Save`. No native action was executed and this does not prove host authorization was bypassed.

## Patch

Candidate binding now insists that the selected point lies in exactly **one target in the entire observed accessibility set**, AND that same target is in the eligible subset. It fails closed with `invalid_backend_output` for a point inside an excluded/otherwise overlapping observed target. Existing protected-target and redaction checks remain active; the validation does not turn unqualified inference into production execution.

Regression:
- A genuinely reproduced red test was captured before the source fix (`F.`; the overlapping excluded-target case incorrectly received a successful proposal).
- After patch, the same test rejects as `invalid_backend_output` and normal candidate cases remain green.
- A distinct redacted target whose bounds **do not** include the chosen point stays correctly eligible for the first target (negative control).
- **Limitation:** accessibility may include overlapping parent/child rectangles intentionally. This stricter gate may increase abstentions. That is the selected fail-closed behavior; real UI/Arabic end-to-end applicability has not been measured.

## Evidence, limits and remaining gates

All tests are offline CPU with black synthetic PNGs and a fake unqualified model. They prove only deterministic target geometry gating, not HalaType actual neural grounding, visual understanding, user permission, real screenshots, completion, GPU cancellation or deployment. The response retains `execution_authorized=false`, `confidence=null` and independent-verification requirement. Revalidate observations at host execution time; no target clicked by these tests.

Actual Windows Hala regression after patch: `python -m unittest discover -s tests` reported **455 tests run, 454 passed, 1 skipped**. `python -m compileall` of changed source/tests PASS. Genuine local Graft **wiring-only** build/check PASS (**1,396 nodes; 3,645 edges**); deep semantic graph remains unbuilt. Genuine Alibaba Open Code Review `ocr review --preview` listed the changed source; it did **NOT** run vendor model review and excluded the new tests by default. No PStack independent reviewer, paid GPU, actual external model, live UI, in-flight H23 CUDA or production execution is claimed. Exact-head GitHub CI must be verified after push. Founder merge/release review remains required.
