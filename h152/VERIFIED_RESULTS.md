# H152 — independent horizontal vision-only native drag controller A/B

Owner-verified research evidence, NOT a trained-model score.

## Why H152 exists

The H151 frozen external Qwen3.5-2B research model plus H150's
image-difference controller completed **12/16** original browser tasks
on newly shifted layouts. Of the four failures, three were in the
`left_up` layout, despite source and heading model points still
falling inside their visible card and heading boxes. The three
heading predictions were roughly 38px right of the owner's visible
heading center, with no detected active image-difference region.
The fourth H151 failure was a true heading point error (~61px
vertically). All H151 cases are consumed and are NEVER rerun here.

## Preregistered new original owner-only test

- Source: pinned original unmodified MIT React Kanban application,
  browser-injected shifted layout CSS only.
- New task source IDs **>=174000**, exactly 16 complete cases,
  independent of all prior model and actuator evaluations.
- Four cases each for Arabic/Doing, Arabic/Done, English/Doing,
  English/Done, with FOUR cases per layout (`up`, `down`,
  `far_down`, `left_up`).
- Both comparison arms received an identical exact visible source
  card center and an identical *model-like but artificial* heading
  point displaced horizontally by **+/-38px** from the true visible
  heading center (zero vertical error); all points remained inside
  the visible original heading bbox.
- Baseline: frozen H150 screenshot-difference single vertical scan,
  infer a release point using rendered destination pixel changes.
- New H152: try frozen horizontal lanes `[0,-40,+40,-80,+80]` relative
  to the **supplied heading point**, with the same vertical
  screenshot-based drop activation and interior release finder.
- Neither mouse controller ever reads hidden native drop target
  geometry, DOM drag-event telemetry, task status or oracle heading
  boxes. The owner checks actual application task status AFTER release,
  then separately verifies that the result persists after reload.
- The A/B policy was authored and frozen BEFORE the new test results.

## Actual independent original browser outcomes

| Original native controller | Persisted requested task success |
|---|---:|
| H150 baseline under +/-38px heading X error | **0/16** |
| H152 screenshot-only horizontal lane search | **15/16** |
| H152-only successes | **15/16** |
| H150-only successes | **0/16** |

32 genuine native mouse gestures executed. Both arms used isolated
browser contexts with matching task seeds. The new controller detected
and retained a visible screenshot-difference signal in 16/16 cases,
but only 15 persisted a correct target task move; **visual detection
is still not sufficient proof of native drop**.

Owner-private post-hoc failure analysis: the one H152 failure was
English/Doing in the `down` layout, using a -40px horizontal lane,
with a screenshot region marked stable. This is not a reason to
tune the consumed H152 cohort and retest it.

No neural inference, optimizer steps, paid compute, H140 sealed
BLIND calls, flagship Hala 35B change or release. The model-driven
H151 result stays **12/16**; the controller-only H152 **15/16** result
must not be mislabeled as a 93.75% neural computer-use score.

## Next qualification

A separate NEW H153 once-only original-browser neural cohort,
sha-frozen and owner-private, must evaluate the **unchanged external
H142 2B research adapter** with its model-predicted START and heading
points driving the H152 screenshot-only controller. Compare genuine
task persistence after reload; preserve prior one-shot locks and do
not unseal H140 OOD BLIND. One-app transfer is not proof of
cross-app generalization.

Private detailed results:
`private-h152/independent-horizontal-vision-ab16-v1.json`.
