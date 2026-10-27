# H150 — genuine owner-private independent native drop release improvement

This is a **source-reference, zero-neural-call** test of a new
screenshot-only native drag actuator, **NOT** accuracy of the Hala 35B
flagship or the external Qwen3.5-2B research model.

## Preregistered method

The unchanged original MIT React Kanban app, pinned donor Git SHA
`fa047f7a32afb57750ed5246456dc282ea7befcd`, is run in isolated
Chromium sessions. A NEW 16-task owner-private evaluation corpus with
IDs >=154000 (4 Arabic/Doing, 4 Arabic/Done, 4 English/Doing,
4 English/Done) and four CSS-shifted column layouts is used.

Both arms use identical native-source VISIBLE card-center references and
noisy VISIBLE destination-heading points (inside its header bbox).
Neither mouse controller reads the hidden native drop rectangle, DOM
element/CSS class, any event instrumentation or task state before
mouse-up.

- **Baseline H147:** detect a coarse screenshot difference in the
  destination region, release on the first visual activation.
- **New H150:** detect the first change; compare the pre-drag screenshot
  with the newly rendered browser image and estimate the bounding
  vertical span of the newly visible destination-drop region using
  actual changed pixel samples. Move the still-pressed pointer to an
  **interior inferred release point** before mouse-up, then verify the
  visual signal remains present. This is image-only; no oracle hidden
  geometry is passed to the controller.

## Actual new native Chrome results

| Model-independent actuator | Persisted requested task after reload |
|---|---:|
| Prior H147 visual entry | **8/16** |
| H150 image-difference inferred drop interior | **16/16** |
| Only H150 succeeds | **8/16** |
| Only H147 succeeds | **0/16** |

All **32 actual original native mouse drags** completed as trials, and
every successful transfer was independently verified after a FULL page
reload. The H150 visual region was identified and its pixel-change
signal remained active on all **16** cases. The difference spans four
genuinely CSS-shifted layouts and both languages/target columns.

These 16 examples were never used in H143's frozen real model cohort,
H145–H149 previous controller tests, or source TRAIN/DEV.
**Do not retest H150 to tune thresholds on its consumed episodes**.

## Scope and next gate

The improvement qualifies this **native controller strategy on NEW
original React reference-point QA only**, NOT a 100% model success
claim or cross-app transfer. Next, **freeze a new H151 evaluation
cohort**, make 32 actual inference calls with the **unchanged H142 2B
LoRA adapter** on new SHA-qualified original shifted screenshots, and
drive the H150 screenshot-only actuator with **only model-predicted**
START and visible destination-heading points. Check native task
persistence after reload. Do not unseal H140 independent OOD BLIND.

No training steps, paid GPU, model promotion, Hala 35B update, public
weights, GitHub merge or release. The complete owner-private results
are `private-h150/visual-diff-center-eval16-v1.json`.
