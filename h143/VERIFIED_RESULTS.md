# H143 — actual frozen shifted-layout model/browser transfer

## Status: independent same-app layout challenge evaluated, NOT release-qualified

The baseline is a PRIVATE research LoRA of **external Qwen3.5-2B**,
not the flagship Hala 35B. H142 achieved 16/16 visible heading point
hits on same-layout DEV16; its heading outputs had only two fixed
positions. H143 tests that limitation in genuinely shifted original
browser screenshots WITHOUT further training or consulting H140 BLIND.

## New source and frozen holdout

- Original MIT React Kanban code remains unmodified at SHA
  `fa047f7a32afb57750ed5246456dc282ea7befcd`.
- Browser-injected CSS shifts the original THREE columns among four
  preregistered layouts: up, down, far_down, left_up.
- Exactly 16 fresh original screenshot tasks: 4 layout cases each;
  4 Arabic+Doing, 4 Arabic+Done, 4 English+Doing, 4 English+Done.
- All **16 actual reference mouse drags persisted** the requested
  card move across browser reload. All 16 fresh original SHA screenshots
  are disjoint from previously excluded historical H136–H142, the H141
  consumed neural test, and the sealed H140 OOD BLIND source.
- Owner source manifest SHA-256:
  `57272bc4e056aaf6ad8c3da049a06868a596e823a88b477be3ad7ec6fc6be330`.
- Oracle-free private inference ZIP SHA-256:
  `ff341686f7bb1cae305dea83d848f3af1e5604de6f73008f9529e33ab897bb90`.
- Inference ZIP manifest SHA-256:
  `29a2b780d71aadf0e9aff2785ea204966407ce2bd52b39422ca7f1409740fe11`.
- A SECOND isolated Chromium pass reproduced **all 16 native original
  screenshots pixel-for-pixel**, matching the SHA used by the model.

## Actual model inference and native browser outcomes

Private no-cost Nvidia T4 Kaggle kernel:
`abdulazizshehri/hala-h143-private-shifted-original-2b-t4`.
Owner-only dataset:
`abdulazizshehri/hala-h143-private-shifted-layout-eval-v1`.

Kernel status **COMPLETE**, with **32 actual GPU inference calls**
(START+visible-heading for each of 16 images) and **ZERO new optimizer
steps**. Original previously trained H142 2B adapter SHA-256:
`992fa18fead6724329dbe7ec7b95c8acb0a56229c1fa5b56719e5f4829130a68`.

The owner Windows one-shot evaluator admitted exact model-image-source
SHA and exactly 16 paired source/heading model predictions. No actual
DOM gold coordinates were supplied to the native mouse controller.
All 16 browser scenes matched the model-observed original pixels.

| Measure | H143 genuine new shifted layouts |
| --- | ---: |
| START predicted inside actual visible source card | **16/16** |
| Predicted heading inside actual shifted visible destination | **15/16** |
| BOTH predicted points within target boxes | **15/16** |
| Requested task changed and **survived page reload** | **8/16** |
| Genuine new model-train optimizer steps on H143 holdout | **0** |
| Calls on H140 sealed unseen OOD BLIND | **0** |

Actual persistent task completion by stratum:
- Arabic **4/8**, English **4/8**.
- Doing **5/8**, Done **3/8**.
- Layout down **4/4**, far_down **1/4**, left_up **1/4**, up **2/4**.

The source-only reference-actuator result of 16/16 is **not** neural
success; model-driven result is exactly **8/16**, and must not be
reported as 15/16 successful browser tasks.

## Interpretation and next gate

This is evidence of partial positional transfer within ONE original
MIT application across genuinely different CSS column layouts and
both languages/goal columns. It does NOT prove general computer-use
competence or transfer to other applications.

Among 15 geometrically correct START/heading pairs, only 8 tasks
persisted. The remaining failures indicate an action/drag-release
policy weakness even when the model points are close to correct.
The current deterministic actuator moves the cursor to 12 pixels
below a nominal rectangle centered on the predicted heading. Native
drag/drop hit testing is more stateful than one point-in-box metric.

**Do not re-run or tune on this consumed H143 cohort.** The one-shot
owner lock `H143_SHIFTED_LAYOUT_ONCE_MODEL_ACTION_CONSUMED.json`
and full case outcomes remain owner-private. For H144, independently
collect new original scenes and preregister a vision-observe-drag
policy (no hidden DOM target in model input) to test the gap between
point grounding and truly persisted task success. Preserve previous
results and the sealed H140 OOD BLIND, using free compute only.

No paid GPU, public dataset/weights, GitHub merge, model release,
or Hala flagship 35B change.
