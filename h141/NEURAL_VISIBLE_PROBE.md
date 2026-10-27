# H141 original visible semantic goal — real neural probe

**Phase:** distinct H141 source-only, independent diagnostic and model evaluation;
**NOT** frozen H140 BLIND and **NOT** any new flagship Hala 35B training.

## Why this separate benchmark exists

The original `dandemslay/React-KanbanBoard` application hides a specific
drop-zone rectangle (`.hide_drop {opacity:0}`) until dragging.
H140 tried to infer that invisible, unnamed rectangle from the initial
screenshot. Increasing END sample weight from V2 to V3 regressed, so the
old H140 DEV and sealed BLIND are not used for this recovery.

H141 instead asks the model to infer the **visible source task card** and
the **visible destination-column heading** from a real at-rest screenshot.
A separate deterministic actuator then approaches 12 px below the
visible column heading while dragging, without DOM access to the
invisible destination slot. The policy may fail for other apps/layouts;
it is not claimed as a universal computer-use controller.

## Evidence achieved on owner Windows

- 8 of 8 native reference-based coarse approach actions persisted the
  requested card to Doing after full reload; 4 EN and 4 AR. The
  **source and heading points were supplied by the owner test DOM**,
  **not predicted by a neural model**. This is an actuator source QA.
- 8 original hidden→visible reference trajectories were captured with
  two screenshots each, 4 EN/4 AR, all eight actual native reference
  drags persisted. Their DURING screenshots contain oracle-pointer
  highlights and remain **INELIGIBLE** for ML training/evaluation.
- A separate 16 original visible-at-rest source pilot has 8 EN/8 AR,
  16 distinct 960x720 original SHA PNGs, 16 real persistent reference
  drags and a visible heading gold box. It is explicitly diagnostic only.
- An entirely NEW third source cohort at original task IDs **48000+**
  contains **16 evaluation-only examples** (8 EN / 8 AR). All original
  browser reference drags actually persisted. Native PNGs disjoint from
  **993 earlier prior-source hashes**, including H140 TRAIN/DEV/BLIND
  and H141 oracle-guided diagnostics. Training eligibility is **false**
  and model-evaluation eligibility is **true at capture time**.
  The private source manifest SHA is
  `20a472485cd04f3b6284c0a8d8e7ab01b5ff1f09154d8b5ab4479a313da8ad3b`.

## Model input and inference authorization

- A create-once PRIVATE model-input ZIP with 16 images + text fields
  only (NO source/goal bboxes, hidden target DOM or oracle pointer)
  has SHA `01aeb6d2a95d906e7b7c80087cdf8183a5aeb128e8c83bb8db2b6083061802ba`.
  Its redacted `manifest.json` SHA:
  `33bfa47da0f3d30e995d19dafdaa16f457bc833ce168bf05f1114b523ff2d438`.
- The preexisting H140 V2 research LoRA (external Qwen3.5-2B, **not**
  main Hala 35B) has independently checked adapter model SHA
  `9595e463f2b621e308beaaefffd9ade8c38210bdc59550cb87d8e5f66d2a4bd0`.
- The data and adapter files were uploaded **privately** to the
  owner Kaggle dataset `abdulazizshehri/hala-h141-private-visible-probe-v1`.
  Kaggle automatically **extracts** ZIP files, so the exact screenshot
  PNG and adapter weight SHA checks work on either archive or extracted
  directories. No H140 sealed BLIND images or owner labels were uploaded.
- Private free-only Nvidia T4 inference kernel:
  `abdulazizshehri/hala-h141-private-visible-original-2b-probe-t4`.
  The kernel needs exactly one private evaluation media set; requires
  pinned Qwen3.5 base revision and pinned 2B adapter. It performs
  **32 real inference calls** (16 START, 16 visible-heading), **0 SGD
  optimizer steps**. Its model-source checkout is pinned to original
  H141 branch commit `783bf405812b0bdbfe0a9b1b1c840734044d1c23`.
- Earlier failed kernel workers ended with **zero model inference calls**
  because the Kaggle mount folder was not the expected source archive;
  failures are archived privately and **are not scores**.

## Owner-only actual task evaluation

`h141/evaluate_visible_neural.py` will accept only the private, complete,
SHA-pinned model predictions from the NEW 16-case cohort. It locks
one-time evaluation, reconstructs the exact original 960x720 screenshot
in an isolated local browser, then supplies only the model-predicted
START point and model-predicted visible heading point to the actuator.
The policy itself never receives true DOM source or hidden destination
geometry. The owner checks true localStorage task status **after** the
action and confirms persistence after reload.

Report separably:
- Predicted START point inside visible original card bbox.
- Predicted heading point inside visibly labeled destination column bbox.
- Both points correct on the same source case.
- The **actual original browser task moved to Doing and survived reload**.

A success in this one original React app would be a new H141 local
diagnostic, **NOT** independent generalization to the two sealed H140
OOD apps, and **NOT** evidence of Hala 35B improvement. No model
promotion, merge to main, release, paid GPU, or public weights.

Current real model outcomes must be read from the private Kaggle
receipt and the owner-only Windows result; until those are present
do not invent a success percentage.
