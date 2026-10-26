# H139 — Real Multi-Application Generalization Recovery

## Why H139 exists

The independent H138 full-application test **failed**: frozen external
pretrained GUI Assist Grounder 2B and the H136 64-step LoRA each completed
**0/8** real persisted React card moves. START dropped from 6/8 in the
external pretrained model to 0/8 after LoRA; END was 0/8 in both.
These eight H138 screenshots must never enter H139 training or model
selection. H137's 36 external-library screenshots and all H136 sealed
and DEV pictures are similarly out of scope.

## New legally qualified source applications

Three exact original, untouched upstream repos, all with actual **MIT
LICENSE** files and preserved creator copyright:

- TRAIN/DEV: CodeWithSindhu `kanban-board-js` (modular vanilla JS
  HTML5 DnD), pinned at
  `82325355d168a9f4dba9f683899a0a6991034111`.
- TRAIN/DEV: Arjun Vashishtha `todo-drag-drop` (separate vanilla
  HTML5 implementation), pinned at
  `a4426599cb3778c8ee3d7c6b1fd27eed76bf5fe9`.
- Independent unseen **BLIND TEST ONLY**: CodeWithGionatha-Labs
  `dnd-kanban-board` (React DnD HTML5 backend), pinned at
  `7ce11ad463b01a7b12b105417453d5251eabcd51`.

Local authorized Windows hosts all repos and binaries in
`private-h139/upstream/`, NOT GitHub's Hala repo.
No model, datasets, or apps are published. Source fidelity is checked
with git SHA, tracked-source diff and explicit MIT copyright notices.
Repositories relying on unconfigured Firebase, or merely asserting an
MIT license without a license file, are not admitted.

## Experiment architecture

Owner creates fresh browser screenshots of these *complete native apps*
with deterministic localStorage task seed, then actually drags the
reference task and requires a persisted cross-column state change.
Training data never includes DOM coordinates or gold targets as part of
model input. Labels used for SFT come from genuine original app
draggable and destination bounding boxes. Arabic instructions
are external to the unchanged English application interface.

- Real-app TRAIN: 64 (32 per training app)
- Real-app DEV: 16 (8 per training app, new task IDs and cases)
- Independent out-of-repository blind React-DnD heldout: 8
- 960x720 original screenshots, English/Arabic balanced per app.
- Image SHA and source repo SHA checked; no source image overlap across
  splits or with H134-H138 historical private data.
- One owner-private FREE Kaggle T4 run, no paid fallback.
- Frozen external GUI Assist Grounder Qwen3.5-2B base; H139 trains a
  **NEW separately recorded rank8 64-step research LoRA from frozen base**,
  rather than pretending the H136 adapter was Hala flagship 35B.
- Compare genuine BEFORE/AFTER DEV neural responses. Final unseen-app
  score is one-time only with frozen adapter and native response replay.
  Primary measure: real persisted browser action success, not point hit
  or training loss alone.

Do not tune, retrain or select checkpoints from the H139 independent
React blind split after it has been opened. A failed blind test is
recorded as failure and requires newly preregistered application sources.
No release, deployment, merges or expensive services. Hala flagship
35B unchanged; H127 independent human assessment on actual H71 evidence
remains open.

## Privacy-preserving blind inference transport

`h139.blind_export` can prepare a local, owner-private ZIP containing
only eight frozen React blind screenshot PNGs and the external bilingual
instructions required for model inference. It deliberately excludes
gold START/END rectangles, original task IDs, seeded state and oracle
pointer geometry. Before exporting it audits all three H139 splits and
optionally the previous historic manifest SHA exclusions.

The output stays beneath owner `private-h139/` and is never included in
this public Git repository or the Kaggle TRAIN/DEV kernel. Do not mount
the ZIP in an optimizer kernel; it may only be used in a **separate**
inference-only evaluation after a checkpoint and its SHA have been
frozen. Real-world success must then be scored on owner Windows by
replaying the model's native points in the original unchanged React app.
