# H141 — interactive visibility-grounded computer-use recovery

**Research status: original browser source QA only; no model training and
no new independent blind inference.**

## Verified previous negative result

The frozen H140 external Qwen3.5-2B research LoRA V2 executed **128/128
free Kaggle T4 optimizer steps** on original TRAIN256 (EN/AR balanced)
and reached START/END point-rectangle hits **25/64** on original DEV64,
vs **5/64** for the pinned pretrained external grounder.

H140 V3 executed **160/160 genuine free T4 steps** with additional
balanced END samples and instead regressed to **18/64** on original DEV64.
START fell from 64/64 to 63/64; END fell from 25/64 to 18/64.
React END fell from 5/22 to 3/22. V3 failed the preregistered DEV
admission gate (5 of 7 checks failed). Its adapter was rejected,
**not evaluated on BLIND and not promoted**. Hala flagship 35B
still has zero training or model changes.

## Proven structural reason for unreliable React END targets

The exact unmodified original React MIT source code
`dandemslay/React-KanbanBoard` at
`fa047f7a32afb57750ed5246456dc282ea7befcd`
uses:
- `src/components/DropArea.css`: `.hide_drop { opacity: 0; }`
- `src/components/DropArea.jsx`: a React `onDragEnter` handler
  turns the target from `hide_drop` into visible `drop_area`.
- `TaskColumn.jsx`: several drop areas exist per column, and the
  source allows dropping into different slots within the same status.
- H140 React END gold references a **specific invisible `.hide_drop`
  DOM rectangle** on an at-rest screenshot. The instruction only asks
  to move into the visible "Doing" column, not into that particular
  invisible gap. Thus the old rectangle-hit score can reject otherwise
  plausible visible GUI behavior, and the label is unidentifiable
  to an image-only agent at this phase.

Owner Windows real Chromium evidence, no tracked upstream source edits:
- Before dragging: original target `.hide_drop`, computed opacity **0**,
  original bounding box approx **73.8×18 px**.
- After pressing and dragging a visible source card into the region:
  React `onDragEnter` changes the target to `.drop_area`, computed
  opacity approximately **0.94**, visible box about **253×96 px**.
- Actual card moved from To Do to Doing and remained moved after full
  page reload.

This is a **missing observation / multi-step agent action** problem,
not merely an END sample-weight imbalance. It is not defensible to
retrain repeatedly on frozen H140 DEV or to tune via its sealed OOD BLIND.

## New H141 source-only original interactive diagnostic

`h141/interactive_source.py` captures **NEW original React app scenes**
on fresh post-H140 task IDs from **41000+**. Each owner-private scene
records original screenshot before dragging (target opacity 0),
then a second original screenshot while native drag enters the target
(target opacity > 0.7), and finally confirms the actual native task
transition persisted after full reload.

**Crucial leakage warning:** the second screenshot is captured AFTER
an ORACLE-GUIDED source reference moves the pointer onto the correct
destination. It therefore visibly identifies the answer and is
**NOT** a clean independent END model input. Source QA manifests
explicitly say `training_eligible:false`,
`blind_eligible:false`, `model_evaluation_eligible:false`.
Do NOT recycle it as training or evaluation images without a different
observation-capture design that avoids oracle cursor hints.

## Required next research stage, without compromising BLIND

1. Define task success as **actual persisted column movement** on
   unmodified apps rather than hit-testing a chosen invisible gap.
2. Create a model-driven staged policy: from visible screenshot, predict
   source card and target column, begin drag, approach target without
   oracle DOM coordinates; observe UI changes, then decide to release.
3. Collect fresh post-H140 genuine TRAIN/DEV trajectories with
   independent screenshots and no target-revealing gold cursor in
   the model's visual inputs. Reuse no H140 frozen DEV/BLIND pixel
   data to optimize or pick hyperparameters.
4. Qualify each real action against original source application
   state and reload, then consider another free T4 experimental
   adapter. Only one-shot independent BLIND after final DEV gate passes.
5. Preserve H140 negative result and incomplete accuracy limitations
   in public docs; never modify the H140 evidence to hide the failed V3.

**Constraints:** no paid GPU, no model/cloud release, no merge to main,
no blind exposure, no editing donor sources, and unchanged flagship
Hala 35B.
