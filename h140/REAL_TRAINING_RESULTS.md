# H140 original-app model evidence — frozen V2 result and V3 plan

Research record date: 2026-10-27. Owner-private source evidence;
**never release the 35B Hala flagship or promote the 2B adapter from DEV.**

## Immutable original source and training receipts

- TRAIN256 original screenshots, balanced Arabic128/English128, from three independent unmodified MIT Kanban applications.
- DEV64 new SHA-disjoint original screenshots from exactly those three applications (32 Arabic/32 English); ground truth remains owner-private.
- The frozen manifest SHA-256 values:
  - TRAIN: `a59438ca178c3b86da721c164e3ae653566c2001c25bb9208ff42afed0c14f38`
  - DEV: `433f4001481d5db704e3b8b19b47154d5637ffc570533e6223880ecdc172d87a`
- Old H136–H139 originals and prior H140 diagnostics are excluded.
- Two further unmodified independent MIT original applications are sealed in
  a **private 32-example BLIND** set, `6de30cff6ef31b0095c30a73bbe3dff3c4d98fbc72679d1d684f75f50ed6e162`.
  It remains **UNEVALUATED**; no oracle bboxes on Kaggle or GitHub.
- Redacted model-input-only owner ZIP is created and audited,
  `5d12b6435dc61a85bbb261a8211d996fd377f72c49a67bfeba9accd957df4f3d`.
  **Do not upload it to the TRAIN/DEV kernel**.

## Real private FREE Kaggle T4 training — V2

H140 V2 has **actually finished 128/128 GPU optimizer steps** on the
pinned *external* Qwen3.5-2B GUI grounder; rank-8 LoRA, 5,455,872 trainable
parameters, 512 START/END role examples.

Private adapter SHA-256:
`9595e463f2b621e308beaaefffd9ade8c38210bdc59550cb87d8e5f66d2a4bd0`.

These are **single-point localization hits inside original DEV source DOM
bounding rectangles**, NOT completed model-directed browser drag-and-drop
transactions:

| DEV64 outcome | Frozen pretrained 2B | H140 V2 LoRA |
| --- | ---: | ---: |
| START point inside correct task card | 32/64 | 64/64 |
| END point inside correct target | 14/64 | 25/64 |
| BOTH START and END correct on same case | 5/64 | 25/64 |

By original app, **BOTH** correct:
- Vanilla HTML5: 4/22 → 10/22
- React: 0/22 → 5/22
- React + dnd-kit: 1/20 → 10/20

DEV paired full-case change: 21 previously incorrect cases improved, one
previously correct case regressed (exact two-sided McNemar p≈0.000011).
The saved 21.85 MB adapter passed owner-independent SHA verification.

**Critical NO-GO:** V2 fails the preregistered coordinate-collapse guard:
on React **END** only 11/16 first audit predictions are valid, with only
6 unique coordinates and a dominant coordinate count of 6. Seven
React English END responses generated repetitive pseudo-chat JSON text;
although the first JSON object could be parsed, **0/7** of those points
hit the actual owner-private END bounding rectangle. It would be
misleading to repair the decoder and claim restored accuracy.
The DEV result is measurable learning, but NOT proven generalization to
unseen apps. Hala flagship **35B weights remain unchanged**.

## Preregistered one-time V3 attempt — active independent kernel

- Distinct private Kaggle free T4 kernel:
  `abdulazizshehri/hala-h140-v3-private-end-recovery-t4`.
- Separate source-file experiment `h140/train_gpu_v3.py`; **160** planned
  optimizer steps × accumulation 4 = 640 genuine frozen TRAIN role
  examples. Includes every original START and END once, plus 128
  stratified additional END samples: React32 per language, Vanilla16
  per language, dnd-kit16 per language. Original AR/EN role balance
  is preserved.
- Learning rate reduced from 0.000075 to 0.00005. The original source
  model/processor revision, application images, and DEV stay frozen.
- **Qualify V3 on DEV before touching one-time BLIND**, using
  `h140/evaluate_v3_gate.py`. Required: no coordinate-collapse across
  apps/roles, ≥28/64 both correct, ≥60/64 START correct, ≥28/64 END
  correct, ≥8/22 React END correct, no app's full-case success regresses
  below V2, genuine new adapter weights.
- **No claim that these criteria have been achieved until a real V3
  Kaggle receipt and full GPU inference outputs exist.**

If the DEV gate fails, keep BLIND SEALED and record a negative result;
do not repeatedly select hyperparameters using the untouched BLIND
labels. Even if DEV passes and one-shot BLIND is tested, the actual
model-directed browser gesture must be run and independently checked
for persisted success before claiming end-to-end task reliability.

No paid compute, deployment, public training media, model release,
or main Hala 35B update.
