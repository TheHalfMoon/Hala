# H153 — verified one-shot frozen neural model + native controller A/B

**Owner Windows verified 2026-10-27. Draft research, not a model release.**
The same **frozen external H142 Qwen3.5-2B research LoRA** predicted both START
and visible destination-heading points on **16 NEW original MIT React GUI
screenshots**. Those predictions were used unchanged for a **paired native
mouse comparison**, each arm in an isolated fresh Chromium context.

## Source, inference and action admission

- Original owner-only MIT React screenshots, source task IDs >=184000:
  16 unique SHA-256 images, balanced four Arabic/Doing, four Arabic/Done,
  four English/Doing and four English/Done; four browser-only CSS-shifted
  layouts. No upstream donor edits or gold DOM targets in model inputs.
- Frozen source manifest SHA-256:
  `df1357827e1356ca1b69ffea0d44dd56c3d897b7854598f62323e91cb4f078e1`.
  SHA-history audit excluded **1,125 earlier images**; all 16 source
  reference native moves persisted. Exact browser screenshot replay was
  verified before evaluating model-driven actions.
- Kaggle private free Nvidia T4 kernel:
  `abdulazizshehri/hala-h153-private-frozen-2b-controller-ab-t4`,
  completed with **32 actual neural model calls**, **0 optimizer steps**.
- Frozen research adapter SHA-256:
  `992fa18fead6724329dbe7ec7b95c8acb0a56229c1fa5b56719e5f4829130a68`.
- **32 true native drags**, two controllers using the **same model predictions**
  on each of 16 tasks; requested target state checked after full page reload.
  Owner gold boxes used only for post-hoc point scoring, never as action inputs.
- One-shot evaluation lock exists: **do not rerun this consumed cohort**.

## Actual persisted native outcomes

| Metric | Result |
|---|---:|
| Frozen 2B model START point hits visible card | **16/16** |
| Frozen 2B model visible heading point hits reference | **15/16** |
| Both model points inside references | **15/16** |
| **H150 model + prior screenshot vertical-only actuator** | **11/16 (68.75%)** |
| **H152 model + screenshot-only horizontal-region actuator** | **15/16 (93.75%)** |
| H152-only persisted successes | **4** |
| H150-only persisted successes (regressions) | **0** |
| H150 retained screen signal | 11/16 |
| H152 retained screen signal | 15/16 |

| Stratum | Prior H150 | New H152 |
|---|---:|---:|
| Arabic (8) | 6/8 | **8/8** |
| English (8) | 5/8 | **7/8** |
| Doing (8) | 5/8 | **7/8** |
| Done (8) | 6/8 | **8/8** |
| up (5) | 4/5 | 4/5 |
| down (4) | 4/4 | 4/4 |
| far_down (3) | 3/3 | 3/3 |
| left_up (4) | 0/4 | **4/4** |

This is an actual **within-cohort paired +4/16 completed-task gain** when
substituting the H152 native action controller for H150, with identical
model outputs and source GUI cases. The same model weights were used in both
arms: **this is NOT a 2B model weight/training accuracy improvement**.
The one remaining failure cannot establish a general model limitation.

## Guardrails and next evaluation

These findings concern ONE original MIT React app and 16 GUI tasks,
not a broad computer-use benchmark or cross-app transfer claim.
The H140 sealed OOD BLIND cohort was not accessed (0 calls).
No Hala flagship 35B weight update, optimizer steps, paid GPU,
public model checkpoint, repository merge, or product release.

**Next:** preregister a genuinely new multi-app, owner-licensed GUI
holdout with several action types (not only Kanban drag-and-drop), and
assess the frozen integrated model + controller with task persistence.
Do not tune to or reuse H151–H153 consumed test screenshots.

Detailed original task-level results, raw screenshots, model outputs,
GPU receipts and one-shot locks remain owner-private in `private-h153/`,
outside tracked GitHub files.
