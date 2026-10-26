# H129 — frozen pretrained cursor-marker feedback experiment

**Exploratory frozen-model inference. No Hala model training or claimed
improvement. Static visual-feedback simulation, NOT real desktop interaction.**

H125 and H128 both failed strict source-bbox full-drag qualification.
H128's model-anchored crop did not improve 0/16 strict full drags. H127
independent human visual source-identifiability reviews remain outstanding.

H129 is a fresh, one-step visual corrective experiment on a source-disjoint
original H71 TRAIN-only 16-scene set, 4 per original GUI domain.

## Preregistered comparison

- Frozen base: ShowUI-2B revision
  `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`, no fine tuning.
- Exclude ALL 2,704 distinct original screenshot IDs **and SHA256 media
  hashes** previously used or withheld by H105/H123/H124/H125/H126/H128,
  regardless of earlier TRAIN/holdout status.
- Baseline: one native point for START and END from the original screenshot
  and exact model-native baseline prompt.
- Visual feedback: overlay a magenta ring/crosshair at **that first model
  prediction**, on a **copy** of the original screen. Explain explicitly
  that it is a prior guess, not ground truth; re-infer the SAME role and
  translate the native relative model output into original pixels.
- Compare on same 16 original scenes: strict START/END source bbox hits,
  simultaneous full-drag hits, coarse-screen quadrant and change counts.
  16 scenes × 2 roles × 2 passes = **64 real model calls** on free
  owner-private Kaggle T4, with zero optimizer steps.
- Ground-truth rectangles are available **only for later scoring**.
  Neither overlay, crop, model instruction nor revised inference receives
  target source boxes or correct pixel coordinates.
- Important: no interaction with a live UI and no actual computer-control
  trajectory. Do NOT describe this as demonstrated interactive GUI skill.
- This is one exploratory 16-case paired test, not a powered estimate or
  a reason to claim Hala's general model accuracy improved. Negative results
  are retained. H127 human-source-identifiability remains a separate gate.

## Reproducibility

Private Kaggle kernel configuration: `kernel-metadata.json`, free T4 only,
source SHA pinned, pinned pretrained model and runtime. No paid fallbacks.

`python -m unittest discover -s tests -p 'test_h129_cursor_feedback.py' -v`

No public EVAL/TEST, prior withheld reuse, images/labels/raw model outputs
in GitHub, merged changes, deployments or model release.
