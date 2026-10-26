# H135 — owner-original visual-resolution ablation after H134 negative sealed trial

H134 trained a genuine language-only LoRA for 24 steps, but the once-only sealed
validation found **zero START-handle hits** and **zero full GUI task successes**
both before and after training. The adapter improved sealed median START
distance to handle boundary from 65.85 px to 33.55 px, suggesting imperfect
fine visual alignment rather than missing JSON-coordinate format. END bbox
accuracy improved 3/16 to 5/16, but full task remained 0/16.

H134 resized source screenshots to 640x480 while the original pretrained
GUI Assist Grounder model card describes its own aspect-preserving
smart_resize visual preprocessing. H135 isolates this mismatch **before
modifying weights again**. Neither pilot nor sealed H135 screenshots were used
for H134 training, evaluation, or model selection.

## Owner original desktop-style task fixture

H135 is an independently authored offline desktop UI (sidebar + task pane,
four color themes, randomized locations and distractors), with **six** GUI
interaction families: timeline knob, panel grip, data grid divider, card to
zone, audio slider, schedule card drop.

- PILOT12: IDs 410000-410011. Owner SHA-qualified real Chromium screens,
  prompts and true START/END target boxes. Both original and H134 adapter
  are frozen; compare exact same images in two preprocessing modes.
- SEALED18: IDs 620000-620017. Captured and frozen before the pilot model
  inference. NOT attached to any model preprocessing pilot or optimizer.
- Every owner synthetic scene passed actual Chromium pointer drag with true
  target box centers; true boxes used ONLY for owner scoring, not model
  prompts or model action policy.

## Predeclared GPU research A/B experiment

- Free PRIVATE Kaggle NVIDIA T4 exclusively.
- Frozen external pretrained GUI Assist Grounder Qwen3.5-2B model
  revision 31632edaa20749c7f532868a42635f7240033106.
- Frozen H134 rank8 LoRA adapter SHA256
  711f9dc76cde160818625c9d154c299ad1e098ae48bd554277634ef4b9defdad;
  **0 training steps** in H135, no changes to weights.
- Exact two image modes: 640x480 manual downsampling as in H134 or
  960x720 native screenshot into pretrained processor smart resizing.
- START/END independently requested using the unchanged published
  point_2d 0-1000 model-native JSON prompt, no correct coordinate leakage.
- Two arms × two modes × PILOT12 × START/END = **96 real neural calls**.
  Score native valid response, target-box hits, median error, and actual
  Chromium full tasks per frozen arm/mode. No oracle action substitution.
- After pilot model selection, run frozen selected treatment ONCE on the
  SEALED18 screen split, without feedback-driven tuning.
- Because the new scenes are synthetic and sample small, any pilot or sealed
  success is research evidence, not qualified real-world or Hala 35B accuracy.

No H71 originals, paid GPU, publication, merge, or release. Keep external
model credits/rights; owner synthetic originals and private raw model
responses stay off GitHub. H127 blinded real-H71 visual review remains open.

## Owner PILOT12 real measured result — 2026-10-26

96 actual neural START/END calls succeeded on free owner-private Kaggle T4.
Owner independently checked immutable source manifest, frozen model/adaptor
identities, response native JSON, and private raw file SHA; actual Chromium
replay used **48 neural-generated drags**, without oracle coordinate fallback.

| Model | Image | START hits /12 | END hits /12 | Real full tasks /12 | Median START bbox error |
| --- | --- | ---: | ---: | ---: | ---: |
| Pretrained external specialist 2B | manual 640x480 | 0 | 0 | 0 | 152.85px |
| Pretrained external specialist 2B | native 960x720 | 2 | 5 | 1 | 78.72px |
| Frozen H134 LoRA research adapter | manual 640x480 | 0 | 1 | 0 | 75.90px |
| Frozen H134 LoRA research adapter | native 960x720 | 1 | 7 | 1 | 42.00px |

Predeclared pilot selection: maximize (actual GUI successes, START hits,
negative median START pixel error, native tie-break), using frozen **H134
adapter** on the same 12 pilot scenes. Selected `native_960`, frozen
owner decision artifact SHA256:
`2574aa16e2a4d112793859852fbe5027f9cb15c2699cdb1712946762dc5857fa`.

Result is a **positive within-pilot image-processing signal** for native
pretrained processor input, **NOT a demonstrated Hala skill improvement**:
the pretrained and trained external 2B arms each succeeded on just 1/12
pilot tasks at native resolution. Exact paired neural full-task discordance
test for image modes: p=1.0 for each arm, tiny n12. Native resolution is
a frozen candidate, not a generalization claim. No training took place.

Original H135 SEALED18 manifest was frozen *before* pilot output:
`5dacbcd0dbaaca0f976d0cc342328da9f22a0adf12f3f67401fe88489914ab91`.
Separate once-only free owner-private sealed evaluation will use ONLY these
18 untouched images and the already frozen H134 adapter, without training or
mounting the pilot dataset.
