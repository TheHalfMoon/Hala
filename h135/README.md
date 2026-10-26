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
