# H122: original TRAIN absolute END / spatial cell fine-point research

## Motivation and decision
H121 owner-neural raw output analysis verified signed displacement collapse:
H119 repeated one vector in **47/64** unseen attempts (12 unique vectors);
H117 repeated one vector in ~59% (16 unique vectors). On H120
**H114, H117, H119 each had 0/64 correct complete drags** after the
independent unchanged H61 strict scorer replay. H119 1024-step loss did
not translate to actual GUI improvement.

H122 is a **different spatial target objective**, not a retuned signed
delta. Predict only original screen's **absolute END**, with two
self-consistent attributes: a conceptual 10x10 coarse image cell and
normalized [0,1000] point, both derived from H113 original source point.
The original screenshot is not resized/modified and the source H71
training media SHA and authentic END bbox must still match. Output
`{"role":"end","cell":[cx,cy],"point":[x,y],
"coordinate_space":"normalized_0_1000"}` with exact keys.
The cell MUST contain the precise point. Invalid JSON or off-cell
output is refused rather than repaired.

## Disjoint original TRAIN-only scientific design

Use frozen owner-verified H113 **512 original TRAIN screenshot pairs**
and H105 original source plan, checked to SHA
`9a1bebc03c22825a6e0abef54edd2778f29e8144685884b811d95a7d8a7b02c9`.
A fixed image-SHA-based proportional family+language allocation
reserves **64 original TRAIN screenshots** as heldout (16 AR, 48 EN).
The remaining **448 original TRAIN screenshots** supply exactly **448
fresh optimizer updates**, 112 AR and 336 EN, with all five families.
No heldout TRAIN image may enter the optimizer; no original PUBLIC
EVAL/TEST labels, H118 or H120 task answers, nor raw neural attempts
may form training targets or prompt examples. No EVAL/TEST samples
are used to tune H122. The holdout remains TRAIN-split but optimizer-
disjoint and is NOT equivalent to a fresh official benchmark.

Freeze before GPU:
- Private optimizer-target SHA:
  `8f5e51a1fd0e06f2c22a30715cb29d5785296733b7401fd3c6807e33c1f376ac`
- Private heldout-source SHA:
  `53e85c5e9070b8c1c2f6cce91b30a5e5c6d46505551a30b0090aa52cb90cd074`
- Private split schedule SHA:
  `f986eeb5ca0c242a1d000ebe4474dc429abdc54514b20dd9e82e1537cbd57ba2`

The original H106/H114/H117/H119 default trainer paths remain
unchanged. `end_grid_sft_path` is opt-in; requires Kaggle free
noncommercial T4, exact H105 original TRAIN, fresh rank8 FP16 vision+
language LoRA, 448 steps, no adapter warm-start, source-point/grid
token-weighted loss, frozen source-bbox proof and 64-step checkpoint
interval (7 complete snapshots). No local model inference, no paid
resource. D1/PStack/e2e if available only as optional local review,
never a gating claim if absent.

## Release and evaluation gates

1. Owner-private original 448/64 data SHA and optimizer exclusion
   verification.
2. Real GPU 448 optimizer and positive gradients with untouched
   H71 original source; exact binary adapter+checkpoint+optimizer SHA
   receipt and independent Windows readback.
3. **Before touching new official PUBLIC EVAL**: model must perform
   heldout-TRAIN-only original 64 END-point inference with source
   digest, prompt SHA, model raw bytes, target box success, per-family
   balanced performance, cell/point consistency, and output-vector
   diversity. The new `h122/heldout_gate.py` preregisters strict
   thresholds *before model inference*: at least **8/64 END hits**,
   **48/64 native-valid spatial JSON outputs**, **24 unique valid
   locations**, modal location frequency **at most 25%**, and hits
   in **at least three of five** original TRAIN families. These are
   internal exploratory qualification limits, not external benchmark
   significance or clinical/general GUI skill proof. All 64 model
   prompt/image/raw-response SHAs must match; EVAL/TEST labels may
   never be used to choose checkpoints. Any failed gate stops before
   fresh public EVAL; no bypass.
4. Only if heldout gate passes: evaluate new original official DragOn
   EVAL indices 108..123 per each four domains with unchanged H61 scorer,
   matched H114 and frozen H119 baselines, separate model-produced
   START anchor; owner local replay. Do not select later samples after
   seeing failures.

**Currently no H122 model skill, improved GUI performance, or release
claim.** No weights pushed to public GitHub/Hugging Face, no merging,
no deployment, no paid GPU.
