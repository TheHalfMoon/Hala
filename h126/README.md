# H126 — new original GUI source language/geometry sufficiency pilot

**CPU source audit only. NOT a trained model, benchmark score, human visual
annotation or proof of task identifiability.**

## Why H126

H125 real separate UGround-V1-2B and ShowUI-2B inference generated 128
START/END points per model across identical 64 original DragOn scenes.
All outputs native-valid, but strict original full drags **0/64 for both**.
Independent owner Windows byte-exact source and neural replay **24/24 PASS**.
Additional owner-private aggregate diagnostic showed real localization
errors on many cases extend beyond the tiny ~6–14-pixel source target
boxes and often miss the coarse 4x4 quadrant. So simply loosening bbox
tolerance or training longer on the same small 2B backbone would not
address the observed information/grounding mismatch.

## Qualification and planned analysis

- Derive frozen genuine H105 first512, H123 full1024,
  H124 full1024 and H125 64 original TRAIN source selections.
- Exclude **all 2,624 old screenshot IDs AND media SHAs**, including
  all prior TRAIN and heldout images.
- Select **64 NEW original H71 TRAIN-only screenshots**, 16/domain
  sheet, slide_resize, slider and text_highlight, with source-supported
  spatial/directional balance.
- Verify each selected original screenshot actual byte SHA and media
  dimensions, original task instruction plus both source START and END
  ground-truth rectangles, privately.
- Compute only descriptive **aggregate source-text** features:
  instruction length, lexical presence of direction/numeral/source
  referent/deictic target words; original tiny START/END bbox pixel
  dimensions and fraction of screenshot area. Do not publish original
  per-scene annotations or raw task text.
- **Important scientific limit:** lexical features alone CANNOT tell
  whether humans/models can visually identify the desired exact START
  and END. Establishing true semantic identifiability requires
  independent blind human/visual reference annotation on fresh scenes
  BEFORE any further model adaptation. This stage merely qualifies
  those source scenes and quantifies task specification.
- No zero-shot GPU, LoRA, new optimizer, official public EVAL/TEST,
  hidden test reuse, paid compute, deployment or release.

Source-only test gates are fail closed. Old failed model evaluations stay
frozen in H123/H124/H125. No adaptation on any prior heldout.
