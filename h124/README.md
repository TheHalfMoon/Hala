# H124 — Joint START and END grounding, entirely new TRAIN sources

This is a **source-only research stage**, not a trained model or a release.

H123 genuinely trained 896/896 optimizer steps (owner binary SHA replay
19/19 PASS), but its separate 128 original TRAIN-heldout evaluations
strictly scored **3/128 END target-box hits** (1 sheet, 0 slide_resize,
1 slider, 1 text_highlight), **124/128 valid spatial JSON**. Owner
full original H71 reconstitution and neural replay 12/12 PASS.
The pre-registered H123 >=32 END hits with >=4/domain FAILED.

H123 aggregate failure analysis reports correct 10x10 END cell
7/32 sheet, 0/32 resize, 6/32 slider, 4/32 text_highlight.
The original true target rectangles are very small (~6–11 pixels
median width/height) and valid JSON alone is not GUI localization.
Do not reuse any H123 heldout outcomes as H124 training targets.

## New hypothesis

Earlier oracle START conditioning (H117/H119) had train/inference
anchor distribution mismatch. H123 END-only supervision lacked
joint START/END geometry. H124 therefore trains one model-generated
JSON response containing genuine normalized **START, END, START cell,
END cell and signed delta**. At inference the model must predict
both points; **no oracle START or END is supplied**. This is a
research hypothesis, not evidence of better skill.

## Source governance

1. Re-derive and verify original H105 first 512 source SHA.
2. Re-derive exactly SHA-frozen H123 896 TRAIN +128 heldout sources.
3. Exclude **all 1,536 previous original screenshots** by BOTH
   original source ID and actual media SHA.
4. From new H71 original TRAIN-only DragOn screenshots choose
   256 per complex family; 32/domain remain unseen heldout,
   224/domain are for potential new training (896/128).
5. Verify each of the 1,024 authentic original image bytes and
   both genuine START and END source bounding rectangles.
6. A strict parser checks both 10x10 grid cells and
   signed delta = END - START without modifying model outputs.
7. Freeze real-source selection SHA and a *new* heldout evaluation
   rubric before any GPU work.

No original EVAL/TEST, paid compute, hidden scoring, public
images/targets/model weights, merge or release.
H124 is English-only complex-GUI exploratory research, not a
bilingual/general GUI performance claim.
