# H118 — preregistered new 64-scene strict GUI experiment

This is an evaluator-only Draft. H117's adapter is NOT admitted until
actual 256 optimizer updates, a private checkpoint and an independent
owner-Windows adapter/checkpoint/optimizer byte-SHA replay are verified.

Prior benchmark: H115 on genuine pinned original DragOn public EVAL
60..75, four domains, 64 unseen scenes. H106 scored 0/64 full drags;
H114 scored 0/64 even though START target hits rose to 15/64.
H116's posthoc diagnostic found an END role-collapse signal:
original TRAIN label separation median 153.5 px; model prediction
separation 6.11 px, with END closer to actual START in 39/64 cases.
These old source labels are diagnostics ONLY and never retrained on.

## Frozen new experiment

Original public EVAL source indices 76–91 in EACH original domain:
sheet, slide_resize, slider, text_highlight. Exactly 64 genuine new
screenshots; no hidden TEST and no re-use of older public EVAL scenes.
No source labels may enter model prompts, conditioning or checkpoints.
All variants use identical source screenshots and original unchanged
H61 native strict scorer:

- H106 original single-pass drag action (64 model calls).
- H114 independently-trained original-image START and END absolute
  normalized points (128 model calls); strict original pixel conversion.
- H117 newly-trained START point first. If and only if this model's
  START is strictly valid, give the SAME original screenshot to its
  signed END delta predictor with the model-generated START anchor.
  END must be an exact native signed normalized displacement; compose
  true source-image pixels without clipping, format repair or oracle.
  Invalid START/END means invalid whole drag. Approximately <=128 calls.

Measure: native valid actions, original START/END box hits, strict
complete drag successes, degenerate drag rate, per-domain results,
paired source comparisons, model-call overhead and owner audit.

H117's training uses a source-verified original TRAIN START anchor as
context. Inference deliberately uses its OWN predicted START, which
may introduce substantial distribution shift. This is a known limitation,
not hidden by an oracle anchor or test-set hand correction.

Fail-closed at source: H117_ADAPTER_SHA/H117_CHECKPOINT_SHA remain unset
until actual owner independently verified H117 new model. No H118
Kaggle kernel should run before verification. If any change to H117
weights, code, data or source indices occurs, require a new committed
and separately identified evaluation attempt.

No merge, deploy, publish source/media/weights, paid services or release.
Lower optimizer loss does not itself count as GUI model improvement.
