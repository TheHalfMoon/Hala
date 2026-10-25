# H120 — next-64 real strict DragOn evaluation (preregistered)

This stacked draft **does not train, load, or evaluate a neural model**.
It freezes the next 16 original PUBLIC EVAL scenes, indices 92–107,
from each of slider, sheet, slide_resize, text_highlight. The actual
source listing was owner-checked to contain **250 images per domain**,
enough for the selected indices.

## Strict independent experiment
Only run after H119's fresh *actual* 1024-step optimizer is independently
replayed against owner-private real adapter, optimizer, checkpoint binary
SHA. Compare frozen qualified H114 / H117 / H119 on precisely the same
64 original screenshots; only model-produced START anchors are permitted.
No oracle start/end coords or task bounding boxes in neural prompts.
Neither H118 index 76–91 nor H120 index 92–107 may be used to train.
Use unchanged H61 `score`, per-domain hits, exact raw neural SHA,
model inference count and two-sided paired exact McNemar. No optional
post-hoc replacement of failing scenes. No model skill claim based on
loss or native JSON validity.

The frozen CPU-only `protocol.py` selects source scenes and compares
strict scores. The `eval.py` GPU evaluator is implemented for three
private checkpoint arms with genuine neural START and END predictions,
untouched original H61 scoring, private raw output and source/prompt
SHA evidence. The private Kaggle `kaggle_private_run.py` is prepared but
**MUST NOT be submitted** while H119 adapter/checkpoint/optimizer SHA
pins in `eval.py` are `None`. An owner Windows replay program is also
prepared outside the GitHub repo; no real H120 neural inference or owner
replay has yet occurred. Raw media, checkpoints, target boxes and neural
outputs stay owner-private.
No paid resources, public model publication, merge or release.
