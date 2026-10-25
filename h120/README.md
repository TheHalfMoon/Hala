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

H120's current `protocol.py` is the **CPU-only scene selector and
frozen strict-score comparison**. GPU inference and Windows-independent
prompt/media/raw-output replays are future work contingent on training.
Raw media, checkpoints, target boxes, neural outputs stay owner-private.
No paid resources, public model publication, merge or release.
