# H320 — Model-visible RGB freshness for the dynamic GUI

H319 introduced a genuine new dynamic browser interaction: a native click
reveals an input and a second screenshot-conditioned action may type into it.
Its original callback lock compared PNG file bytes. PNG ancillary metadata or
encoding differences can change those bytes without changing a single pixel
visible to Qwen. In addition, the prior flow did not explicitly require the
second screenshot to differ visually from the first.

H320 checks decoded 960x720 RGB pixels in private memory before and after
each callback. A second callback is refused **before any model invocation**
when the new frame is visually identical to the prior model-visible frame.
A genuine visible change is permitted. No screenshot, digest, task gold or
prediction is logged or written. The original HTML source blob and 12 frozen
public/exposed research scenes remain unchanged.

Five invented-only adversarial CPU tests cover PNG metadata differences,
duplicate visual frames across stages, a genuine changed frame, mutation during
prediction, and invalid dimensions. H319's separate real Chromium/Edge source
browser tests remain required; no fake callback is a Qwen model result.

This is a safety/measurement prerequisite, not new pretrained inference,
adaptation, lawful TRAIN admission, independent blind OOD or release approval.
H199 remains empty; H173 and H212 are excluded; H293/H294 safety blocks remain.
