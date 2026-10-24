# H109 — Safe source-pixel coarse-to-fine ROI geometry preflight

## Rationale

Owner-independent H107 original 64-case DragOn assessment gave 0/64
native strict complete GUI drags for Qwen base, H99 and H106 despite
H106 generating 63/64 valid source-pixel JSON actions. H106 median
start/end bbox misses were 115.55/170 px and median predicted
displacement 20 px (reference 117 px). **The bottleneck is
visual grounding, not action serialization.**

H108 tests a separate 2×2 visual pixel ruler ablation with original
public EVAL 36–43. The first notebook version failed before inference
because it incorrectly required every aspect-preserved H61 frame to be
exactly 1024×640. Version 2 fixes this; results pending at start of H109.
Its 32 source frames include seven distinct aspect-preserved sizes.

## What this PR does and does NOT do

This H109 phase implements the geometry contract ONLY for an optional
coarse-to-fine crop. It is not an inference result, trained model,
reasoning breakthrough, or evidence of stronger GUI skill.

Given an actual model-proposed source pixel (never ground-truth box), crop
a source-centered **320×320 pixel region**, clamped to the actual image
without changing original source pixels. Enlarge the crop by exactly
**2×**. All crop-frame and source-frame dimensions are recorded as
explicit integer values; mapping source→zoomed→source must round-trip
exactly for integer centers, with no silent coordinate clamping.
Out-of-bounds model predictions, cropped coordinates, frame sizes and
wrong-type values are rejected.

Pure-CPU tests cover **all seven H108 actual source-frame shapes**,
edge-centered ROIs, reversible transforms, source-bbox containment,
incorrect dimensions and nonmutation of the original image. They
specifically do not claim the coarse predicted region covers the
actual target. For a credible next inference experiment, measure
**both region recall before zoom and full strict drag success after
zoom** on completely fresh source scenes, comparing matched no-crop
and crop-refinement controls. Do not use EVAL target boxes to choose
regions, leak target annotations into the model prompt, or relax the
original H61 strict action scorer.

Research motivation: GUI-Spotlight iterative focus refinement,
ICML 2026: https://proceedings.mlr.press/v306/lei26h.html .
This implementation is **not** GUI-Spotlight's original model or
an implementation of its full learned iterative-focus method.

Private owner Windows CPU tests only. No GPU allocated by H109,
no optimizer/weights/sources changed, no paid service, no publication
or deployment. Draft and stacked on H108.
