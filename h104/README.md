# H104 — optional reversible source↔vision-processor coordinate bridge

H103 measured **2 of 8** source TRAIN frames resized internally by the
pinned Qwen3-VL processor: **495×640→480×640** and **853×640→864×640**,
with 16-pixel image patches. The other 6 were unchanged; TRAIN and inference
pixel tensors, prompt masks and digit-token loss weights matched 8/8.
This is plausible native image patch alignment, **not proof that this causes
the previous 0/16 strict GUI success**. Existing output contract explicitly
requires ORIGINAL screenshot pixel coordinates, and must remain unchanged.

## Controlled offline source-bound hypothesis test

An optional, **unused-in-production** reversible pixel-center map is tested:
a source pixel maps to its processed patch-frame pixel center, then an output
processor pixel maps back to the ORIGINAL source frame. Validate mapped
source-box-center drag actions using the **unchanged H61 strict scorer**.
Reject out-of-bounds boxes, collapsed drags or round-trip failures; NEVER
silently clip or mutate frozen H71 raw labels.

- Synthetic unit tests cover both actual H103 resize geometries, identity
  1024×640 public-EVAL geometry, invalid coordinates, and mapping reversibility.
- A private **Kaggle CPU-only** sample of the same **8 original H71 TRAIN
  screenshots** will rederive the source archive SHA and original labels,
  check source image SHA/decoded dimensions, use the *actual pinned
  Qwen3-VL AutoProcessor* to derive patch grid dimensions, and replay the
  pixel-center reversible bridge with the original H61 oracle.
- Raw images, coordinates, tasks, IDs or model weights are NEVER uploaded
  to GitHub; only source-bound code and aggregate receipt. No GPU,
  optimizer, training, hidden TEST, model deployment or publication.
- The **single H102 half-open bbox source exception** in the 10k remains
  unresolved, preserved and not among the actual H99 128 TRAIN samples.

**Passing a round-trip with GROUND TRUTH is a data-contract fact, not
model intelligence.** It must not be presented as improved GUI success or
as a ready-to-enable training intervention. Source→processor conversion
should not be activated in H61/H99 without a separately preregistered,
source-disjoint, matched candidate-versus-control inference evaluation.
