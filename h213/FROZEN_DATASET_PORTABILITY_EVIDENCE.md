# H213 — bitwise cross-OS reproducibility gate for original H212 GUI research candidates

**Strictly source-only and draft-only.** Stacked exactly on H212 #233
head `5f602734d2bcde18fe191cc53e36badc05b9582f`.
Do not merge to protected main or publish any model.

## Concrete training-relevant risk addressed

H212 generated 96 self-authored bilingual 960×720 GUI images with
fixed Pillow 12.3.0, versioned Python source and 96 pinned PNG plus
96 decoded RGBA SHA-256 values in
`h212/RESEARCH_CANDIDATE_METADATA_NOT_ADMITTED.json`.
But H212's CI only checked SHA uniqueness, not that the **actual
committed hash values** regenerate identically on other operating
systems. A Kaggle/Linux or Mac worker could silently create
different pixels/labels from the Windows-frozen source while all
source-level uniqueness tests still passed.

H213 adds `h213/frozen_candidate_reproduction_gate.py`:
- Opens **one fixed checked-in nonprivate H212 metadata file**;
  refuses a missing file, symlink, oversized JSON or unexpected schema.
- Regenerates **all 96 actual PNG byte streams and decoded RGBA
  byte fingerprints** directly from the immutable H212 source.
- Reconstructs the expected canonical research metadata, including
  generator-source SHA, 96 exact ordered image hash pairs, index,
  source family, research role and English/Arabic language.
- Rejects any byte/role/language/source drift, duplicate JSON fields,
  NaN/Infinity, corrupt counts, extra keys, reordered rows or changed
  `official_admitted`, `dev_gold_sealed` or OOD research status.
  Allows the normal CRLF-vs-LF text checkout difference while requiring
  identical canonical content. Never reads private media or labels.
- Refuses unexpected H199 official TRAIN admission, **does not
  change** `ADMITTED_TRAIN_MANIFEST_SHAS=frozenset()`, and cannot
  create a valid training-release signature by itself.
- Its stdout gives aggregate 96/96 pass or a redacted failure; it
  does not export screenshots, SHA strings or target action labels.

## Actual owner-device evidence

**Windows 11 owner CPU, Python 3.11.15, Pillow 12.3.0**:
- H213's 13 adversarial source-only tests pass in normal mode (13/13)
  and `python -O` (13/13): **26 passed total**.
- Invoking the actual verifier in both modes reports:
  `H213_ALL_96_GUI_RESEARCH_FINGERPRINTS_REGENERATED_IDENTICAL
  cases=96 png=96 rgba=96 official_train=0 dev_sealed=0
  official_ood=0 private_media=0 model_calls=0 optimizer_steps=0`.
- `compileall` succeeds. All checks use isolated disposable
  `D:\Hala-H213-QA` (Pillow and PyYAML only); no PyTorch/weights.

**Independent Apple M2 macOS host**, source exact H212 head
`5f602734d2bcde18fe191cc53e36badc05b9582f`, Python 3.13.5,
isolated Pillow 12.3.0:
- Independently regenerated all **96** original PNG and RGBA digests
  and compared each against the H212 Windows-frozen metadata.
- Observed exact output:
  `H213_CROSS_OS_H212_FROZEN_HASH_PARITY cases=96
  png_mismatches=0 rgba_mismatches=0`.
- This is a second-platform *observed* match, not proof of source
  independence from old images or model generalization.

## New hosted CI matrix

`.github/workflows/h213-cross-platform-frozen-source-cpu.yml`
provides a six-way matrix on **ubuntu-24.04, windows-2025, macos-15**
and **CPython 3.11/3.12**, using only pinned Pillow 12.3.0:
each job runs all 13 negative/positive tests in normal and optimized
mode and regenerates all 96 PNG+RGBA hashes.
This CI does not download neural weights, perform GPU/model inference,
train a model or access any owner data.
**Hosted success must be explicitly verified at H213's pushed commit.**

## Hard scientific limits

1. **All 96 images remain research candidates, NOT official TRAIN**,
   nor sealed DEV or true independent OOD. They are public code and
   reproducible labels; a blind DEV release cannot be inferred.
2. Cross-OS matching proves *deterministic raster source at the tested
   Pillow versions*, not historical source/fork/old-RGBA
   non-overlap. H204/H206's older PNG/RGBA provenance blocker
   remains unresolved. The 20 owner original-GUI prospects remain
   quarantined and unadmitted.
3. Zero Hala optimizer steps, zero new model calls or native browser
   events. H211's public 256M real pretrained GUI response was
   **0/2 strict JSON clicks accepted**, and no better model has yet
   been trained or evaluated.
4. Never inspect or reuse consumed H173 OOD, reserved Gantt OOD,
   private owner screenshots or sealed gold; no paid compute,
   main merge, publication or model release.
