# H366 — Overflow-safe native Qwen visual grid integrity (Draft / NO TRAIN)

This change is based on the exact independently qualified H365 integration commit `5e6d643ae9dc63792d25454672163a8818578a34` and makes no changes to `main`, no model weights, no blind OOD claims, and no training admission.

## Verified underlying defect and patch

H365 validates each processor image grid by `int(grid[0].prod().item()) == int(raw_pixels.shape[0])`. But signed int64 `torch.prod()` can overflow. A **strictly positive signed-int64** fabricated image grid `[7, 7905747460161236784, 1]` multiplies to **2640 modulo 2^64** even though its true product is astronomically greater than 2640. This is a concrete false-accept for the old volume equality check on a 2640-visual-patch source row. The witness is constructed as `b = (2640 * pow(7, -1, 2**64)) % 2**64` and independently checked in source-only tests.

H366 derives the processor's three positive integer grid dimensions as Python integers, rejects any dimension larger than actual patch rows, checks every partial product is bounded by actual rows, then requires exact volume equality. No tensor integer multiplication can wrap. It also refuses mixed image `pixel_values` dtypes and mixed grid dtypes across a batch, avoiding silent `torch.cat` dtype promotion.

Unit tests cover exact valid volume, malformed shapes and types, wrong volume, zero/negative values, boolean masquerading as an integer, and the explicit signed-int64 overflow witness. The opt-in real pinned offline CPU processor runner now additionally forges an overflowed grid that would have fooled the old guard and a mixed-dtype second source row; both must be refused. Source/original screenshot and label files remain ephemeral; the console receipt contains only aggregate counts and boolean refusal evidence.

## Boundaries

The original H365 valid EN click + AR type_text processor batch must still pass with its original 2 image grids, 5,280 total visual patches, assistant-only supervised labels, right-padding and masks. This validates model **input data plumbing only**. It does not run model forward passes, GPU, LoRA, training, model evaluation, independent blind OOD, or legal admission of H357/H358 images. H199 official TRAIN allowlist remains empty; qualified independent rights reviewer Issue #384 and fresh source-disjoint bilingual OOD custodian Issue #385 are unresolved. Do not merge/release until all standing gates are met.

## Evidence

- Parent H365 #395 exact HEAD: 60/60 GitHub checks and 4/4 dedicated browser/CPU-label cross-platform checks, already separately qualified (not proof of H366).
- H366 source-only Python tests: **21/21 PASS on owner Python 3.11 and 3.12** (also Python 3.14); `git diff --check` PASS.
- H366 actual pinned, offline, real `Qwen3VLProcessor` CPU-only original browser run: **PASS**, complete SHA audit of all 192 synthetic source candidate rows, 2 image grids, 5,280 visual patch rows, 39 supervised assistant-only tokens, 12 correctly masked PAD tokens; both existing grid/patch mismatch guards and both new **`wrapped_int64_grid_refused=true`** and **`mixed_patch_dtype_refused=true`**. `model_forward_calls=0`, `optimizer_steps=0`, no weights or GPU. The same real run checked the old tensor-product oracle would falsely accept the forged grid, but the new collator refused it.
- H366 **exact-head GitHub CI** must be checked separately after the draft PR is opened. Parent H365 checks do not qualify the H366 head.
