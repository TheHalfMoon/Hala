# H251 — original complete Arabic RTL and English LTR app native replay

**SOURCE-NATIVE PRIVATE RESEARCH ONLY: NOT TRAIN/DEV/OOD, model inference,
GPU LoRA training, held-out benchmark or model improvement.**

## Why H251 is materially new

H250 successfully entered Arabic text inside an ORIGINAL English interface;
that was not native Arabic GUI grounding. H251 uses
[3mmar404/LIfe-Os-Free](https://github.com/3mmar404/LIfe-Os-Free),
an independently authored complete multi-module application with a separately
pinned MIT source LICENSE, original Arabic HTML lang=ar dir=rtl and original
English HTML lang=en dir=ltr. Original upstream source commit:
b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c.

The H251 preflight demands a clean unchanged original external checkout,
pinned exact commit and Git remote, and SHA256 of original LICENSE, two HTML
pages, CSS, core/UI scripts, two bookmarks modules and two documentation
modules. H238 minimum public prior-source floor is checked. This does NOT
prove complete past-forks/npm/deleted-source or old PNG/RGBA history.

## Real source-native browser execution

A new isolated native Edge context per language has strict Playwright offline
mode before navigation, external HTTP(S)/WebSocket request abort and service
workers disabled. This is defense in depth, not an OS-level proof.

Original app requires a password-protected local vault. Bootstrap uses only
a fresh RANDOM synthetic fictional secret in memory via original UI, never
written to logs, screenshot, public Git or user account; bootstrap is NOT a
Hala supervised model action. One fictional link is added through each
ORIGINAL Arabic/English UI, not via hidden DOM state or authored fixtures.
The Arabic and English original bookmark-add forms are DIFFERENT; H251
respects both unchanged native implementations.

For each language, five source-native one-action transitions are captured:
1. Actual click on original Bookmarks nav button.
2. Actual screenshot-coordinate click focusing original search input.
3. Typed Arabic/English query via actual keyboard.insert_text; assert native
   query and filtered original bookmark text.
4. Actual screenshot-coordinate click on original Documentation nav button.
5. Real mouse wheel scrolling of native documentation; assert window.scrollY.

Click points are H248 elementFromPoint hit-tested 960x720 screenshot pixels,
then truly dispatched via page.mouse.click at exactly those same pixels.
Each transition uses H199/H249 exact-one-action JSON. No compound actions,
selectors, oracle labels or browser DOM are ever passed to a model.

## Owner Windows verified 2026-10-30

- 2/2 original multi-module native app journeys PASS: genuine AR RTL + EN LTR.
- 10 real source-native one-action steps, and 12 original browser PNG frames.
- Original bookmark-add, field focus, text search, doc navigation and real
  scrolling state checks pass. No INCOMPLETE marker in successful output.
- Private images, pixel references, per-frame PNG and decoded RGBA SHA,
  state oracles and owner-only receipt remain outside ALL Git at:
  a new create-only owner-private directory on Windows D: outside Git
- 0 model calls, 0 optimizer steps, 0 admitted TRAIN/DEV/OOD, no paid resources.

## Scientific blockers remain absolute

A separate MIT license covers the upstream application source, but does NOT
establish rights for every third-party font, official logo, remote icon or
transitive visual asset. Old source/fork/npm/image/RGBA histories have not
been completely certified. H199 ADMITTED_TRAIN_MANIFEST_SHAS remains empty.
H173 consumed 32-case scored OOD stays sealed. H251 source research cannot
be promoted by flipping a manifest flag, and no test here qualifies a model.
Even the two languages are one original app FAMILY, not independently held-out
model evaluation.

Highest-impact next work: legally verified multi-family source/history and
asset rights, independent frozen TRAIN/DEV, new untouched blind OOD, then
actual free noncommercial Qwen3-VL processor+LoRA optimization and strict
blind model evaluation. Do not equate CPU UI oracle pass with ML progress.

## How to reproduce (owner Windows only)

Run this code from Hala checkout with clean pinned original source and
a NEW, absent owner-private output path:

    python -m h251.original_lifeos_rtl --source-root CLEAN_ORIGINAL_SOURCE_ABSOLUTE_PATH --output NEW_PRIVATE_ABSENT_OUTPUT_PATH
    python -m pytest -q tests/test_h251_original_arabic_rtl.py

Never upload original app code, original screenshot pixels, private labels,
private frame hashes or credentials to the public Hala repository.

## Verified public CPU source compatibility (owner Windows)

Focused H251/H250/H249/H199/H247 tests: **44 PASSED**. H251 on optimized
Python -O: **4 PASSED** (usual pytest warning). Broad frozen source/history/
action/split tests H174/H176/H178/H185/H198/H199/H204/H236/H238/H241/
H242/H243/H244/H245/H246/H247/H249/H250/H251: **381 PASSED,
6 SKIPPED, 2598 unrelated deselected**, no failures. Fresh D pytest
basetemp worked around Windows C temp ACL issue without disabling tests.
Python compileall and Git whitespace gates passed.
