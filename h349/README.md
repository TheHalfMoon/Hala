# H349 — development model consumes verified training screenshot bytes

**Engineering Draft only. No newly trained Hala Qwen, model gain, GPU run,
independent rights reviewer, or release approval.**

## Concrete bug

The separately maintained historical `training/development_data.py::encode_row`
loaded a train-only original, CC0-authored Type PNG, verified a bounded
image byte snapshot against the reviewed `row["image_sha256"]`, then passed
the **mutable on-disk Path** to an independent `image_open(path)`.
This created the same hash-before-read TOCTOU as the older H348 GUI
path: the processor could consume different bytes/pixels from those
whose SHA-256 passed. This is not the H346/H348 trainer and its
training controls must not be silently conflated with official H199
source admission.

## Fix

Pass `BytesIO(raw)` containing the **already size-bounded and
SHA256-verified original snapshot** to the injected Pillow image
decoder. Keep original PNG header, 2 MiB, 800x600, path traversal,
family and train-only boundaries. Mind text-only processing still
never accesses an image. The injected `image_open` interface remains
a single Pillow-compatible readable object.

## Repro and verification

New `tests/test_h349_dev_snapshot_pinning.py` uses entirely invented
800x600 PNGs and deterministically changes the on-disk image
**after hashing but when the image decoder is invoked**. It proves
the real `encode_row` processor receives the original pixels and
refuses a substituted digest before any decoder; a third regression
checks Mind remains text-only.

Owner Windows local:
- H349 tests: **3/3 passed**
- Existing development training/adapter/precision/provider/runtime:
  **49/49 passed**
- No model import/GPU, source/labels or weights were disclosed.

Dedicated no-GPU GitHub Actions matrix checks both Ubuntu 24.04 and
Windows latest, Python 3.11/3.12, with pinned Pillow==11.3.0.
Only the *exact-head* passing CI outcome is acceptance evidence.

## Release, source and science gates

`h199/native_action_sft.py` official TRAIN admission registry remains
empty. Existing historical development bundle is CC0-authored but
does not grant current independent review for RailKit/SignalDesk
rendered-image training. All such data remain blocked for promotion.
No source-family-disjoint blinded OOD, legal independent screenshot
rights review, genuine improved adapted Qwen, bilingual model-time
win, or Goldfish approval exists. No merge or publish.
