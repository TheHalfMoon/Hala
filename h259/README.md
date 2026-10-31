# H259 — H250 inline icon provenance: Heroicons v1 MIT match

**PARTIAL RIGHTS EVIDENCE ONLY. No TRAIN/DEV/OOD admission, model inference,
LoRA training, weights, screenshots or original source bytes in Git.**

A source-only investigation of H250's untouched original app found that its
empty-state clipboard/check glyph uses the same path sequence as
[Heroicons v1.0.6 clipboard-check](https://github.com/tailwindlabs/heroicons/blob/v1.0.6/optimized/outline/clipboard-check.svg).
The original official icon and
[Heroicons v1.0.6 MIT LICENSE](https://github.com/tailwindlabs/heroicons/blob/v1.0.6/LICENSE)
were fetched **only into owner-private Windows storage** and pinned to their
official public Git blob IDs, verified against GitHub's original repository.
No SVG paths, original donor bytes, icon raster, or private source receipts
are copied into public Hala.

## Actual owner Windows result — 2026-10-31

- H250 original upstream Git checkout is clean and at its pinned commit.
- Heroicons v1.0.6 official icon and MIT license both pass pinned Git blob
  verification; official license names **Refactoring UI Inc. (2020)**.
- H250's two SVG paths structurally match the official icon's single path
  with a relative move separating the checkmark subpath.
- A native **offline Microsoft Edge** in-memory 192x192 raster comparison
  with identical normalized stroke styling differs at **5 of 36,864 pixels**,
  maximum **1 channel value**. This is near-equivalence, **not byte-identical
  rendering** and not proof of the original author's acquisition route.
- H250's root MIT LICENSE names the app author but **does not contain the
  Refactoring UI Inc. notice**. A text search of the original tracked
  repository did not find an explicit Heroicons/Refactoring UI notice.
  This is a **potential third-party MIT notice-compliance gap** requiring
  review, not a legal finding that infringement occurred.
- The app's other SVG, offline-blocked remote fonts, historical source
  identity and pixel-history completeness remain unresolved. No media is
  promoted to TRAIN/DEV/OOD; H199 allowlist remains empty and H173 scored
  OOD remains irrevocably sealed.

## Reproduce without uploading media

Use the authorized owner Windows device. Download only the official Heroicons
v1.0.6 icon and LICENSE into an **owner-private directory outside Git**, then
run:

```powershell
python -m h259.inline_icon_provenance `
  --h250-repo D:\Hala-H250-Original-simple-to-do-list `
  --heroicons-svg PRIVATE_HEROICONS_ICON_ABSOLUTE_PATH `
  --heroicons-license PRIVATE_HEROICONS_LICENSE_ABSOLUTE_PATH
```

The program prints only safe aggregate booleans and pixel-difference counts.
It fails closed if pinned bytes, original Git state, structure or bounded
raster agreement changes. It never writes any PNG or opens owner research
frames. Hosted CI runs **invented SVG synthetic tests only**, not the real
original-app/Edge provenance check.

Owner Windows qualification: H259 invented synthetic tests **9/9 PASS**,
Python -O **9/9 PASS** (expected pytest warning); H250-H259 focused
integration **95/95 PASS**; broad selected historical/native regression
**467 PASS, 6 SKIP, 2598 unrelated deselected** in 70.00 seconds with a new
isolated D: pytest basetemp. `compileall` and Git whitespace check pass.
No GPU, model inference, optimizer steps or new model accuracy measured.

**Next gate:** resolve all third-party media rights and notices, and complete
historical source and visual disjointness before any clean independent split,
fresh sealed blind OOD or genuine Qwen3-VL evaluation.
