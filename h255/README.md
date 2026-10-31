# H255 — Actual in-memory pixel ablation of original-app media

This is **rights-risk research only**, not an admission, a media-rights clearance, a neural evaluation, or proof of OS-level network isolation. H254 proved that an original LifeOS image loaded; H255 tests whether hiding that original image changes actual screenshot pixels in a stable browser scene.

Run on owner Windows, with the clean pinned original public source checkout:

```powershell
python -m h255.original_media_pixel_ablation --repo D:\Hala-H251-Original-LifeOS --sha b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c --entry index.html --selector img
python -m pytest -q tests/test_h255_original_media_pixel_ablation.py --basetemp D:\Hala-H255-Private\pytest-new
```

The browser uses the existing H243 offline protection before page creation, blocks external HTTP and out-of-root file requests, then makes four **in-memory** PNG captures: baseline, repeated baseline control, image-hidden ablation, and restored control. Only aggregate changed-pixel counts inside and outside the visible target rectangle are emitted. No image, donor asset, hash, source identity from private records, label, gold, or owner screenshot is saved or uploaded.

A causal visual-dependency observation requires zero baseline/restoration drift and nonzero ablation differences. If changed pixels extend outside the target rectangle, the result is separately classified as VISIBLE_DEPENDENCY_OUTSIDE_IMAGE_RECT: simply masking the image DOM rectangle would be insufficient. Dynamic baseline/restoration differences are explicitly INDETERMINATE, not a rights clearance. H255 does not infer why outside pixels changed. The original source is pinned, licensed-source audit is required, and the original checkout must be clean. No original image bytes are added to Hala.

Even a positive ablation **does not prove ownership, licensing, or complete historical disjointness**. It only proves that this unresolved-rights local asset influences rendered pixels. The original H173 scored OOD remains sealed, H199 admitted TRAIN allowlist stays empty, and all research captures are barred from optimization until independent source/media/history/split qualification.

## Observed actual original LifeOS run (owner Windows, 2026-10-31)

With the pinned original LifeOS app in Edge, repeated baseline and restoration screenshots were pixel-identical (0 changed pixels each). Hiding its original local logo changed **1,760 pixels inside** the image rectangle and **774 outside** it. Two external HTTP requests were blocked; zero external responses, out-of-root file requests, or observed WebSockets. Thus the image visibility intervention has a reproducible screenshot effect extending beyond the image DOM rectangle. A simple rectangle mask cannot be assumed sufficient for clean training data. This is **not** proof that all outside changes originate from the image bytes themselves, and **not** rights clearance or dataset admission.

No screenshots, original media bytes, or raw private digests were written to disk or Git. The result is aggregate-only and remains INCOMPLETE_NO_ADMISSION.
