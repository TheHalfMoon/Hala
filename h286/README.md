# H286 — Qwen-relative alignment of original ShiftDesk TRAIN candidates

**SOURCE ONLY, TRAIN CANDIDATES ONLY. No new Qwen training, inference, official data admission or blind OOD qualification in H286.**

H285's author-owned 24-scene bilingual ShiftDesk factory yields genuine real-browser verified 960x720 screenshot-pixel click supervision. The successful pretrained H279/H282 Qwen native path instead generates normalized 0..999 coordinates and converts them to H199 pixels before actuation. Training on pixel numbers but evaluating normalized numbers would confound an adapter comparison.

H286 supplies a strict prospective training-only inverse: normalized = ceil(1000 * pixel / extent). All 960 x and 720 y positions are exhaustively checked against H273's unchanged normalized-to-pixel decoder. It preserves the exact pixel, rejects floats, booleans, extra JSON keys, abstain and non-click labels, and uses exactly H282's model-visible relative prompt rather than H199's conflicting pixel prompt. This never rewrites model predictions.

The wrapper invokes H285's already browser-qualified generator, validates all 24 source records, source blob, 12 EN/12 AR coverage, source-owned real click results and no admission/DEV/OOD promotion. Only after all 24 transformations succeed does a trusted owner-private TRAIN sink receive volatile (PNG, relative prompt, strict relative JSON). The sink itself is not atomic and must not publish/store labels or images in public or evaluation outputs.

Run invented-only source tests:

    python -m unittest discover -s tests -p test_h286_relative_candidate_training.py -v

These CPU tests do not render a browser, call a model, read consumed H173/H212, publish gold or change H199's intentionally empty admitted TRAIN manifest. H285's independently qualified Chromium/Edge browser workflow remains the real input source proof; H286 is a coordinate-frame alignment prerequisite, not learned skill.

A subsequent owner-private noncommercial FREE T4 research-only LoRA trial must use the SAME relative prompt and grammar during training and native inference, with source-disjoint (but previously exposed, not blind) paired research cases. Genuine lawful admitted multi-family TRAIN/DEV, entirely untouched private blind OOD and actual adapted native task-state gains remain mandatory before any release.
