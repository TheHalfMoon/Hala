# H48 — source-verifiable first-party browser GUI grounding SFT bridge

**2026-10-21. Draft engineering data export only. This is not model training, model quality improvement, or independent benchmark evaluation.**

Base: [H48 original screenshot evidence PR #35](https://github.com/TheHalfMoon/Hala/pull/35), exact signed parent `e3c28d76beb9e00faf6878fcbb96d40e1d4710c7`. This independent bridge uses the existing H48 source, verified Chromium screenshots, DOM measurements and SFT artifact loader. No copies of the screenshot PNG files are needed.

## Delivered

- `h48/sft_bridge.py` reconstructs 64 training-only pixel-point action examples from H48's originally measured button rectangles, rather than trusting supplied labels. It invokes the independent `h48.audit.verify` replay for **all 96 original 1024×640 screenshots**, then compares four saved source projections byte-for-byte with that derived audit.
- The supervised training split is exactly **32 Arabic RTL + 32 English**, across four distinct authoring families. Calendar and expense are publicly visible DEV-only families and are **never** copied to SFT.
- The action completion uses the existing `training.artifacts.validate_action_completion` exact logical click schema: `type=click`, **integer WINDOW pixel** `point`, `image_scale=1` and constant `element_ref_hint=visual_target`. This is a **training annotation**, not an observed output from any model. H47's separate normalized-point inference protocol remains untouched.
- The model-facing prompt includes only a natural-language goal, screenshot size and an action format request. It contains no privileged DOM, expected coordinate or winner label.
- `training.artifacts.load_sft_artifact` accepts the H48 route **only when the entire manifest exactly equals a fresh independently reconstructed manifest** and its JSONL and original PNG file SHA256 hashes match. The existing legacy original synthetic Pillow-path rights gate remains intact; we do not relabel these Chromium assets as `data.factory.RIGHTS`.
- Arabic is recorded as verified **browser-rendered RTL PNG origin**, not verified Pillow RAQM shaping. Model-level Arabic text recognition and human linguistic quality are still unqualified. The H48 source authoring/template families are limited and publicly inspectable.

Run from Hala repository root:

```powershell
python -m h48.sft_bridge --root h48/assets
python -m unittest tests.test_h48_sft_bridge -v
python -m unittest discover -s tests -q
```

The export is **create-only**. It writes two files inside the original H48 image directory; they can be source-controlled without duplicating image bytes:

- `h48/assets/h48-sft-authoring.jsonl`
- `h48/assets/h48-sft-manifest.json`

The builder refuses an existing target instead of overwriting it. Any changed original screenshot/source/geometry, altered DEV family, substituted provenance, forged Arabic render route, tampered completion or repackaged JSONL SHA causes a fail-closed error. The new manifest explicitly records `training_executed=false`, `adapter_qualified=false`, and `independent_blind_test=false`.

## Why no GPU launch is authorized by this bridge

Existing `training/scripts/train_adapter.py` is a candidate remote FP16 LoRA implementation with a historical technical H15 two-step smoke. It explicitly refuses Kaggle training because that trainer's inspected provider-eligibility policy has not been reconciled. **Do not bypass that eligibility guard.** H47's original 16-trial private Kaggle public DEV kernel is a separate, already submitted version-1 job; do not submit it again, alter its frozen protocol or train on its exposed public DEV tasks.

Before any actual H48 adapter experiment: establish platform/research eligibility and checkpoint/processor licenses, free GPU availability, loader and output-head compatibility, measured Arabic preprocessing, a newly independently sealed family-disjoint held-out evaluation set, and a frozen same-harness baseline. Capture genuine adapter parameter updates, source/media/checkpoint hashes, every failure, memory/latency/time consumption and per-language task outcomes. Do not use benchmark TEST trajectories or benchmark privileged answers in training.

The H48 pilot labels a **single next-click only**, not full OSWorld tasks or long-horizon completion. Neither the 64 training annotations nor the public 32 DEV examples qualify Hala as production ready or better than Holo4, OpenCUA or UI-TARS.
