# H48 — source-verifiable first-party browser GUI grounding SFT bridge

**2026-10-21. Draft engineering data export only. This is not model training, model quality improvement, or independent benchmark evaluation.**

Base: [H48 original screenshot evidence PR #35](https://github.com/TheHalfMoon/Hala/pull/35), exact signed parent `e3c28d76beb9e00faf6878fcbb96d40e1d4710c7`. This independent bridge uses the existing H48 source, verified Chromium screenshots, DOM measurements and SFT artifact loader. No copies of the screenshot PNG files are needed.

## Delivered

- `h48/sft_bridge.py` reconstructs 64 training-only pixel-point action examples from H48's originally measured button rectangles, rather than trusting supplied labels. It invokes the independent `h48.audit.verify` replay for **all 96 original 1024×640 screenshots**, then compares four saved source projections byte-for-byte with that derived audit.
- The supervised training split is exactly **32 Arabic RTL + 32 English**, across four distinct authoring families. Calendar and expense are publicly visible DEV-only families and are **never** copied to SFT.
- The action completion uses the existing `training.artifacts.validate_action_completion` exact logical click schema: `type=click`, **integer WINDOW pixel** `point`, `image_scale=1` and constant `element_ref_hint=visual_target`. This is a **training annotation**, not an observed output from any model. H47's separate normalized-point inference protocol remains untouched.
- The model-facing prompt includes only a natural-language goal, screenshot size and an action format request. It contains no privileged DOM, expected coordinate or winner label.
- New `h48.sft_bridge.load_export` admits H48 **only when the entire manifest and JSONL match independent reconstruction from original Chromium/DOM evidence**. Existing `training/scripts/train_adapter.py` dispatches explicitly to this new route; the H14-frozen `training/artifacts.py` is preserved byte-for-byte. The legacy original synthetic Pillow-path rights and RAQM gates remain intact; Chromium assets are not relabelled as `data.factory.RIGHTS`.
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

## Exact-head repair and negative evidence

The initial signed head `0cfee3484a04f579fbb6f15d324947d3b8da2c10` changed `training/artifacts.py`, one of the byte-frozen H14 original-source files. GitHub's **original** [push CI run 39519594251](https://github.com/TheHalfMoon/Hala/actions/runs/39519594251) and PR run 39519684842 correctly failed the original H14 frozen source/specification tests. This is a **real regression**; do not discard or describe those runs as passing.

A long local initial suite against that head also failed: **577 tests, 2 errors in H14 source-freeze verification and 1 timeout-sensitive pre-existing serving test assertion**, 2 skips. The serving timeout assertion was not proven a newly introduced functional regression; an isolated identical test subsequently passed. The original failure transcript is archived without correction as `reviews/h48-sft-full-suite-20261021.log` SHA256 `517fa1e1375dbba11a8bcad78eaeb1888619cdf1431cf4187c0a7e90508c8656`.

The forward-only source fix on signed head `352dfbf470725a53b660596992f49a4232bb0d74` restored the exact original H14-frozen `training/artifacts.py` bytes and moved H48-only admission to `h48.sft_bridge.load_export`, with explicit dispatch from the otherwise-unfrozen `training/scripts/train_adapter.py`. A new test checks that the H14 frozen file digest matches its original manifest.

Post-repair focused actual Windows test suite: **9/9 PASS**, including both original H14 freeze checks, six H48 source/forgery/frozen-asset tests and the earlier timeout-sensitive serving test. Original evidence retained `reviews/h48-sft-repair-focused-20261021.log` SHA256 `1ccebd38677c41c83b37f3ba2e92b53ce7e5281d712bdcd6148c2c2a9bc36bf4`.

Post-repair actual Graft **wiring-only** build/check: **PASS 1,815 nodes, 4,916 edges across 221 source files**, deep semantic meaning tier NOT RUN. Retained `reviews/h48-sft-repair-graft-20261021.log` SHA256 `0f0c4e56efeee96e6b37626a3ccf5b8c0ee943afd5719454211820b99dc7d235` and `reviews/h48-sft-repair-graft-check-20261021.log` SHA256 `f534fb1e8a5f392b5ddca429038f9a73300652eb28988c368c28cf4bef118443`.

**Final GitHub exact-head Python 3.11/3.12 CI must be read live for the final signed evidence commit**. A failed/queued older run never qualifies a new commit. Local D1/PStack/Alibaba standalone CLI semantic reviews remain unavailable/unperformed; do not fabricate their approval. No H47 Kaggle resubmission, adapter training, new GPU model run or third-party benchmark occurred.
