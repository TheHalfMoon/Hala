# H37 — offline real-browser-rendered synthetic GUI grounding benchmark

**Status:** Source and first-party browser pixels captured and CPU-qualified, but **NO new neural model outcomes until Kaggle run is completed and independently replayed**. Draft Hala-only. Not a real external website screenshot dataset, not a cryptographically blind heldout TEST, and not native OS automation qualification.

## Purpose and distinction from H34–H36

H34–H36 used 32 already-exposed raster drawings and returned only 8/32 correct-target hits with historic Qwen3-VL-2B constrained-prompt baseline. H36's removal of the numeric example degraded output formatting (22/32 strict JSON; ten duplicate `point` keys), and produced 6/32 correct hits. We need **new rendered image layouts and domain families** rather than reusing H34 for prompt optimization.

H37 provides 64 deterministic first-party HTML/CSS pages, **rendered to actual PNG pixels by Microsoft Edge headless (Chromium 155.0.4283.62)** via pinned `playwright-core 1.56.1`. Edge read only local `file://` pages while Playwright's context was offline; there were no external URLs, scripts, clicks, customer data or website data collection.

The output is *authored browser-generated synthetic UI*, **not screenshots of independently existing real-world applications**. It tests a more realistic typography/layout rendering path, but does not qualify user-facing computer use.

## Fixed prospective dataset

- **64 cases**, evenly split 32 DEV and 32 distinct-domain TEST, 32 English and 32 Arabic.
- Four new task families: DEV `makerspace`, `community_theater`; TEST `trail_maintenance`, `library_workshops`. None is in H34's four original families.
- Four CSS arrangements: `toolbar`, `sidebar`, `grid`, `table`. Two alternate permutations per family/language/layout, four unique on-screen control labels each.
- Exactly 16 intended cases for each control position (indices 0–3). Original fixed viewport **960×540** at device-pixel ratio 1, no clicks.
- All **64 screenshots have different SHA256**. Browser DOM `getBoundingClientRect()` provides authoritative observed rectangles for the authored UI; the target ID is defined in source case labels, never in model image or prompts.
- Model-only `h37/assets/model_inputs.json` contains solely case id, split, language, image path, user goal and viewport; separate `evaluator_only.json` contains DOM geometry and expected label. The H34 strict normalized-point scorer remains **unchanged**, rejects malformed JSON, duplicate `point` keys, ambiguous/wrong/off-target points.

### Reproducibility pins

H37 authored scenes SHA256: `8064af96a3fd4c2389e1c494829a56549bacd5a21ea2cc20f9fef987be5634d2`.

Actual offline Chromium capture `h37/assets/capture.json` SHA256: `f8cb6b6a3b7093b7ce26fdb8ef18fdd50e87a269c25a9b5a8657e62372e956f9`.

Model-only projection SHA256: `12f4b4615b2da5da5513dab51ffba434d5ccfd9a726ba978d4ed1f50ba7ff00d`.

Evaluator-only labels / browser geometry SHA256: `80983a038e9b58191c99ba6254103fda622c5a1224f11348f9895f7c8bde3cd0`.

Each original HTML file, Edge PNG and control bounds is retained under `h37/assets/` with per-file hashes. An independent CPU `checked_png()` rejects truncated/oversized PNGs, checks IHDR dimensions/CRC and IEND, refuses changed source/destination. Unit tests ensure no oracle leakage, non-overlapping/visible controls, center-of-button hit, different control miss, balanced splits/target positions and source manifest consistency.

## Genuine remote model baseline (separate release gate)

A create-only private Kaggle notebook uses Hala source's current signed exact HEAD from `ps/h37-browser-rendered-dev`, with model `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`, fixed H34 numeric-example JSON prompt, exact FP16 T4 loader and matching H34 package pins. It sends only screenshot plus task text, **not evaluator labels**, to the model. 64 one-shot inference attempts (no interactive actions/training) are required.

The `h37.audit` verifier demands independent SHA256 of the raw journal, verifies all 64 source screenshots and separate model/evaluator files against the original signed metadata, checks exact attempt order, image/prompt identity, and recomputes unconditional hit/wrong/abstain/invalid/error results by DEV/TEST/language/layout/domain. Missing attempts remain denominator failures; no retries or successful-case filtering. The authorship-domain TEST is a pre-registered disjoint split **but already public in the repository**, hence not independently blind.

Only launch on verified permitted noncommercial **zero-cost** Kaggle Batch T4 quota; no billing, paid API, subscription or unreviewed package pins. Hardware observations are not cryptographic attestations. A green CI run is not a model-quality result.

Run local source gates:

```powershell
python -m unittest tests.test_h37_browser_grounding tests.test_h37_kaggle_protocol -v
python -m unittest discover -s tests
python -m compileall -q h37 tools/build_h37_kaggle.py
```

After source is signed and pushed, generate a create-only private Notebook from a **clean detached exact source SHA**:

```powershell
python -m tools.build_h37_kaggle --root . --source-commit <exact-signed-source-sha> --output <new-bundle-folder> --research-allowed
```

Only the explicitly opted-in research notebook can submit via `kaggle kernels push`. After a completed run, retain notebook, raw outputs, metadata, fresh official GPU quota, original screenshot hashes, external raw SHA256 and standalone `h37.audit` report under a new H37 evidence Draft PR. Keep the exact source branch stable for reproducibility.

## Do not overclaim

No actual trained-model improvement, no first-party external application or user screenshot, no independent Arabic expert judgement, no blindly controlled real-world TEST, no true H23 CUDA mid-generation cancellation/worker restart evidence, no Aladdin changes, no native host action authorization or founder release approval is demonstrated by H37 source alone. The reference model may perform poorly on this task too; preserve negative outcomes if observed. New genuinely independently captured apps and expert validation remain required before release.

## Post-registration results: actual free private Kaggle baseline, 2026-10-20

**Retrospective annotation added only after the original source was frozen at `4c061e9c2e15df0306062b5a5666c6e0d7697502`.** Private Kaggle research notebook v1 completed 64/64 actual Qwen3-VL-2B model requests on the new pinned browser-screenshot PNGs, with zero runtime errors or UI clicks. Exact raw SHA256 `40b88ceb592446630817b9562238b40765b371fac61ccb4c06b427fab92cef1d` was independently verified against retained `raw_attempts.jsonl`, and strict `h37.audit` produced report SHA256 `987c8338342faaf8d2718288623a9f485d058acdde0f21f5f3133befaa1451f2`. All 64 outputs were parseable; **32/64 intended target hits (50%), 16/64 wrong visible controls, 16/64 no unique control bind**. DEV 14/32, authored-domain TEST 18/32, English 18/32, Arabic 14/32. By layout: toolbar 9/16, sidebar 6/16, grid **3/16**, table **14/16**. See immutable [H37 Kaggle raw evidence and replay](../reviews/h37-kaggle-run-20261020/README.md).

The reference H34 run on a different authored raster corpus reported 8/32; **direct improvement is not demonstrated**, because input data, layouts, UI family, rendering and evaluation conditions differ. H37's authored-domain TEST is publicly visible, not blind or independently externally sampled. Real-world application screenshots, independent Arabic semantic assessment, H23 genuine CUDA lifecycle and native host safety continue to block model adoption, merge and production release.
