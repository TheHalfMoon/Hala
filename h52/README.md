# H52 — First-Party Multi-step Task Factory

**2026-10-21. Actual browser-confirmed synthetic TRAIN data, not neural model inference.**

## What is verified

- `h52/factory.py` defines **96 genuinely different first-party fictional workflows** across four train-only families: `ar_judicial_office`, `ar_traffic_office`, `en_fhir_workspace`, `en_clinical_ops` (24 per family, 48 Arabic / 48 English workflows). These are not real Najiz, Absher, EHR or clinical accounts and do not use their protected UI assets.
- Each workflow contains **3 sequential required decisions**, 8 visually plausible controls per step, step-dependent shuffled placement, multiple grid/list layouts and themed palettes. Goals are tied to a complete task sequence; the next screen state changes only after the correct action. 96 workflows × 3 steps = **288 screenshot/action training pairs** (144 Arabic, 144 English).
- `h52/capture.js` launched a *genuine installed Microsoft Edge headless browser* with pinned `puppeteer-core 25.12.0`. Captured **288 genuinely distinct PNG screenshots**, verified every visible target's unique pixel midpoint, executed **288 actual correct browser clicks** with step-progress postconditions, and tested four intentionally wrong clicks which did not advance the state. Every HTML, PNG and per-step action has an original source/capture SHA256.
- `h52/audit.py` regenerates all first-party source plans and HTML bytes, rechecks every PNG byte, 288 JSONL rows, each expected correct stage/target and true browser-observed result. **No model was invoked.** This is browser/supervision verification, not 288 successful neural agent tasks or independent benchmark data.
- The first attempt failed honestly on offscreen controls for the third-column layout. The original CSS was repaired, incomplete local output removed before freezing, and the complete revised generation/Edge capture successfully replayed. No partial data were uploaded or counted.
- Original H52 train JSONL SHA256 `bca090f0a31aee88fba0cf47aaadaa895ca1fc7ede7c41fbf123d65d6ccb4c73`. Original source `h52/assets/source.json` and full signed browser output `h52/assets/capture.json` are in GitHub; all 288 PNGs remain original, first-party and model-free.
- H52 is a seed for a real Hala training pipeline. A future adapter runner must preserve **multi-step episode identity**, original observer/source separation and actual agent success across all three steps. Simple per-image click imitation is necessary but not sufficient: this data does not yet establish generalization, refusal/abstention or long-horizon recovery.

## Private Kaggle data library, version 2

Live private dataset: https://www.kaggle.com/datasets/abdulazizshehri/hala-gui-training-library — owner `abdulazizshehri`; remote Kaggle metadata `isPrivate=true`; dataset ID `12748399`.

- The original **H50 first-party TRAIN archive** was retained with its exact original compressed bytes and SHA256 `783426a4f8ae10c2ce85fa6b9cc861f69857e64372e1a4dc8a4cf81b8053c37c`.
- New H52 version-2 archive: `hala-h52-multistep-train.zip`, 7,856,827 bytes, SHA256 `37d7514538d51fd76ca8edf56c14ecacb80235e6ad1a0a3835e307f455ad113b`; ZIP contains 288 PNG files, 96 original first-party HTML files, original plans, capture receipts and 288-row training JSONL. Kaggle materialized all original archived members in version 2.
- **Independent authenticated remote readback**: downloaded `library-v2-manifest.json`, SHA256 `dfd7a0133ce892ad95db55a2de48ddd38db3857e56a7b7289fcdb40e4e08072a` — byte-identical to the staged version. Downloaded remote `authoring_sft.jsonl`, SHA256 `bca090f0a31aee88fba0cf47aaadaa895ca1fc7ede7c41fbf123d65d6ccb4c73` — identical to original H52 source. Kaggle metadata returned `isPrivate: true` after version upload.
- `research/data_library/h52/build_private_kaggle_v2.py` creates an entire original-data version-2 upload snapshot while retaining H50 archived zip. **No paid service** or new Hugging Face account needed. It requires an authorized Windows Python/zlib environment to reproduce the exact historical v1 compressed ZIP; cross-OS compression bytes may differ even when original source files are identical.
- The updated `research/data_library/catalog.json` marks H50 and H52 **stored_kaggle=true**, but **training_completed=false** and **independent_evaluation_completed=false** for every dataset. External competitor datasets remain pointers/source inventory only until actual source-media rights, train partitions and SHA verified.

## How to reproduce locally

```powershell
# In a clean Hala checkout with the original pinned puppeteer-core package:
python -m h52.factory --output h52/new-assets
# The capture script operates on one create-only, original source folder:
$env:PUPPETEER_CORE_PATH = 'PATH_TO_PINNED_PUPPETEER_CORE_25.12.0'
node h52/capture.js h52/new-assets
python -m h52.audit --root h52/new-assets
python -m unittest tests.test_h52_task_factory tests.test_hala_data_library -q
```

Do not run `capture.js` on the already qualified `h52/assets`: capture output is intentionally create-only. Never remove the original evidence before remote Kaggle readback and a signed GitHub commit are verified.

## Scientific and release gates still open

**0 actual H52 model training runs**. No GPU session, Qwen/Holo4 adapter checkpoint, loss curve, trained weights, learned success metric, independent sealed English/Arabic TEST or matched OSWorld/ScreenSpot performance report has been created. The model's prior H46 original Qwen3-VL-2B grounding result **0/8** is unchanged. The H47 original Kaggle run remains separate and queued at last live check; do not duplicate it. Real future training must compare source-frozen base vs H49 64 rows, H50 384 rows, and H52 288 **distinct-screen multi-step** rows under comparable actual GPU effort, preserving provenance, per-language wrong-target/recovery rate, latency, VRAM and successful-task cost. Holo4 benchmark test trajectories must never be mixed with train source.

H51 independently inspects genuine DragOn/AgentNet/GUI Odyssey upstream provider metadata in a different Windows worktree; H52 does not alter or report completion of that research.

## Genuine source test interruption during version-2 catalog update

The first Windows full `unittest discover` launched **before** the H52 Kaggle remote readback and `catalog.json` update were complete. It correctly **FAILED two catalog tests** on 630 tests (628 passed or skipped; 2 failures, 2 skipped), because at import/discovery it observed the original **13-entry** ledger, while `test_hala_data_library.py` had just changed its expected cardinality to 14 as the concurrent catalog mutation finished. It was a source-tree concurrent-timing defect in this engineering session, **not a reported model failure**, and the raw original `reviews/h52-full-suite-20261021.log` is retained rather than erased. The subsequent final focused H52+Kaggle suite **8/8 PASS** on stable 14-entry source and actual version-2 Kaggle SHA. A fresh serial full-source test run on stable final tree is required before describing Windows full suite as PASS. Exact-head GitHub-hosted CI is the independent qualification gate.
