# H48 — original diverse bilingual visual-grounding data pilot

**2026-10-21, noncommercial first-party research only.** Source H47 verified commit `abbdc218e8562e800b6335afc71ff60af8f5ba81`, [H47 Draft PR #32](https://github.com/TheHalfMoon/Hala/pull/32). Actual previous H46 pinned model benchmark: **0/8 correct** on original eight public first-party DEV photos (8/8 valid JSON, 3 wrong unique, 5 unbound). [H46 authentic negative evidence](https://github.com/TheHalfMoon/Hala/pull/31). H47 paired prompt study has its own [concurrent evidence PR #33](https://github.com/TheHalfMoon/Hala/pull/33), and any queued Kaggle GPU run must not be re-submitted from H48.

## Actual delivered assets

**96 genuine headless Edge Chromium screenshots** (1024 × 640), from fully first-party fictional HTML authored in `h48/factory.py`. Browser `Edg/155.0.4283.62`, locally cached pinned `puppeteer-core 25.12.0`. Browser capture `tools/h48_edge_render.js` denied all external network requests. There are **48 EN + 48 AR** screenshots with six visible, nonoverlapping buttons each (**576 browser-observed DOM control rectangles**). They are actual screenshots from code, **not generated or reused photos of real proprietary apps**. The text, item codes, task instructions, UI and screenshots are first-party original CC0-1.0 synthetic examples without PII or third-party benchmark task answers.

Four original authoring families (64 cases, 32 EN/32 AR): **modal approvals, inbox triage, access settings, ticket board**. Two **public DEV-only** families (32 cases, 16 EN/16 AR): **calendar slot reservation, expense review**. Every family has distinct CSS layout and visible item/button grouping; goal requires a particular fictional item code and a specific action. The target position cycles through all six controls; no hidden class, visible gold highlight or extra API granted to a model. Four style palettes per family. Arabic screenshots use `dir=rtl` and locally installed browser fonts. This tests varied **single next-click grounding**, not long-horizon state transition or OSWorld benchmarks. Template-generated content remains substantially less diverse than real software and may invite shortcut learning; 64 training examples cannot justify state-of-the-art claims.

**Immutable source/evidence:**
- `h48/assets/source_manifest.json` SHA256 `70278a175a331222ad2e54890020a07849980addf0ec1bb77be9c9441f15e139`.
- `h48/assets/capture.json` SHA256 `a01db14518f6ce7a2bf7fa89212eae4a78edbd69ea2fbc5b8bd5ccaa8dc3f127`.
- `h48/assets/model_inputs.json` SHA256 `42fb0d6da2ea0c201e64d4e2590eab20ea2135399a678c929eacbd1966291653`.
- `h48/assets/authoring_labels.json` SHA256 `d49938f20de583f289d3013e97c50e80750417d9e501bdb726fc4ac033242d22`.
- `h48/assets/evaluator_only.json` SHA256 `efc11edc0de0aa46b8d36ad61339ffce41d02b123544e4d15301d6d235f1d004`.
- `h48/assets/audit.json` SHA256 `e8c5ce24bd9cb7a00fa108f826a84aaf1e05f1d18a1fbbf7f8061ea701184c63`.

Original `html/` and browser-rendered `png/` assets are create-only and archived in Git. Offline `h48/audit.py` independently rebuilds each original HTML from the canonical source and checks every image SHA, real PNG format/dimensions, all DOM button names, pixel-visible nonoverlapping geometry, and unique target center. **Model-only** input files contain **PNG path/hash and natural-language goal**, never the correct target index, labels, DOM controls or coordinates. **Authoring-only labels** contain 64 next-click coordinates and explicit rights, with **zero public DEV export**; the separate evaluator-only reference file covers all 96 cases and is never to be passed to the model. Repository presence of evaluator labels means there is **no cryptographic filesystem isolation** and visible DEV tests are **not blind**.

## Qualification and material next work

- Run focused positive/negative tests, full serial Hala suite, Python syntax and genuine Graft wiring (record exact source checks). Deep Graft, PStack, D1, Alibaba OCR and independent adversarial/visual review must be honestly marked **not run** until evidence exists.
- Publish signed Draft PR with all source PNGs, provenance, raw capture and audit receipts. Do not merge or release merely because CI passes.
- **Next actual learning experiment:** create a principled training split, choose licensed Qwen3VL-compatible adapter (PEFT/QLoRA) only after exact processor/tokenizer/GPU quota admission; do a small free GPU adapter experiment **only if** labels, synthetic realism and independently evaluated non-overlapping tasks are adequate. Compare frozen original vs adapter on **the same independent heldout family/app**, report unsuccessful attempts and inference cost, preserve raw evidence. Never mix official benchmark trajectories into training.
- Expand real complexity and independently licensed task/visual families; evaluate actual OSWorld 2.1 and ScreenSpot-Pro with official scorer, multilingual human arbitration, test contamination checks and matched hardware/budgets before competitor-leadership claims.

**Current state:** H48 creates original data; no neural model inference, finetune, actual improvements, third-party app interactions, safety/production authorization or benchmark success occurred.

## Verified engineering gates — Windows, 2026-10-21

- Actual **96/96** Edge original PNGs and **576/576** browser DOM button boxes validated against canonical first-party source. Source/PNG SHA256 evidence is retained. H48 focused **5/5 tests passed**, including forged image hashes, DOM labels and positions, overlapping controls, false claims, rights and data-family splitting.
- Complete Hala serial Windows Python suite **572 run / 570 passed / 2 skipped**, exit 0 in **129.476 seconds**. Original transcript `reviews/h48-full-suite-20261021.log` SHA256 `08b8eb50b096e2fe25b15bdab300cdb4c4853e4dd69b9c4b98dfaaab6a73d6f6`. `compileall` PASS.
- Real Graft **wiring-only** `graft build .` and `graft check .` PASS **1,801 nodes / 4,874 edges / 219 files**, source graph in sync. **Deep semantic meaning 0%, not run**.
- Independent local D1/PStack/Alibaba Open Code Review/Graft deep checks are **not claimed** (executables unavailable as directly callable CLIs in this workspace).
- No GPU, Kaggle kernel submission, model training, user computer click, internet-accessed web app, external benchmark or production action. H47 queued private Kaggle study and its concurrently advancing evidence branch are deliberately not modified.
