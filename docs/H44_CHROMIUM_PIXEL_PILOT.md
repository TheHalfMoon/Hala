# H44 — 24 authentic Edge Chromium screenshots of first-party bilingual task pages

**Status:** source, real browser pixels and evaluator geometry; **NO model run, no training or independent benchmark**. H44 is stacked on signed [H43 PR #27](https://github.com/TheHalfMoon/Hala/pull/27) head `6c0055ce954d74062e7ef4782705884ea4716e89`; the original H43 state-task source/data remain unchanged.

## Actual renderer and provenance

H43 contributes 24 entirely original first-party CC0-1.0 simulated three-step tasks (12 English, 12 Arabic) with 16 authoring examples in four app families and 8 public DEV examples in two different app families. H44 renders their **first observation only** into local self-contained HTML/CSS, preserving language `lang`, bidirectional `dir`, visible labels and distinct app/layout presentation (list, grid, table, sidebar). HTML is first-party synthetic, not independent external app content, and contains no scripts, network URLs, private files, credentials or evaluation answer class.

Observed actual capture: **Microsoft Edge Chromium `Edg/155.0.4283.62` headless**, 960×540 at deviceScaleFactor 1, with `puppeteer-core 25.12.0`, stored in 24 original PNGs. The source replay script `tools/h44_edge_capture.js` requires an explicitly supplied **locally installed pin-exact** Puppeteer-Core package, and refuses a different version; it refuses any non-file browser requests, launches only the installed Edge binary, observes real DOM `getBoundingClientRect` on five visible controls per page and saves the PNG bytes and actual browser metadata before exit. Package bytes are NOT vendored and there was no npm install or paid API in this operation. A different Edge browser patch or package may yield different pixels: use the source/digest receipts, not claims of universal bit-for-bit rerendering.

HTML and PNG evidence under `h44/assets/pages/` and `h44/assets/screenshots/` is created once, no overwriting. Actual raw capture `h44/assets/capture.json` SHA256 **`ba87f75f412ba235903ec72e119fc8fb021455a646f8a0c9becb5adc455cd14c`** is retained with every screenshot and page SHA256, viewport, browser version, five observed target roles/names and their pixel rectangles. All captures passed PNG magic, CRC and IHDR width/height checks. Five observed controls per page are inside the viewport with no overlap or inaccessible zero-size rectangles.

## Deliberate no-answer-leak input split

- `h44/assets/model_inputs.json` SHA256 `f432747051c5fc2b6741db1a8c9b4b58f04d0dd7248c77546baa191bcdc29193`: **only goal, PNG path/digest, task metadata and viewport**; no DOM bounds, target answer, golden actions, desired state or evaluator labels.
- `h44/assets/evaluator_only.json` SHA256 `974b6ce6a51d272437e67471c10ee07f03b9303427bac9ec66e0391fe254ee22`: original expected **first action** (`row_target`), actual observed DOM rectangles and labels for strict clicked-target adjudication. Keep this separate from model prompt/inference runtime. The user-authored source pages contain `data-ref` IDs in DOM for evaluator reconstruction; a cheating DOM-aware model could read them, so the image-only study MUST expose **PNG pixels only** and not HTML or DOM IDs to the model.
- `h44/assets/audit.json` SHA256 `336414ea11bdc8231841e1ba23f74c08699eace091197e6cc3ce9db8916ff0bc`: exact 24 screenshot/DOM integrity, 12/12 language balance, 16 authoring/8 public DEV source identities, and explicitly all qualification bits FALSE.

The strict independent **CPU replay** `h44/audit.py::verify` recomputes each HTML source from the unmodified H43 canonical author, compares immutable PNG SHA, validates Edge metadata/versions and all control roles/names/bounds against original rights-cleared sources, detects overlap/missing controls and verifies oracle separation. `tests/test_h44_browser_pixels.py` includes 4 focused CPU tests, actual 24-image replay, tampered synthetic browser receipts, and split/answer-leak refusal. There was **no network activity in captured HTML pages**, no clicks, native OS control, actual HalaType neural inference or GPU consumed.

## What H44 does and does not establish

H44 is **real browser-rendered synthetic UI pixel data** suitable as a future *prospective image grounding pilot* after separate data rights/quality review. It does **NOT** implement the second/third step's changing screen, real user typing, rendered transient errors, long-horizon action recovery, independent Arabic linguistic assessment, unseen independent app tests, actual model inference/training or OSWorld 2.1/ScreenSpot-Pro leaderboard performance. Like H43, all six app contexts share the same underlying small state transition. Public DEV isn't sealed blind TEST. No claim that the model is better than competitors is justified.

Next experimental grain: have an independently evaluated Hala candidate predict ONLY the first screenshot target on this data, with a preregistered baseline and uncertainty/fail-closed; then build *real multi-step HTML state* rendered across transitions and an independently sealed outside-app holdout. Retain all wrong/unbound outputs and avoid train/DEV crossover; train/adapt only on clearly rights-cleared authoring records and after genuine hardware and source safety admission. H41 original CUDA cancellation run remains separate and still unqualified.

## Actual Windows local gates (2026-10-20)

- Authentic Microsoft Edge headless `Edg/155.0.4283.62` with locally cached `puppeteer-core 25.12.0`, no new npm installation: **24/24** first-party browser PNGs and their 120 observable DOM control bounds captured. Full original case/HTML/source, exact PNG CRC/dimensions/hashes, language/RTL and geometry replay PASS; `h44/assets/capture.json` SHA256 `ba87f75f412ba235903ec72e119fc8fb021455a646f8a0c9becb5adc455cd14c`.
- H44 focused **4/4 tests PASS** (tampered capture/PNG/controls, source/output split); `compileall` PASS.
- Full Windows serial **554 tests run / 552 passed / 2 skipped**, exit 0, **131.988s**. Original source test log retained in `reviews/h44-full-suite-20261020.log`, SHA256 `bd381e1cbb68e54a4f9cc16cdb870301d176bd4be7f3c76804f9b34037771824`, zero matches in bounded basic credential-like pattern scan.
- Graft **wiring-only** build/check PASS **1,718 nodes / 4,623 edges / 203 parsed files**, source graph in sync. **Deep meaning tier not run** (0% built).
- GitHub exact-head CI on Python 3.11/3.12 must be checked after final signing and publication. Independent PStack/D1/Alibaba Code Review/Jev reviews have **not** been executed or claimed.
- No new model/GPU/weights, external website, paid inference, OS click, actual device or benchmark completed in H44; H41 private original CUDA Batch remains separate and unqualified.
