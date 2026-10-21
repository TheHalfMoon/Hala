# H58 — First-party mixed ordered/unordered original Edge PUBLIC DEV

**2026-10-21. A genuine 24-scene, bilingual, task-source-disjoint PUBLIC development suite for validating mixed-domain original native drag gestures. NO real neural Qwen/Hala inference or training was run. This is NOT an independent blind benchmark.**

## Why H58 follows H57

Original H Company DragOn TRAIN has four drag skill domains: `slider` previously imported in H53/H54 and three additional `sheet`, `slide_resize` and `text_highlight` authentic training domains imported privately in H57. H57 source has **932 direction-sensitive `ordered=true`** and **654 direction-insensitive `ordered=false`** original labels from 117 new JPEGs and 1,586 verified original tasks. The H55 first-party public-DEV slider-only test cannot qualify a genuine **mixed-domain** model baseline.

H58 adds *new original first-party* fictional DOM interfaces: eight cell-selection sheet scenes, eight slide element-resize scenes, eight text highlight scenes, **12 Arabic and 12 English screenshots** from a real installed Microsoft Edge (1024×640 PNG). All original source HTML, screenshots, goals, immutable source manifest, source author geometry, true browser postconditions, and hashes live under `h58/assets`; H58 is PUBLIC DEV only, **never TRAIN**, and no actual H Company source images, original tasks, eval benchmark screenshot, government account, health data, real user or live service is involved.

**Immutable original H58 source SHA256:** `a1c17c5dca028096bdd653de927fae18b1797a241ca3529646e10a1eb5f4f3de`.

## Actual browser qualifications

`h58/original_multidomain_dev.js` uses **pinned Puppeteer-core 25.12.0** and genuine installed Microsoft Edge to perform original source-author pointerdown/move/up, reject live outbound requests, and capture PNG screenshot BEFORE executing gestures. On real DOM:

| Gesture | Real Edge test | Result |
|---|---|---:|
| Original authored correct START → END | 24 distinct screens | 24/24 accepted |
| Deliberate wrong start/origin | 24 distinct screens | 24/24 rejected |
| Deliberate wrong end/target | 24 distinct screens | 24/24 rejected |
| Inverse END → START for `ordered=false` sheet+highlight | 16 screens | 16/16 accepted |
| Inverse END → START for `ordered=true` slide resize | 8 screens | 8/8 rejected |

`h58/audit.py` checks exact original source SHA and original plan/source/HTML/PNG/screenshot SHA, 24 unique real Edge 1024×640 PNG, all original per-scene GUI receipt fields, six balanced domain/language groups, required original `ordered` flags, proper separation from TRAIN and absence of claimed model inference. **These are author-generated source-oracle tests, not a measurement of visual-language model correctness or of a trained Hala model.**

## Ready for future unchanged real model outputs

`h58/score.py` accepts **exact 24** original image-SHA-bound raw model JSON strings under `hala.h58.raw-public-dev-native-drag-attempts.v1`. It checks a strict single native drag object containing `type`, `start_point`, `end_point`, `coordinate_space=source_image_pixels`, and boolean `ordered`. For **unordered sheet/highlight**, it considers correct source endpoints in either direction semantically equivalent. For ordered slide resize, **only the forward direction** earns credit. Invalid raw JSON, duplicate keys, mismatched action schema, noninteger/out-of-image coords, wrong flags, missing/duplicate images and wrong source provenance are rejected or counted as non-success. This metric is **geometry diagnostics**, NOT actual GUI task success.

`h58/replay_raw_actions.js` executes each supplied **unaltered** raw candidate action in actual Edge offline against the original source HTML after independently checking original source and screenshot bytes, then records real DOM successful/failed task completion per domain and language. Wrong action `ordered` flag is rejected instead of silently overriding the original task semantics. The runner has NO model weights and cannot produce model inference. Both scorer and Edge replay explicitly report that provider inference authenticity is NOT independently attested.

The raw input envelope must contain exact 24 original source IDs, their exact image SHA256, and **unchanged provider model responses**, tagged `origin=provider_raw_model_outputs`. These cannot be manufactured from the author-golden `capture.json`; only a genuinely executed frozen Qwen/Hala checkpoint inference session can produce authentic model predictions. Public DEV is not a blinded test. Future model training cannot use these H58 screenshots or their golden coordinates.

### Synthetic unit qualification of the future model evaluator — NOT model performance

Using source **author-golden fixtures tagged `origin=synthetic_unit_fixture`**, the real Edge replayer completed **24/24** correct proposed gestures. Reversed fixture gestures completed **16/24** — all 16 intentionally unordered tasks, and zero of the eight direction-sensitive resize tasks. These directly verify real Edge gesture semantics; they are NOT examples returned by Qwen, Hala or another neural model and do not count toward model accuracy.

Immutable provider-free local evidence:

- `h58/evidence/synthetic-forward-edge.json`, SHA256 `2a0b306890dc3a22d1603ff3b3ad6a367c30e7180a2ebed4b08b134ebede0884`.
- `h58/evidence/synthetic-reverse-edge.json`, SHA256 `2d331b7b8110c9139fe1045a61eca839dfd6c015a25a15101243f74e0d3d5ad5`.

## Reproduction (Windows Edge original UI; no GPU, no external app accounts)

```powershell
$env:PUPPETEER_CORE_PATH = 'C:\Users\Shehr\AppData\Roaming\npm\node_modules\@wonderwhy-er\desktop-commander\node_modules\puppeteer-core'
# SOURCE ARTIFACTS ARE ALREADY FROZEN; ONLY RE-CREATE IN A NEW PATH:
node h58/original_multidomain_dev.js h58/new-original-dev-capture
python -m h58.audit --root h58/assets
python -m unittest tests.test_h58_original_multidomain_dev -v
# Only after obtaining untouched real pinned Qwen GPU generation outputs:
python -m h58.score --root h58/assets --raw PATH_TO_REAL_MODEL_24_RAW.json
node h58/replay_raw_actions.js h58/assets PATH_TO_REAL_MODEL_24_RAW.json PATH_TO_NEW_CREATE_ONLY_REPLAY.json
```

This source development evaluation includes separate fictional app families from HCompany H57 original TRAIN; however its **public availability means it is not a sealed independent test**. Actual independently sequestered, disjoint applications and genuinely executed neural model raw output/weights are required for reliable competitive claims against Holo4, UI-TARS and OSWorld/ScreenSpot.

## Live engineering/gates

- **[DONE]** All 24 original Edge screenshots and source-author postcondition receipts, including both ordered and unordered directions.
- **[DONE]** Original SHA-backed source auditor, raw model-envelope geometry score and genuine Edge replayer. Real Edge positive/reverse **synthetic unit fixtures** succeeded 24 and 16 as expected; seven focused CPU tests initially passed.
- **[NOT DONE]** Real GPU inference for the 24 H58 mixed-domain screenshots, real LoRA gradient training, a new actual Hala model checkpoint, measurable neural improvement, release qualification, a blinded benchmark or a competitor victory. H47 original Kaggle queued task remains independent.
- **[DONE]** Real Windows source suite **693 total = 691 PASS, two existing SKIP, ZERO failures**, 130.700s, exit code 0. Original raw `reviews/h58-full-source-suite-20261021.log` SHA256 `6e0e447e67e2f293af5af7919457521a8c5aebc2175b374bd8807faac3f9f1c7`.
- **[DONE]** H58+H57+data-library focused **18/18 PASS**, original `reviews/h58-focused-source-and-library-20261021.log` SHA256 `e97f0ad4510876ed1f5b21b12afebafa897271715fa2d50638479563ff762333`.
- **[DONE]** Graft **wiring-only** build/check PASS, **269 files/2,162 nodes/5,725 edges**, check SHA256 `b742a8ab4a230f354bbc697e440db8c188434644778dbf77b2740b2767a6da10`. Deep semantic Graft NOT run.
- **[PREVIEW ONLY]** Alibaba OpenCodeReview local offline file delegate preview **35 reviewable/33 excluded**; `reviews/h58-alibaba-offline-preview-20261021.json` SHA256 `8f4f6e7bc51a093ecd70f01f07e7af85a48f539bfc017b62b5547250c3343666`. This is NOT an LLM model-backed semantic review. D1/PStack independent review NOT done.
- **[PENDING]** Signed new GitHub commit and exact new-head Python 3.11/3.12 CI results; do not claim hosted approval before terminal results.
