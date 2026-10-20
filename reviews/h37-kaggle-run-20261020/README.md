# H37 — actual private Kaggle browser-rendered GUI baseline, 2026-10-20

**Study type:** first-party rendered browser UI, research-only, NOT customer/app screenshots, no native OS actions, no approved production model. Raw results are retained without edits. **Frozen source SHA:** `4c061e9c2e15df0306062b5a5666c6e0d7697502`, signed and verified in [source Draft PR #19](https://github.com/TheHalfMoon/Hala/pull/19). Keep its source branch pinned so the submitted notebook remains reproducible.

## Authentic execution, environment, and integrity

- Private Kaggle Batch notebook: https://www.kaggle.com/code/abdulazizshehri/hala-h37-browser-grounding-dev-20261020 (v1). Worker state `completed`; 64/64 actual model attempts, worker exit `0`, deadline not exceeded, **134.34 seconds** worker-reported duration. Local model input generation and external SHA audit conducted separately.
- Model: `Qwen/Qwen3-VL-2B-Instruct`, immutable revision `89644892e4d85e24eaac8bacfd4f463576704203`; one FP16 model load on observed Tesla T4 cuda:0. Observed package pins: torch `2.11.0+cu128`, transformers `5.16.1`, accelerate `1.14.0`, safetensors `0.8.0`, Pillow `12.3.0`. Hardware/provider metadata is not independently attested.
- Actual locally independently computed `raw_attempts.jsonl` SHA256 **`40b88ceb592446630817b9562238b40765b371fac61ccb4c06b427fab92cef1d`**, matching Kaggle worker console. Strict independent `h37.audit` report `audit.json` SHA256 **`987c8338342faaf8d2718288623a9f485d058acdde0f21f5f3133befaa1451f2`**.
- Actual notebook submitted bytes are retained under `submitted-bundle/`; notebook SHA256 `5062e90d12a5231638b8d2982b886b87fb7a0b2ef8e26c2b0d1dfafb81e35206`; private kernel metadata SHA256 `478570663a0b07c58ac94a11bb9b4a33484e2c559b6ec41d5215534fae95f3d6`. Archived Kaggle stdout/stderr `run.log` SHA256 `c4de558e1068a340a66e5c7845ea041367e787104f3bfd4b17511494ca4031e6` (pre-commit credential pattern scan zero matches).
- Original **64 Edge Chromium rasterized local HTML/CSS screenshot PNGs**, DOM/evaluator oracle and model-only input hashes are in `h37/assets/`, not duplicated here. H37 source pins: scene `8064af96a3fd4c2389e1c494829a56549bacd5a21ea2cc20f9fef987be5634d2`, capture `f8cb6b6a3b7093b7ce26fdb8ef18fdd50e87a269c25a9b5a8657e62372e956f9`, model inputs `12f4b4615b2da5da5513dab51ffba434d5ccfd9a726ba978d4ed1f50ba7ff00d`, evaluator only `80983a038e9b58191c99ba6254103fda622c5a1224f11348f9895f7c8bde3cd0`.
- The Kaggle worker read only `model_inputs.json` plus PNG bytes; evaluator labels stayed separate and were read only by the CPU audit. **No clicks, training, host control or production deployment**.

## Actual strict-target results on model output

| Metric, unconditional | All 64 | DEV, 32 | Public authored disjoint-domain TEST, 32 |
|---|---:|---:|---:|
| Strict valid JSON | **64** | 32 | 32 |
| Intended button hit | **32** | **14** | **18** |
| Correct unique visible target binding | 32 | 14 | 18 |
| Wrong visible control | **16** | 11 | 5 |
| No unique visible control binding | **16** | 7 | 9 |
| Explicit abstention | 0 | 0 | 0 |
| Backend error | 0 | 0 | 0 |

| Stratification | Intended hits | Wrong visible controls | No unique control binding |
|---|---:|---:|---:|
| English (32) | 18 | 8 | 6 |
| Arabic (32) | 14 | 8 | 10 |
| Toolbar (16) | 9 | 0 | 7 |
| Sidebar (16) | 6 | 10 | 0 |
| Grid (16) | **3** | 4 | 9 |
| Table (16) | **14** | 2 | 0 |
| Makerspace (16) | 7 | 6 | 3 |
| Community theater (16) | 7 | 5 | 4 |
| Trail maintenance (16) | 9 | 2 | 5 |
| Library workshops (16) | 9 | 3 | 4 |

This is **50% (32/64)** correct on new authored browser-rendered examples. The difference from 8/32 on H34 is **not a controlled neural improvement**: layout, rendering, domains, labels and control geometry all changed. The large grid/table gap is an exploratory hypothesis, not a claim of universal robustness. All 64 images and correct answers were authored/published by this project; TEST is disjoint in domain only and is **not an independent, blind, real-world test**. The model stays unqualified.

## Replay offline using frozen source

```powershell
python -m h37.audit --output reviews/h37-kaggle-run-20261020 --expected-raw-sha256 40b88ceb592446630817b9562238b40765b371fac61ccb4c06b427fab92cef1d
python -m unittest tests.test_h37_browser_grounding tests.test_h37_kaggle_protocol -v
```

One added actual-evidence regression enforces exact raw SHA and the above outcomes without replacing the original scorer or granting extra tolerance. Local synthetic oracle fixtures used in older tests are **not model results**.

## Remaining gating work

Independently gathered real application pages, right-to-left screenshot/semantic validation, third-party Arabic expert review, genuinely blind held-out tasks, better controlled and reliable visual grounding with safe uncertainty behavior, H23 actual CUDA fault/cancel/worker-restart verification, independent code review and founder production-release decision. No claims of completion or deployment until those gates are established. This evidence PR remains **Draft** and must not merge into main without governance.
