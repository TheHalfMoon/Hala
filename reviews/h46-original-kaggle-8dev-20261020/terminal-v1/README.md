# H46 v1 — terminal actual Qwen3-VL-2B public DEV findings (2026-10-20)

**Original provider result:** private [Kaggle H46 v1](https://www.kaggle.com/code/abdulazizshehri/hala-h46-eight-original-bilingual-dev-screenshots) status `COMPLETE` on a genuinely allocated Tesla T4 (15,636,037,632 VRAM bytes). Frozen revision `89644892e4d85e24eaac8bacfd4f463576704203`, `Qwen/Qwen3-VL-2B-Instruct`; historical H34/H37 prompt and precisely pinned runtime: torch `2.11.0+cu128`, transformers `5.16.1`, accelerate `1.14.0`, safetensors `0.8.0`, Pillow `12.3.0`. Worker exit `0`, `state=completed`, **8/8 model attempts retained**, elapsed worker **70.004 seconds** inclusive of cold model load and inference; parent worker exit `0`, approximately 71.783 seconds. No model training, native UI actions or external benchmarks occurred.

## Independent score — no exclusions

Independent CPU evaluator `h46.experiment.replay` ran from the previously signed H46 code and SHA-checked **original provider** `raw_attempts.jsonl`, exact H44 inputs and evaluator-only DOM bounds. **8/8 parsable point JSON, 0/8 intended unique control correct**. **3/8 uniquely bound to the wrong control; 5/8 unbound to any unique visible control; 0/8 abstain and 0/8 runtime errors.** By language: **English 0/4** intended correct, 2 uniquely wrong; **Arabic 0/4**, 1 uniquely wrong. Original denominator is always 8; no success-only pruning. Score `0/8` is **a genuinely negative model result**, not a failed simulator or a failed attempt to load a model. The model is **not qualified**.

Observed response points: `[0.5,0.5]` twice (English harbor grid), `[0.5,0.45]` twice (Arabic harbor grid), `[0.25,0.75]` twice (English summit table), `[0.5,0.58]` and `[0.5,0.65]` once each (Arabic summit table). The exact `[0.25,0.75]` vector occurs in the model instruction JSON example and in two raw outputs, a **plausible but unproven prompt-coordinate anchoring effect**; a prospective equal-task paired experiment is necessary to establish causality. This study is too small to make any broad claims about Arabic-vs-English or relative model quality.

**Important scope:** eight real Edge screenshot PNGs of first-party fictional apps, all public synthetic DEV, 4 EN/4 AR; H37's 32/64 uses different screens and source tasks and **cannot be directly compared**. No independent real third-party app, OSWorld/ScreenSpot, blinded sealed task family, multi-step model actions, independent human adjudication, H23 CUDA or Holo4 comparison. The Kaggle COMPLETE flag is separate from model success.

## Immutable original and derived evidence (do not rewrite)

| File | SHA256 |
|---|---|
| `raw_attempts.jsonl` — original 8 Kaggle model outputs/errors | `a23923be51fddfb4e6b876e5cdf842e5827a507147c640d3370767d9cfe45a2c` |
| `run_metadata.json` — actual worker, hardware/model, source and pins | `58f12899778ee777113b57f7d2f22ab65e1ed07a711c41cdfd6d7df5c37da48e` |
| `h46-parent-exit.json` — original Kaggle supervisor | `4392e2b57ed116e259f1556ce4073c0d73968f08889aba62ee0bdb9a9c0bd674` |
| `independent_replay.json` — derived separately from original raw SHA and original evaluator-only H44 source | `68cb5360769b735d9cbad5c8922fdc320983d43e6ace255bd0d815123752b9b0` |

All these retained files were copied **without content edits** for provider originals, then rehashed. No cloud notebook rerun or paid resources. Original Kaggle notebook and explicit consent before run are retained in the sibling `submitted-bundle/` of this PR. The run's original checked source commit is `eb049f489524ff188d4d326ab7efe3b6f9ce6b65`; H41 H23 failed-frozen-font lifecycle remains a separate negative result.

## Next experiment, not an asserted fix

A paired H47 study should compare **the unchanged original prompt** against one carefully pre-registered prompt **without embedded numeric coordinate example** on **the same eight H44 DEV images**, **in the same free Kaggle VM with frozen model**, counterbalanced order and exact original numerical point parser. Record all 16 neural raw outputs, GPU and generation budget, and use an independent evaluator-only replay. This probes example anchoring, but is not model fine-tuning, independent blind test, or evidence of generalization; no global improvement claim can follow from DEV alone. A genuinely different visual target or model variant must eventually be compared against an independent app-family-disjoint evaluator.
