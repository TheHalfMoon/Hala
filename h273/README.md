# H273 — Qwen3-VL normalized GUI coordinates vs native pixels

**Prospective original first-party research only. No new learned adapter,
no independent OOD and no model release.**

## Evidence motivating this independent engineering test

Previous private free-T4 H268–H271 genuine Qwen3-VL screenshot
predictions generated strict JSON browser actions but consistently
missed the first Stockroom `+1` button (H271 verified **0/2 exact
button hits** in both pretrained and vision-only trained arms). All
prior pilots used *absolute* 960×720 pixel point coordinates.

The **Qwen3-VL developer documentation instead describes native
2-D object-location coordinates on a relative 0–999/1000 grid**,
with a resolution-based conversion back to pixel positions. This
is a testable possible **coordinate-frame mismatch**, NOT a
proven diagnosis and NOT evidence that prior model predictions
were numerically correct. Official reference:
https://github.com/QwenLM/Qwen3-VL and
https://www.alibabacloud.com/help/en/model-studio/vision

## Pre-registered A/B and strict privacy

Run the SAME unadapted pinned pretrained
`Qwen/Qwen3-VL-2B-Instruct` revision
`89644892e4d85e24eaac8bacfd4f463576704203`
once per arm and scenario, four genuine image-conditioned
`model.generate` calls in the private remote FREE Kaggle T4.

- Scenario A and B: unchanged original H262 first-party Arabic
  Stockroom `ar_sorted_tape` and `ar_sorted_gloves`,
  selected BEFORE looking at new GPU outputs. Only exact
  same screenshot PNG pixels are admitted to the paired
  pixel/relative arm comparisons (verified privately in RAM).
  This first-party source and research task are **previously
  exposed** and NOT a new independently blind evaluation.
- Arm 1 — `native_pixel_original`: unchanged original
  H199 action prompt, H268 model-time strict grammar
  `x=0..959, y=0..719` in absolute 960×720 screenshot
  pixels, H199 strict validation, H264 screenshot freshness
  check, H263 real browser actuation. No conversion.
- Arm 2 — `qwen_relative_0_999`: prospectively ask the
  pretrained Qwen model for coordinates on its documented
  normalized 0..999 full-image grid, with a distinct fixed
  token-level grammar accepting exactly `click` or safe
  `abstain` and no Markdown/extra keys. **Before native
  actuation**, apply a deterministic trusted coordinate
  frame conversion (NOT an oracle correction):
  `pixel_x = min(959, floor(x_rel * 960 / 1000))`,
  `pixel_y = min(719, floor(y_rel * 720 / 1000))`.
  The unchanged H199 parser still validates the converted
  exact native JSON before H264/H263 can execute it.
  Neither target DOM rectangle nor any reference point
  participates in this mapping.
- Original first-party target DOM geometry is inspected
  **only AFTER** model inference and native input, to report
  a coarse correct-FIRST-button hit boolean and true inventory
  increment. The model-chosen raw or converted points,
  raw response, full screenshots, page gold, and model weights
  are NEVER persisted. Refusal or malformed output = failure;
  no posthoc Markdown-fence stripping or hidden fallback.
- All four actual pretrained model generations and both
  token-level grammar calls must complete. A report records
  base vs relative per-scene dispatch, target-hit and true
  task success, still explicitly denying trained improvement,
  independent blind OOD and data admission. Coordinate-frame
  success would be **inference-interface engineering
  improvement**, not a trained new model or generalizable gain.

## Owner-private Kaggle FREE T4 and source reproducibility

Public first-party code and invented-only tests:

- `h273/relative_native_bridge.py`: strict normalized schema,
  deterministic frame conversion and fixed normalized policy
- `h273/private_relative_probe.py`: genuine pretrained paired
  original GUI/real browser research, no training
- `h273/build_private_payload.py`: single Python script
  combining the two checked-in public text modules with
  private-free-T4 pinned dependency installation
- `tests/test_h273_relative_native_bridge.py` and
  `tests/test_h273_private_relative_probe.py`: no GPU,
  no screenshot, no personal/gold data; fail-closed schema,
  conversion, identical-image and no-accuracy-laundering tests

Original Git source pinned at H272 commit
`6091a13497881774b41fc6cc1244d2b0e882c0fe`
and original firstparty Stockroom HTML pinned SHA256
`3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a`.
Pinned Transformers 5.16.1 and source-verified
MIT LMFE core `817f944fcc8851917c40300a47f62d1defc3ffc3`.
All Qwen weights/source and real screenshot bytes in
Kaggle ephemeral `/tmp/hala-h273-relative-point`,
deleted at exit. The ONLY intended Kaggle output is the
small owner-private sanitized aggregate JSON
`hala-h273-private-normalized-vs-native-aggregate.json`.
Never run Kaggle output without its exact `--file-pattern`;
never download full cached pretrained weights.

Owner Windows CPU/source verification: 147 relevant PASS
(3019 unrelated tests deselected), H273 optimized Python
6 PASS, compileall success. None proves model correctness.

Create code-only private upload on owner-authorized Windows:

    python -m h273.build_private_payload --source h273/private_relative_probe.py --bridge h273/relative_native_bridge.py --destination D:\Hala-H273-Private-Upload
    kaggle kernels push -p D:\Hala-H273-Private-Upload --accelerator NvidiaTeslaT4
    kaggle kernels status abdulazizshehri/hala-h273-private-relative-pixel-comparison-t4

The same previous freeze and safety gates still apply:
H199 ADMITTED_TRAIN_MANIFEST_SHAS EMPTY; consumed H173
32-case scored OOD permanently SEALED; H212 barred;
H250/H251 third-party visual/asset histories not admitted;
H262 only exposed author-owned one-family research.
No public adapters/weights/release without lawful
source-independent TRAIN/DEV, new sealed blind OOD
and genuine paired native task-state improvement.


## Verified real private T4 v1 A/B (2026-10-31)

Owner-private Kaggle v1
\`abdulazizshehri/hala-h273-private-relative-pixel-comparison-t4\`
**COMPLETED** on two actual FREE NVIDIA T4 GPUs. Read only exact
allowlisted **2,655-byte sanitized private aggregate JSON**
\`hala-h273-private-normalized-vs-native-aggregate.json\`;
no raw model text, gold points, screenshots, original source media,
weights or adapters exported.

Four actual pretrained Qwen3-VL visual \`model.generate\` calls
and four genuine model-time token-filter generate invocations
completed. Both paired scenarios used identical image bytes
across pixel and relative arms.

| Result | Native pixel arm | Qwen-relative 0..999 arm |
| --- | --- | --- |
| Actual completed image-conditioned Qwen generations | 2/2 | 2/2 |
| Model-time token filter applied | 2/2 | 2/2 |
| Valid H199 native format after prospective conversion | 2/2 | **0/2** |
| Actual native browser clicks | 2/2 | **0/2** |
| Correct FIRST +1 button bbox hits | **0/2** | **0/2** |
| Actual source-owned inventory task-state success | **0/2** | **0/2** |

Per-scenario original firstparty Arabic \`ar_sorted_tape\` and
\`ar_sorted_gloves\`: absolute-pixel arm produced a strict \`click\`
and dispatched native input, but missed the target. Relative arm
completed a real constrained model generation, yet the resulting
string **did not pass the preregistered strict normalized format
gate**; no converted action or browser input was executed.
As a result, relative hit/task scores **must NOT be interpreted as
visual-coordinate localization accuracy**. These are format-gate
failures upstream of actuation, not evidence of a spatial miss.

**No useful GUI gain is demonstrated by the normalized arm in
this run**, and this experiment neither proves nor falsifies
coordinate-frame mismatch as an explanation of baseline misses.
Do NOT introduce posthoc Markdown cleanup, relaxed schemas,
oracle-derived points, invented results, or reinterpret these
previously exposed firstparty scenes as a sealed blind OOD test.
A next experiment may diagnose coarse shape/termination classes
in an owner-private aggregate only, with no raw model text or
points saved, before pursuing meaningful spatial localization.

H273 source commit
\`bd7584ea0265c3a0d5c43084fd468a963becefc6\`
[hosted GitHub Actions #41205031424](https://github.com/TheHalfMoon/Hala/actions/runs/41205031424)
**COMPLETED SUCCESS** on both Python 3.11 and Python 3.12.
This is source-only qualification, not native GPU success.

Permanent gates remain unchanged: **0 new optimizer steps**,
no officially admitted TRAIN/DEV/OOD, no new independent blind
OOD, no generalizable Hala trained skill proven, and no weights
or release published.
