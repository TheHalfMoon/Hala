# H271 — private vision-only LoRA with true first-button hit diagnosis

## What changed

The verified free-Kaggle T4 studies H269 and H270 genuinely optimized
Qwen3-VL-2B language-only LoRA, yet failed actual Stockroom task-state
evaluation: H269 BASE 0/1 and ADAPTED 0/1; H270 coordinate-weighted,
eight original training screenshots and 8/8 real optimizer updates
still BASE 0/2 and ADAPTED 0/2. Merely boosting coordinate token loss
did not establish visual grounding improvement.

**H271 prospectively changes the TRAINABLE MODULE scope, not the scorer
or heldout cases.** It reuses Hala's pre-existing H96 exact late-vision
scope validator in training/scripts/train_adapter.py:

- Real Qwen3-VL vision blocks **20–23**, each attention fused qkv and
  projection: exactly **8 checked Linear layers**, rank-4 LoRA,
  **16 trainable A/B adapter tensors**. Zero language-model parameters
  trainable; fail closed if any unexpected target matches.
- Same first-party H262 original HTML source (SHA256 pinned in source),
  8 actual English/Arabic TRAIN screenshots and original training
  labels, completion-only mask, 8× coordinate weighted loss, 8 bounded
  FP16 optimizer updates, and conservative Hala GradScaler init_scale=1.
  Nonzero finite gradients and actual changed adapter tensors are
  REQUIRED every update; no fake/CPU training.
- True frozen pretrained Qwen3-VL and adapted vision LoRA are paired
  on the SAME two existing Arabic Stockroom search/sort research scenes
  (Paper tape, Exam gloves) with EXACT pixel equality before/after and
  byte distinctness from all TRAIN PNGs.
- H208 image-conditioned genuine model.generate, H268 actual
  model-time strict JSON, H264 screenshot integrity/staleness, H263
  one native Chromium input, and independently observed first stock
  state after execution. No reference click fallback; no change to
  native action grammar or original app.
- Extra coarse diagnostic: after an actual real model click, original
  app DOM computes **whether the model-chosen point was within the
  first stock item +1 button's visible bounding rectangle**. This is
  strictly an independent *post-inference* scorer; the model never
  receives page, DOM, rectangle or gold. Raw model output, chosen
  point and target rectangle stay volatile in private RAM only and
  are NEVER saved, printed or exported. The aggregate exposes only
  booleans and summed hit counts, to distinguish wrong screen
  geometry from click-handler/state measurement failures.

A successful hit/stock improvement even on both cases would be only
**previously exposed first-party research**, NOT a genuinely independent
blind OOD or generalizable learned Hala skill. Failure must remain
failure. The prior scored H173 32-case OOD is permanently SEALED and
never used; H212 barred procedural rows and other third-party original
app images not admitted.

## Privacy and execution

Only owner-private FREE noncommercial Kaggle NVIDIA T4, no paid GPU,
no cross-app screenshot upload, no release, no merge. Source is pinned
to H270 Git 4fe9aaac749ec35652eb1b8132167451d9291a6f. Public
pretrained Qwen/Qwen3-VL-2B-Instruct model revision
89644892e4d85e24eaac8bacfd4f463576704203; exact Transformers
5.16.1, PEFT 0.20.0 and LMFE MIT source SHA
817f944fcc8851917c40300a47f62d1defc3ffc3. Preserve Kaggle's
existing CUDA PyTorch; the specific optional torchao0.10.0 repair from
H269 stays ONLY in remote ephemeral environment.

Private runtime stores model weight cache, original app, source-owned
training PNG/labels, optimizer and vision LoRA ONLY under
/tmp/hala-h271-firstparty and deletes on exit. One small
aggregate-only JSON under /kaggle/working is allowed:
hala-h271-private-vision-native-aggregate.json. NEVER use unfiltered
Kaggle output download (could download weights). The public GitHub
PR contains only code, documentation and wholly invented CPU tests.

Reproduce from owner Windows with authenticated Kaggle CLI by creating
a fresh code-only external staging folder:

    python -m h271.build_private_payload --source h271/vision_lora_native.py --destination D:/Hala-H271-Private-Upload
    kaggle kernels push -p D:/Hala-H271-Private-Upload --accelerator NvidiaTeslaT4
    kaggle kernels status abdulazizshehri/hala-h271-private-vision-lora-native-t4

Only AFTER kernel COMPLETE retrieve exact small JSON via --file-pattern,
read sanitized aggregate and delete local copy.

## Never waived

Official H199 ADMITTED_TRAIN_MANIFEST_SHAS remains EMPTY. H173
original 32-case scored OOD permanently CONSUMED and SEALED; H212
scored procedural rows barred; H250/H251 independent other-original
app captures remain private pending historical fork/npm/transitive media
license and PNG/RGBA duplicate rights vetting. H262 app is one
already-exposed original first-party source. No qualification for
release without lawful multiple-original-family source-disjoint
TRAIN/DEV, brand new untouched blind OOD, and a genuinely measured
native GUI benefit.


## Actual owner-private FREE T4 outcome — verified 2026-10-31

The owner-private H271 kernel
(https://www.kaggle.com/code/abdulazizshehri/hala-h271-private-vision-lora-native-t4)
v1 **COMPLETED** on real **NVIDIA T4 x2** (free, noncommercial).
Only the exact filtered 2,806-byte aggregate report
hala-h271-private-vision-native-aggregate.json was inspected. No raw
model text, click coordinates, target rectangles, screenshots, gold,
checkpoint, weights or private media were downloaded/published.

| Verified real inference/training measure | Outcome |
| --- | --- |
| Genuine rank-4 LAST-FOUR vision LoRA optimizer steps | **8/8** |
| Exactly eight visual Linear layers selected | true |
| Trainable vision LoRA A/B tensors | **16** |
| Trainable language LoRA tensors | **0** |
| Each step has supervised coordinate tokens and finite nonzero gradients | true |
| Changed adapter tensors counted across eight steps | **128 total per-step counts** (NOT 128 unique tensors) |
| Source-owned original TRAIN PNGs | 8, 6 EN + 2 AR |
| Two Arabic heldout scenes byte-disjoint from TRAIN | true |
| Same exact heldout screenshot PNG before/after | true |
| Real Qwen generation and model-time strict JSON | true on all four paired calls |
| Arabic sorted Paper tape native task | BASE failed / VISION LORA failed |
| Arabic sorted Exam gloves native task | BASE failed / VISION LORA failed |
| Actual correct FIRST +1 button bounding-box hits | **BASE 0/2, VISION LORA 0/2** |
| Real native clicks dispatched | BASE 1/2, VISION LORA 2/2 |
| Actual GUI task-state success | **BASE 0/2, VISION LORA 0/2**, delta 0 |
| Independent source-family blind OOD / official training admission | false / false |
| Generalizable Hala GUI skill improvement proven | **false** |
| Model or original private media published | false |

The new independent post-inference first-button hit scorer **confirms
the observed failure is model-chosen click geometry missing the first
+1 stock button** in these two previously-exposed first-party
research scenes; it is not caused merely by the JSON contract.
The H271 vision-only adaptation genuinely changed the intended
parameters, but did NOT improve correct hit or task-state outcome.

The BASE Paper tape arm produced a real strict-format generation but no
browser input; BASE Exam gloves and both adapted arms dispatched
clicks, with no correct target hits. Raw model action text and exact
points were never saved; do not infer why the first BASE arm did not
dispatch beyond the known coarse flags.

**Research scope limitation:** This original H262 first-party
Stockroom source and these two Arabic tasks have been inspected
in prior stages. Exact train/eval screenshot byte-disjointness is NOT
an independent source-family split; neither 0/2 nor any eventual
positive result can qualify as new blind OOD.

**Next required scientific work:** use qualified original legally
reusable GUI families (historical fork/npm/transitive visual asset
rights and PNG+decoded-RGBA identity exclusions), TRAIN/DEV distinct
from NEW source-heldout sealed blind OOD, and compare model-based
2-D visual grounding with a causal native browser task-state reward.
Do not keep retuning these same already exposed Stockroom tasks until
they happen to succeed; no production/release claim.

The hosted source-only exact initial H271 CI run
[GitHub Actions #41202254204](https://github.com/TheHalfMoon/Hala/actions/runs/41202254204)
**COMPLETED SUCCESS** for Python 3.11.14/3.12.14 on source head
af102f07fcff150f7429315b22a2239992b87fd7. A subsequent
README-only evidence head requires an independent hosted CI check.
