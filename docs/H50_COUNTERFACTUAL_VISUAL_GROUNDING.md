# H50 — Counterfactual Six-Way Visual Grounding (2026-10-21)

**Research experiment, English source/agent prompts. No GPU used; NO new model weights, validation accuracy, official benchmark score, clinical access or government account actions.** H50 is a bounded training-data quality improvement on top of the existing H49 Draft PR #37 and preserves the H14/H41/H46/H47 freezes. The earlier original H46 Qwen3-VL-2B DEV outcome remains **0/8 correct**.

## Motivation: learned grounding instead of more non-learning wrappers

Original H49 provided 64 unique TRAIN screenshots, each with six plausible visible controls but only **one chosen instruction/target training label**. A model could shortcut toward memorized button positions or layout/template priors rather than attending to the *request*. A counterfactual same-image experiment exposes **six different natural language goals with six distinct target click points for each original image**; a rule that always clicks the same region can be correct on at most one of these six labels, if the verifier's original control rectangles are disjoint. The dataset does not prove a neural model learns the distinction until a real adapter is trained and independently tested.

**H50 source-defined split and provenance:**

| Source | Distinct original PNGs | Grounding TRAIN rows | English | Arabic |
|---|---:|---:|---:|---:|
| H49 original four TRAIN families only (judicial services, traffic, FHIR resources, FHIR bundles) | 64 | 384 (64 images x six *different* goals) | 192 rows on 32 images | 192 rows on 32 images |
| H49 public DEV families (municipal portal and research lab) | 32 | **0** | 0 | 0 |

**These are 384 annotations, NOT 384 independent images.** No images from ScreenRef, OSWorld, ScreenSpot-Pro, Najiz or Absher were incorporated. H50 is CC0-1.0 first-party original fictional offline data derived byte-for-byte from previously captured H49 mock browser images, not from authenticated accounts or patient screens.

## Evidence-bound implementation

- `h50/counterfactual.py` re-audits the original H49 source, all 96 original screenshots and browser captures via `h49.audit.inspect`; derives six goal/target pairs per distinct TRAIN image using only the original six unique measured DOM rectangles and known first-party author labels. It validates every target point is uniquely inside its intended visible control, emits **384 source-bound train-only JSONL rows** and a rights/source/capture/media SHA256 manifest.
- The source PNGs are **copied byte-for-byte** into `h50/assets/png` for the existing remote SFT loader's single-root relative media contract; no image is synthesized or recolored and no DEV image is present. The H49 original 64-row SFT bytes and original source remain untouched.
- `tools/h50_six_click_replay.js` uses original local H49 HTML files with pinned `puppeteer-core 25.12.0` and installed headless Microsoft Edge. It genuinely clicks each of six original controls on every one of 64 H49 TRAIN mock screens, checks in-browser `dataset.selected` and running `selectionCount` **after each of the 384 DOM clicks**, rejecting offscreen controls, mismatched SHA256 and network requests. The original receipt is `h50/assets/click_replay.json`. These are **offline browser-mock click postconditions**, not 384 correct neural model predictions or real tasks on a government website.
- `h50.counterfactual.load_export` refuses incomplete/rewritten source, replay, manifest, media, target or ordered labels, even a coherently rehashed fraudulent completion. The already established `training/scripts/train_adapter.py` now admits only the exact H50 source schema and loader, preserving H49/H48 paths and frozen H14 artifacts unchanged.
- H50 uses the **same exact user-facing prompt and JSON window-pixel click completion schema** as H49. That permits a future *same-prompt, same-model, same-hardware* ablation: unchanged Qwen3-VL-2B, adapter trained with original H49 64-label data, and adapter trained on H50 384-label counterfactual data. All three arms require authentic checkpoints, external GPU/source evidence, honest errors and frozen identical **public DEV** evaluation before any comparison. A prospective **new independently sequestered application-disjoint TEST set** remains separately required.

Original manifest hashes:
- H50 384-annotation training JSONL SHA256: `cb1a8529eaf0f82272d75a459c1634766ee5ec7694f6b21affa2045b11d02dd6`.
- Original 384-browser-click receipt SHA256: `92b3f6112c75fbed192e0a4fec351d237c1a486ac9ab650dd89e63760d6d7afb`.
- Original H49 source and capture are transparently pinned within `h50/assets/sft_manifest.json`, `h50/assets/source_provenance.json` and all copied media digests. Never overwrite the original browser source and receipts; author a new experiment/split for successors.

Repro on an authorized **offline Windows development browser** checkout (generates assets in a new, nonexistent root only):

```powershell
python -m h50.counterfactual --root h50-new/assets
# Tool script currently uses the ORIGINAL published h50/assets and h49/assets
# layout; copy script/adjust only for an independently reviewed new experiment.
$env:PUPPETEER_CORE_PATH='PATH_TO_PINNED_INSTALLED_PUPPETEER_CORE_25.12.0'
node tools/h50_six_click_replay.js .
python -m unittest tests.test_h50_counterfactual -v
```

Do not re-run `node tools/h50_six_click_replay.js` on the original `h50/assets` because the receipt is **create-only**.

## Competitive learning roadmap, licensing and honest gates

The recent 2026 GroundGUI research has an instructive curriculum: text recognition, icon/visual matching, spatial layout reasoning and refusal; subsequent SFT plus post-training stages. The authors report GroundGUI-8B 63.9% ScreenSpot-Pro and 95.2% ScreenSpot-V2; these are **vendor/paper reports, not a verified Hala comparison**. Primary sources:
- GroundGUI code (Apache-2.0 code, inspect per-file NOTICE): https://github.com/PrentisAI/GroundGUI
- GroundGUI-8B model card (Apache-2.0 weights, version-specific due diligence): https://huggingface.co/PrentisAI/GroundGUI-8B
- GroundGUI paper: https://arxiv.org/abs/2610.19100
- **Do not import ScreenRef training images or annotations into a generally releasable commercial Hala model** without resolving the separate restricted academic-research license. It limits redistribution/commercial use and models trained on the data: https://huggingface.co/datasets/PrentisAI/ScreenRef
- Holo4 35B-A3B publicly documented Apache-2.0 checkpoint remains Hala's designated long-horizon architecture/research target; H50 is a practical Qwen3-VL-2B experiment candidate, **not a silent Holo4 model substitution**: https://hcompany.ai/research/models
- Official OSWorld 2.1 release pairs model-independent benchmark code/assets/tasks/web references pinned **within one matching release**; prevent training contamination and don't compare OSWorld, OSWorld-G and ScreenSpot percentages as the same metric: https://github.com/xlang-ai/OSWorld-V2

**Next high-value new data and training work, in order:**
1. Check a genuine no-charge licensed free Colab GPU session. Freeze a clean original baseline and exact H49/H50 checkpoint protocols; do not bypass the existing Kaggle training ban or any required source license/approval.
2. Real H49-vs-H50 adapter experiment under identical frozen base, processor, hardware, optimization budget, seed and prompt, with adapter update metrics, raw model outputs, failure/latency/VRAM/cost evidence. Only actual observed train losses+blinded validation may support neural improvement.
3. Expand independent first-party **unique** screenshot contexts and hard icon/spatial/absent-target examples without recycling exposed public DEV or official benchmark answers. H50 itself is ONLY text-goal contrasting, **not** icon/refusal or multi-step learning.
4. Qualify with independently isolated bilingual app-disjoint TEST, then official version-matched ScreenSpot-Pro and OSWorld 2.1 when lawful free evaluation compute can be allocated. Count actual successful tasks, wrong-target and recovery, end-to-end sequence completion, latency, RAM/VRAM and per-successful-task cost.

**NO-GO remains:** no qualified H50 adapter exists, no real model training was performed on these 384 labels, no official evaluation performed, no independent blind TEST ready, and no competitor-win claim is permitted. Keep H50 in existing Draft H49 PR #37 until actual model, license and reviewer gates pass.

**Instruction variety audit (honest scope):** 384 labels use 64 unique original PNGs but only **24 unique natural-language prompt strings** across four TRAIN application families (six distinct goals per family), repeated on differently ordered/layout-varied images. This is a **counterfactual target-disambiguation improvement**, not a large diverse vocabulary corpus or proof of generalization to completely unseen task language. Follow-on H51 must author genuinely independent additional Arabic and English paraphrases and unique layout/icon/spatial/absent-target app scenes, and evaluate exact held-out instruction families before any performance assertion. Original total H50 evidence files (68 across 64 copied PNGs and four source/receipt/meta files) occupied ~2.12 MB in this Windows checkout; no local model weights were downloaded.

## Actual H50 engineering qualification receipts — 2026-10-21

The following are **real authoring/browser engineering tests**, not model evaluation:
- Original six-way browser replay: `64 images × 6 different keys = 384` real **offline mock-browser DOM clicks**, expected selected keys and click-counter postconditions. Original `h50/assets/click_replay.json` SHA256 `92b3f6112c75fbed192e0a4fec351d237c1a486ac9ab650dd89e63760d6d7afb`, with no online government/EHR service access and no neural inference.
- H50 384 source-bound SFT labels: `h50/assets/authoring_sft.jsonl` SHA256 `cb1a8529eaf0f82272d75a459c1634766ee5ec7694f6b21affa2045b11d02dd6`. Original 64 image media files copied byte-for-byte from H49, no other images, no test/DEV leakage.
- Genuine Windows Python focused H50 audit: **8/8 PASS**, exit 0. Original raw `reviews/h50-focused-20261021.log` SHA256 `f96c43ef10219a6d3ebd1fc38b5c99dc44e048e6556325cc999862f37efa7e16`. Initial isolated-temp test had a **test-fixture directory bug**, documented by the original live terminal failure; the source/data code was unchanged and fixed test passed. The original failed transcript was not persisted because the first command wrote over the same transient draft log, but the failure is not misreported as a success.
- Full Windows serial Python source suite **611 run, 609 PASS, 2 existing SKIP, zero failures/errors**, exit code 0, duration **299.846 s**. Original `reviews/h50-full-suite-20261021.log` SHA256 `4d0745b387bd688b54a3aca7c0d8ce2ec518f13e6277fae4eb13e4380329bc53`.
- Genuine Graft **wiring-only** `build` and `check` PASS **1,908 nodes, 5,146 edges, 232 files**. Original `reviews/h50-graft-build-20261021.log` SHA256 `c5ee2c69fd06aef451e59c27a6391d12d4f3a071a2d3814bd035b29cbbeb7bf1` and `reviews/h50-graft-check-20261021.log` SHA256 `4bf98577b2d86378f87bc74513f8d479209c101ce3cc8ab761e4b5c68a1424dc`. No deep/semantic graph layer.
- Local Alibaba `open-code-review v1.13.2` delegation **preview only** reported three reviewable H50 code files and 76 excluded files. No actual model-backed local semantic review or independent verdict; details in `reviews/h50-alibaba-delegation-accounting-20261021.md`.

**CI and governance:** All above qualification evidence must be rechecked against the eventual exact GitHub signed H50 source HEAD on hosted Python 3.11 and 3.12. Until all required checks succeed, this is **locally source-qualified / hosted-CI pending**. Keep existing stacked PR #37 Draft and unmerged, H47 existing private Kaggle notebook unchanged, zero paid compute, zero new neural training and zero benchmark advantage claims.
