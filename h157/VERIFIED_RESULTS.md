# H157 — independently verified true neural native browser improvement

**Status, 2026-10-27:** H157 external research Qwen3.5-2B
language-LoRA SFT **COMPLETED**, with a SHA-distinct checkpoint and a
large, independently verified gain in **same-app original MiniWoB++ DEV**
native task completion. These are NOT unseen-app, H140 sealed blind, or
Hala flagship 35B results.

## Private data/model provenance

- Original external application: **unchanged MIT** Farama MiniWoB++
  commit `7fd85d71a4b60325c6585396ec4f48377d049838`.
- Source/reference: 352/352 original native Chromium actions completed
  with positive original-app reward; **280 distinct TRAIN** and **72
  distinct DEV** screenshots; 5 action/layout families; original PNGs
  unique and disjoint from 146 previous H154–H157 smoke/source images.
- Frozen original source manifest SHA-256:
  `a28b845b212ba7b745f1b155944174278b8aac69e69ad0f0eee3ca3acd8b7f78`.
- Frozen trained starting H156 research LoRA SHA-256:
  `732f54893172262348f7ef802a3d645626d6721f2214f28944740f0f07c91ac6`.
- New H157 research LoRA SHA-256:
  `181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8`.
- Kaggle private FREE Nvidia T4 **152/152 actual optimizer steps**,
  with two supervised examples per step: 280 unique TRAIN, plus the
  24 new dropdown TRAIN examples repeated once for retention
  (304 weighted examples total). **0 DEV in optimizer**.
- Exact real GPU receipt, updated safetensors SHA and 72 same-app DEV
  raw model outputs for both before/after checkpoints were independently
  re-parsed and reconciled to source SHA and optimizer trace.

## Genuine neural outcome in original Chromium

Before the neural action run, all **72/72 frozen DEV screenshot SHA-256
and visible instructions** reproduced exactly from their original seeds
using the pinned upstream app. Then model-generated full typed actions
from both checkpoint versions were executed **once each in separate
original Chromium contexts** with native mouse and keyboard only.
The reference action DOM/oracle was never made available to the
neural model or the execution controller. Original task reward was read
only after model execution; all 144 episodes completed.

| Original task | H156 baseline native success | H157 trained native success |
| --- | ---: | ---: |
| `click-test` | 8/16 | **13/16** |
| `click-button` (named button/distractors) | 2/16 | **9/16** |
| `enter-text` (focus/type/submit) | 3/16 | **14/16** |
| `enter-text-dynamic` (moving controls) | 3/16 | **16/16** |
| `choose-list` (native dropdown) | 4/8 | **8/8** |
| **TOTAL: real original-app reward** | **20/72 (27.8%)** | **60/72 (83.3%)** |

On **identical paired DEV tasks** the learned H157 adapter improved
**40/72 tasks (55.6 percentage points)** overall: **41 H157-only wins**,
**1 H156-only regression** (`click-test`), 19 both succeed, and 11
both fail. Independent exact two-sided McNemar/sign conditional test:
`p=1.9554136088117957e-11`. These results support meaningful
**within-MiniWoB same-app DEV** improvement, not a cross-app accuracy
claim. The DEV cohort is now **consumed and must not be rerun/tuned on**.

Full model-generated action format remained **72/72 correct JSON
sequences for each checkpoint**, so this gain cannot be attributed to
merely fixing the JSON parser. Correct text values improved from
**39/40 to 40/40** on DEV text-requiring tasks; the main gain is native
spatial targeting and action execution.

## Independent controls and known limits

- GPU audit: `h157/audit_gpu_learning.py` validates the new 152-step
  training receipt, training losses, checkpoint SHA, two model arms,
  image SHA and every raw action prediction.
- Original app DEV identity: `h157/preflight_native_identity.py`
  verified 72 exact source screenshots before any model action.
- Native runner: `h157/replay_model_dev.py` created one owner-private
  consumed lock and **144** original browser episodes.
- Independent reward audit: `h157/audit_native_model_gain.py` checks
  72 exact paired original task IDs, no duplicate/unknown tasks, each
  action-family stratum, per-case positive app reward, before/after
  summary matching, and exact paired test.
- Owner-private native result SHA-256:
  `9b295caafefb0d7b016646bddb8257f932839aa0aec203660711e3ca5a7ac8ba`.
- Research data, screenshots, weights, predictions, episode details,
  reward receipts and model private Kaggle outputs are **NOT** pushed to
  GitHub. Only source/test code and aggregate outcome evidence are.
- **62/62** focused H150–H157 Windows unit tests PASS at last local
  check; GitHub CI has passed on preceding H157 evidence/runner head.
  Check CI again on any new documentation commit.

**Limitations:** This is still just one originally licensed benchmark
web app with five task families, currently English-only and 72 same-app
DEV examples. H157 model was fine-tuned to tasks from that source
distribution, so do not claim performance on unrelated sites,
real enterprise workflows, Arabic, or the Hala 35B flagship.

**Next H158:** Construct a genuinely fresh, licensed second-application
native-action benchmark not derived from MiniWoB, freeze its SHA before
any model run, and use only inference/no optimizer on that holdout.
Only afterward consider further independent training; do **not** use
H140 sealed blind or consumed H156/H157 DEV as new evaluation data.

No paid GPU, H140 BLIND access, model promotion/publication, repository
main merge, or release.
