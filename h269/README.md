# H269 — Genuine free-T4 first-party native GUI LoRA pilot

**Research pilot only, not Hala release or evidence of generalizable GUI
improvement. Never admit this as official TRAIN/DEV/OOD.**

## Why this work is different

H265–H267 independently confirmed genuine real Qwen3-VL image inference,
but unmodified generations produced unusable Markdown-fenced actions.
H268 changed **decoding only**, yielding one exact H199 JSON click that
really reached the native browser on Kaggle T4 x2; however the model
missed the intended GUI stock target (**0/1**). **No Hala weights had
been learned** by H268.

H269 actually attempts a bounded LoRA optimization using **real browser
pixels and original first-party authoring targets**. The previous H15
qualified two optimizer steps of rank-4 FP16 LoRA on another training
source, but did not compare native task-state performance. H269 is
isolated from that work and must not reuse its data.

## Exact pre-registered protocol

One unchanged original Hala-authored, offline Stockroom HTML at SHA256
`3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a`.
Pin original public source to H268 Git commit
`d2de8e264fb2f90fa84a2b50c447bb035e04feef` and check Git tree
clean before reading any source. No other donor apps or media.

- TRAIN = **two genuine browser screenshot scenes**, default English
  and sorted English, newly rendered into ephemeral `/tmp` only.
  Source-authoring DOM button geometry produces exactly one strict
  `click` JSON **TRAIN label** per original screenshot, and is used
  exclusively by the supervised loss code. It is never a policy
  input, fallback answer, test label, or evaluation hint.
- RESEARCH HOLDOUT = fresh **Arabic RTL** first-party Stockroom scene.
  This holdout does not share exact PNG bytes with either TRAIN scene
  (verified using ephemeral SHA checks). The frozen prompt and native
  task are the same for base and adapted model, and **the exact
  screenshot bytes must match** across paired runs. Reference DOM
  inventory count is inspected only **after** each model action.
- BASE = the real pretrained Qwen/Qwen3-VL-2B-Instruct pinned revision
  `89644892e4d85e24eaac8bacfd4f463576704203`, with H268 real
  model-time constrained JSON, H208 visual processor, H264 stale
  screenshot lock, and H263 original native action executor.
- TRAIN = the same loaded base plus exactly **two real FP16 rank-4
  language q_proj/v_proj LoRA optimizer steps**, LR 5e-5, batch 1,
  original Hala completion-only mask via
  `training.scripts.train_adapter.prepare_inputs`, finite positive
  loss and gradient, nonzero actual adapter tensor updates **per step**.
  No checkpoint or weight saving.
- ADAPTED = the resulting same real LoRA model, identical heldout
  Arabic scene and exact same decoding/native executor. Report only
  coarse true `generate`, grammar, valid JSON, action-dispatched and
  actual stock delta booleans for both BASE/ADAPTED. No synthetic
  fallback click, DOM oracle to model, or model output repair.

**A two-step fit cannot establish a reliable performance gain.** Even
if the paired first-party task succeeds after LoRA, it would remain
only one author-owned app from one family. That is NOT true source-
disjoint OOD, nor proof of multilingual generalization or trained
model release readiness. A negative or missing result must stay
negative. The pilot uses previously known source app and the task was
previously used as research; it is not a fresh sealed blind test.

## Owner-private, zero-cost compute and storage

The reproducible source is in `h269/firstparty_lora_probe.py`,
`h269/build_private_payload.py`, this README, and
`tests/test_h269_firstparty_lora_pilot.py`. Kaggle metadata is
**is_private=true**, free `NvidiaTeslaT4` only, zero attached media,
model weights, datasets, kernels or credentials. It requires a real
Kaggle session with explicit noncommercial opt-in. No downloads of
Qwen weights to owner Windows or Mac.

Exact dependencies: `transformers==5.16.1`,
`peft==0.20.0`, Chromium/Playwright 1.58.0, and public
`lm-format-enforcer` Git commit
`817f944fcc8851917c40300a47f62d1defc3ffc3`. The existing
Kaggle CUDA PyTorch must remain unchanged. Weights, LMFE source,
first-party Git source, training PNGs, labels and ephemeral adapter
state live **only under private remote `/tmp/hala-h269-private`**;
the directory is removed when the kernel exits. Output directory
`/kaggle/working` is allowed to contain only one tiny
**aggregate-only** JSON report. No raw screenshots, labels,
coordinates, full generations, gold, model weights or optimization
artifacts may be persisted to Kaggle notebook outputs or public Git.

Private reproducible launch from owner Windows:

    python -m h269.build_private_payload --source h269/firstparty_lora_probe.py --destination D:\Hala-H269-Private-Upload
    kaggle kernels push -p D:\Hala-H269-Private-Upload --accelerator NvidiaTeslaT4
    kaggle kernels status abdulazizshehri/hala-h269-private-firstparty-lora-t4

Only retrieve exact small output name
`hala-h269-private-firstparty-lora-aggregate.json` via `--file-pattern`.
NEVER use unfiltered `kaggle kernels output` (may download model cache).
Remove the local upload scratch after successfully submitting.

Owner Windows first-party actual Edge preflight: **3/3 distinct
original screenshots** (default English, sorted English, RTL Arabic);
all PNGs in memory only, none persisted. H269/H268 synthetic CPU-only
source tests 11 PASS, then standalone H269 5 PASS. Remote optimizer
results require a separate genuine GPU run; a CPU mock cannot attest
real training.

## Unchanged data and scientific barriers

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains EMPTY; historical H173
scored 32-case OOD was consumed and SEALED, never reopen; H212
procedural scores barred; H250/H251 third-party original screenshots
are private research until complete historical fork and transitive
media/PNG/RGBA disjointness and rights qualification. Original H262
first-party scenes are **NOT approved independent TRAIN/DEV/OOD**.

Real project completion needs a lawful original source-disjoint,
multi-family TRAIN/DEV set, a newly never-exposed sealed blind OOD,
genuine adapter training with independent paired BASE/ADAPTED real
GUI task-state gain, and release qualification. No paid inference,
no publishing model/adapter until those gates.

## Owner-private Kaggle v1 actual failure (2026-10-31)

The owner-authenticated private free-T4 Kaggle kernel version 1 was
successfully submitted, entered RUNNING, then ended **ERROR before any
new LoRA optimizer update could be verified**. The sanitized kernel
traceback identified a runtime dependency mismatch inside
`peft/tuners/lora/torchao.py`:
`ImportError: Found an incompatible version of torchao. Found version
0.10.0, but only versions above 0.16.0 are supported`.

This is an **optional dependency conflict**, not a failed scientific
comparison. No numeric loss, adapter update, or post-training GUI task
success is claimed for v1. The owner Windows source CI at exact H269
source commit `625ce27fa244adc08fa82e42cb4986239bd8030d`
[GitHub Actions #41198124306](https://github.com/TheHalfMoon/Hala/actions/runs/41198124306)
**COMPLETED SUCCESS** on Python 3.11.14 and 3.12.14; this is
CPU/source qualification only.

For private v2, only the reviewed **exact optional
`torchao==0.10.0`** package is uninstalled inside the new ephemeral
Kaggle session after PEFT dependency installation. The experimental
LoRA does **not** require torchao, quantized weights, or QLoRA.
The launcher refuses unknown torchao versions and checks that Kaggle's
CUDA PyTorch version remains unchanged. There is no local package
installation or cleanup on owner Windows/Mac. A separate v2 real
training result must be verified; do not count v1 as 2 optimizer steps.

## Verified owner-private Kaggle v2 outcome — real LoRA, no task improvement

The exact owner-private H269 Kaggle kernel v2
`abdulazizshehri/hala-h269-private-firstparty-lora-t4`
completed on **two actual FREE NVIDIA T4 GPUs**. An allowlisted,
**1590-byte aggregate-only JSON** report was privately inspected.
There was no pretrained weight, private screenshot, raw completion,
training gold, point coordinate, or adapter upload. No model release.

The aggregated real-training evidence is:

| Measure | Verified outcome |
| --- | --- |
| Real bounded optimizer steps | **2/2** |
| Finite positive gradient norm per step | **true / true** |
| Finite positive training loss per step | **true / true** |
| Actually modified adapter tensors per step | **112 / 112** |
| Actual base model heldout Arabic image-conditioned `generate` | **true** |
| Actual adapted LoRA heldout Arabic `generate` | **true** |
| Inference-time constrained strict native H199 JSON in both | **true / true** |
| Real original browser `click` dispatched | **true / true** |
| First-party heldout Arabic Stockroom task-state success | **BASE false / LoRA false (0/1 each)** |
| Independently heldout source-family OOD | **false** |
| Official TRAIN/DEV/OOD admission | **false** |
| Generalizable learned GUI improvement proven | **false** |
| Owner weight/adapter or private media published | **false** |

**Interpretation:** H269 goes materially beyond H15's two-step
training mechanism qualification because it actually paired frozen
pretrained and newly trained adapted models with one genuine native
browser task-state measurement on an image different from the two
training images. Yet **there is NO measured GUI task gain** in this
research result. Do not claim model alignment from nonzero gradients,
valid JSON, action dispatch, loss finiteness or a first-party sample.

No raw target points or reference labels were put into the public
report. No missing task outcome was filled in from invented
coordinates, model text, or oracle knowledge. The task remains
unqualified. The next research work should focus on true visual
grounding quality and **source-diverse legally qualified original
TRAIN/DEV** and a new untouched blind OOD, not automatically lengthen
this first-party micro-training run or expose H173's consumed 32 cases.

The hosted initial H269 source CI
[GitHub Actions #41198124306](https://github.com/TheHalfMoon/Hala/actions/runs/41198124306)
**COMPLETED SUCCESS** for both Python 3.11 and 3.12. The narrow
torchao v2 repair source commit `31d9da8078412a3effe4a3ac5cf5eded462cee9e`
was also **COMPLETED SUCCESS** on hosted
[GitHub Actions #41198608845](https://github.com/TheHalfMoon/Hala/actions/runs/41198608845)
for Python 3.11.14 and 3.12.14. A later documentation-only head
requires its own independent hosted check.
