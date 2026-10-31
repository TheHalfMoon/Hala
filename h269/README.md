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
