# H287 — real private 8-step Qwen LoRA cross-family native comparison

**Prerequisite research only.** This is the first H285/H286
original-ShiftDesk-trained, separate-Queueboard-evaluated pilot. It
does NOT grant official TRAIN admission, DEV data rights, a new blind
OOD benchmark, generalizable improvement, model release or publishing
of an adapter.

## Frozen protocol (before any GPU execution)

- Source: **exact** public authored Git commit
  `76a4ebb204d825b280fd4d332ad75591c5f13007`
  on `ps/h286-relative-train-label-alignment`.
- Entirely author-owned synthetic TRAIN GUI: `h285/shiftdesk.html`,
  pinned Git blob `f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac`.
  Generate **24** real-browser original TRAIN candidate screenshots,
  requiring **24** genuine native source-state correct clicks, 24
  distinct compressed PNG and decoded RGBA hashes, and 12 English
  + 12 Arabic samples. Collect all 24 before training data may flow
  to a private sink.
- Adaptation uses **8** frozen balanced steps (TRAIN index
  `0,3,6,9,12,15,18,21`: four EN/four AR). No posthoc cherry-picking
  training cases, source scoring or weights.
- Provenance/prompt bridge: H286 strict prospective H199 PIXEL
  target -> Qwen normalized 0..999 supervised click, exact pixel
  roundtrip verified. H282 fixed relative spatial instruction has
  source-specific wording "approval button"; preregistered H287
  replaces ONLY that instruction's fixed words with "action button"
  in both TRAIN and evaluation **user input prompts**. This never
  alters or repairs any neural model output. Model-time strict LMFE
  normalized click/abstain token grammar (not postprocessing);
  H273 converts model action to actual 960x720 native pixels BEFORE
  H208/H199 validator; H276 checks screenshot is stable.
- Model: unchanged pretrained
  `Qwen/Qwen3-VL-2B-Instruct` pinned revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
  PEFT0.20.0, Transformers5.16.1, pinned MIT LMFE commit,
  rank4 FP16 language q/v LoRA (trainable adapter <=2M scalar
  values), conservative GradScaler initial scale=1.0, LR=5e-5,
  8x weighted normalized coordinate target digits, exactly
  eight genuine changed tensor/finite gradient steps.
- Separate previously published **original Queueboard source family**
  `h275/queueboard.html`, immutable Git blob
  `3cf452e84a75b017905bc14b04a8571411929891`.
  Paired BASE vs ADAPTED: exact `en_priority_q118` and
  `ar_all_q142` scenes, two actual model image-conditioned
  generations each, true model-time grammar, actual one-step browser
  native click/abstain and all-six-ticket source-owned postaction state
  oracle. Before and after must use **pixel-identical** original
  screenshots, neither duplicate any TRAIN PNG, and have no
  injected evaluator DOM/gold/coordinates into policy.
- Source research already exposed: original corrected Queueboard V3
  measured pretrained success **1/2**. H287's BASE arm is a NEW
  prospective protocol and its own measured baseline must be used
  for any adaptation delta (H287 input phrasing differs); do not
  substitute historical 1/2 as a paired V1 model score.

## Private-only data and evidence

Only an authorized owner-private, noncommercial **FREE Kaggle T4**
kernel, never paid/cloud inference, Windows GPU, or Mac. All source
clone, 24 PNG TRAIN files, native labels, model weights, optimizer
state and LoRA are under PRIVATE ephemeral Kaggle `/tmp`, removed
at experiment exit. Code-only `h287/build_private_payload.py`
submits ONE Python script plus private Kaggle metadata; zero uploaded
media, datasets, adapters or model files. Source and model exact-pinned.
Kaggle's pre-existing CUDA PyTorch is not replaced.

The ONLY permitted private result is a small bounded sanitized
JSON aggregate `hala-h287-private-shiftdesk-crossfamily-lora-aggregate.json`
with categorical counts and booleans, exact source/model/hardware
pins, eight genuine optimizer-step flags, paired two-case
first-party task-state result, and explicit false assertions for
independent blind OOD, official data admission, generalized gain,
adapter or image publication.

Once owner-private kernel actually COMPLETES, download the aggregate
using **only** the exact `--file-pattern` and verify
pinned report/provenance with a separate fail-closed reader
before quoting counts. Never use unfiltered Kaggle output: it can
contain raw logs/points/images. Delete temporary downloaded report and
owner Windows source/payload scratch afterwards.

## Source-only qualification

```powershell
python -m unittest discover -s tests -p test_h287_private_shiftdesk_lora.py -v
python -m compileall -q h287
python -m h287.build_private_payload --source h287/private_shiftdesk_lora.py --destination "D:\\<fresh-private-code-only-payload>"
```

Tests reject fake optimizer changes, invalid gradients, fake
model/browser success, wrong paired source scenes, model/source
and hardware mismatch, duplicate TRAIN/EVAL image digests,
non-identical paired pre/post screenshot hashes, fake admission and
unsafe raw metadata. **Synthetic unit-test values are invented, not
a claim that Qwen has trained or improved.**

## Release barrier

Even if H287 measures a positive local Queueboard delta, this
source was **previously exposed and scored**; the only honest claim
would be a small paired *research diagnostic*. H199 official
`ADMITTED_TRAIN_MANIFEST_SHAS` remains empty, H173 consumed old OOD
remains sealed, H212 scored procedural rows barred and H250/H251
third-party media quarantine continues. A release needs separately
qualified lawful, source-diverse multi-family TRAIN/DEV, a NEW
unseen private one-shot independent blind OOD benchmark, safety,
abstention, and reproducible source-disjoint adapted native task
improvement. Keep DRAFT; no merge, no release, no publications.
