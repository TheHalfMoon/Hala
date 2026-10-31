# H290 — preserve failed H287 V1; isolated private diagnostic V2

**H287 V1 actually reached terminal `ERROR` on the owner's
FREE Kaggle T4. It did NOT produce any permitted LoRA result report.
It therefore has no proved optimizer updates, model-state gain, or
base/adapted paired score. NEVER score an errored GPU run 0/2.**

The owner downloaded only the exact-filtered H287 aggregate report
basename and found NO report. The CLI created an EMPTY (zero-byte)
private log. Neither raw GPU logs nor screenshots, coordinates,
predictions, model weights, user data, or gold were inspected or
published. The empty private scratch/log was deleted with absence
verified. V1's exact public source SHA and private kernel version 1
remain preserved: do not rewrite their provenance or repush V1.

## This version's purpose

H290 is an **isolated original LoRA research V2** based on the
H287 8-step frozen ShiftDesk TRAIN-candidate → previously exposed
Queueboard BASE/ADAPTED scientific comparison. Rather than guessing
the V1 failure cause, H290 adds categorical fail-closed diagnostics
to BOTH dependency bootstrap and runtime:

- BOOTSTRAP, GPU_PREFLIGHT, WORKSPACE, SOURCE, DEPENDENCIES,
  MODEL_LOADING, TRAIN_CANDIDATES, BASE_EVAL, LORA_SETUP,
  OPTIMIZER, ADAPTED_EVAL, RESULT_VALIDATION, RESULT_WRITE.
- An explicit coarse exception class from a strict whitelist
  (e.g. `ValueError`, `ImportError`, `CalledProcessError`),
  or `OTHER_ERROR_CLASS` for anything else.
- Strict **zero verified optimizer steps** and **no verified paired
  model score** in failure reports, even if the process actually
  reached a training stage before aborting. This avoids fabricating
  a complete trial.
- No raw exception messages, traceback text, model outputs, image
  tensors, screenshots, labels, coordinates, user assets or weights.

An error creates ONLY the tiny owner-private
`hala-h290-private-lora-stage-diagnostic.json`. Successful training
still creates the original H287 success aggregate
`hala-h287-private-shiftdesk-crossfamily-lora-aggregate.json`,
subject to the pre-existing independent H288 strict verification and
authentication of actual private V2 kernel COMPLETION and source.

**Diagnostic JSON parser** `h290/verify_stage_diagnostic.py` accepts
only the exact hardcoded 13-field schema, whitelisted stages and
exception classes, honest negative success/training/admission flags,
bounded UTF-8 valid JSON, no duplicate escaped keys, no NaN, no
private unknown nested fields, no symlink and no filename aliases.
It prints a small categorical summary only, never the original
error text or raw Kaggle logs.

## Version-locked private protocol

Exactly original author-owned H285/286 ShiftDesk TRAIN-CANDIDATES,
all 24 validated firstparty source-action examples, 8 fixed balanced
EN/Arabic language-only rank4 FP16 LoRA weighted steps, pinned
`Qwen/Qwen3-VL-2B-Instruct` at
`89644892e4d85e24eaac8bacfd4f463576704203`,
then pretrained vs adapter native Queueboard EN Q-118 and AR Q-142
from original public/source-exposed previously-scored family.
The experiment remains first-party research, NOT official TRAIN/DEV
admission, NOT blind OOD, not production-ready even on a positive
two-scene paired delta.

Private kernel metadata uses a **distinct actual owner-private slug**
`abdulazizshehri/hala-h290-private-shiftdesk-lora-stage-t4`,
avoiding overwrite of failed H287 V1. Upload includes ONLY Python
source plus private Kaggle metadata; no third-party media, datasets,
private model weights, adapters, labels, screenshots or model outputs.

The Kaggle free/noncommercial CUDA PyTorch is not replaced, with
the original authorized dependency installer wrapped in a
bounded exception classifier. All remote source/media/optimizer/model
scratch remains in ephemeral private `/tmp` and removed on exit.
No paid compute, Mac, local model weights, publishing, merge or release.

## Code-only CPU qualification

```powershell
python -m unittest discover -s tests -p test_h290_private_lora_stage.py -v
python -m compileall -q h290
python -m h290.build_private_payload --source h290/private_stage_lora.py --destination "D:\\<unique-new-private-tmp>"
```

Seven adversarial tests cover bootstrap script compilation, metadata
privacy, error-stage categories, forged optimizer/improvement flags,
duplicate JSON keys/NaN, local no-side-effects, and independent
H288 acceptance of a synthetic successful paired result. Those are
pure source tests, **not genuine GPU training**.

## Strict remote output retrieval

Before every action authenticate the exact real owner-private kernel
status. Only after terminal status inspect permitted files with an
**exact `--file-pattern`**: first the H287 original bounded success
aggregate; if none, only the H290 stage diagnostic JSON. Never fetch
raw unfiltered logs, media, weights or screenshots. On success validate
with H288. On failure validate with H290; report only fixed
categorical stage/error class. Delete all local private receipt/scratch.
If the private run ERRORs without either whitelisted report, simply
record `ERROR / NO VALID RESULT`, not an invented failure stage or
training score.

The remaining real scientific finish criteria are unchanged:
rights- and lineage-qualified officially admitted multi-family TRAIN
and DEV, separately held-out never-exposed independent blind OOD,
paired trained native GUI gains, reliability and abstention.
