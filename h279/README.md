# H279 — pretrained Qwen on original Queueboard native tasks (private research)

**Preregistered experiment, not a result.** Stacked DRAFT on H278 PR #301
at exact base `c61409e0b0178227180f0d8e05c67f5a210d01f9`.
The source family is H275 original Apache-2.0 bilingual offline Queueboard,
distinct from H262 Stockroom. Both are public, previously exposed
first-party research interfaces; **neither is blind OOD**. No official
TRAIN/DEV/OOD admission or LoRA training is implied.

## Question and frozen protocol

Can the **unchanged pretrained** Qwen3-VL-2B-Instruct, pinned revision
`89644892e4d85e24eaac8bacfd4f463576704203`, choose the right native
button on a second author-owned bilingual GUI family using the H273
normalized 0..999 coordinate frame and genuine model-time constrained JSON?

Run **exactly two** real image-conditioned generations, one per
predeclared independent fresh Queueboard browser context:

| Scene | Original display | Source-owned task | UI setup |
| --- | --- | --- | --- |
| `en_priority_q118` | English | Complete fictional ticket Q-118 | priority filter on |
| `ar_all_q142` | Arabic RTL | أكمل الطلب الخيالي Q-142 | all tickets |

### Critical coordinate-frame validation ordering

During source review after the first private submission, a concrete
integration hazard was found: H208's `PreloadedVisionPolicy` calls H199's
**pixel-frame** `parse_actions` on decoded text. Raw Qwen-relative
coordinates permit y=720..999, which H199 correctly rejects as invalid
**pixels**. Converting only after H208 returns can therefore discard a
syntactically valid relative click. This is a demonstrable code-path
mismatch, **not proof** that it caused H273's previous failures.

H279 repairs this narrowly by wrapping only the pretrained processor's
`batch_decode`: H273's exact strict relative parser and deterministic
coordinate bridge convert the volatile model text **before** H208's
unchanged native pixel validator. All H208 genuine image-tensor,
model-prefix, single-sequence and strict-native-action guards remain
active. Malformed relative output is never relabeled as abstention.
Invented-only regression verifies that normalized [999,999] becomes
native [959,719], while invalid text fails closed. The initial private
H279 v1 submission preceded this correction and must **not** be
represented as a fully qualified normalized-coordinate evaluation;
only a subsequent exact-source v2 run can qualify that protocol.

A single model call receives **only** the screenshot and exact task
plus a fixed relative-coordinate instruction. The source-owned target
ID is not separately passed to the model. No DOM, reference bounding
box, point, gold, or source code enters the model policy. The H273
pinned `lm-format-enforcer` normalized grammar is applied **during**
`model.generate`; the deterministic H273 normalized-to-960×720 bridge
then passes the strict H199 native action to the H276/H278
screenshot-locked browser executor. A malformed completion fails
closed and **does not** become a synthetic abstention. The H277
post-action source oracle checks **all six tickets**, including hidden
rows, and the completed count. A real wrong-row click is a failure,
not partial success. No retry, no target-guided decoding or
post-hoc coordinate correction.

A fixed pretrained model, no optimizer steps, and no adapters are
involved. The only permitted private Kaggle output is the bounded
`hala-h279-private-queueboard-native-aggregate.json`: two scene
identifiers, genuine model/grammar counts, strict-format booleans,
native action kind/dispatch, and actual firstparty task-state
success booleans. **Never save or print** screenshots, raw model text,
click points, reference rectangles, private labels, or weights.
The output does not establish independent OOD or improved Hala skill.

## Execution constraints

The public script pins the H278 source branch and **exact commit**,
original Queueboard Git blob identity, model revision,
Transformers 5.16.1 and original H273 LMFE source qualification.
It refuses an unverified GPU, a source drift, or preexisting output.
Use the owner's private, authorized **free noncommercial T4** Kaggle
kernel only. No local model cache on the owner's Windows or Mac.
The code-only packager creates a single Python script and private
metadata with zero attached models/datasets. Only upload source;
the runtime clones pinned public source and downloads pretrained
model into private ephemeral `/tmp`, which is deleted at exit.

From a clean updated H279 checkout, and **only if no concurrent owner
GPU experiment is active**, package in a fresh short-lived D: scratch:

```powershell
python -m h279.build_private_payload --source h279/private_queueboard_qwen_baseline.py --destination D:\Hala-H279-Private-Upload
kaggle kernels push -p D:\Hala-H279-Private-Upload --accelerator NvidiaTeslaT4
kaggle kernels status abdulazizshehri/hala-h279-private-queueboard-native-t4
```

When COMPLETE, retrieve **only**
`--file-pattern hala-h279-private-queueboard-native-aggregate.json`
into a fresh owner-private directory; never use unfiltered
`kaggle kernels output`. Validate exact schema, two generation
counts, pinned source/model and free GPU before adding any sanitized
result here. Delete only this run's upload/report scratch.
An ERROR requires sanitized exception classification, not publishing
a raw traceback with private model text.

## Science gates remain closed

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` is empty. H173's consumed scored
32-case OOD is permanently sealed; H212 scored rows are barred.
H250/H251 third-party screenshots remain quarantined pending
transitive licensing, history/forks/npm and PNG/decoded-RGBA checks.
No learned visual grounding gain has been measured in H269–H272
paired LoRA trials. Even a successful H279 firstparty click would
**not** be a model improvement or unseen blind benchmark result.
