# H142 — visible-column GUI grounding recovery (external 2B research)

**Research only, never release or update flagship Hala 35B.**
Owner-private original Chromium source screens, free noncommercial Kaggle
GPU only. GitHub contains research code/tests/documentation, **not** original
model-evaluated task images, individual oracle boxes or private weights.

## Actual previous baseline, do not conceal negative results

H141 external Qwen3.5-2B LoRA model correctly localized 16/16 visible
source cards in a new 16-case original React screenshot probe, but
localized the *visible destination column heading* on **0/16**,
and **zero of 16** model-directed real browser drags persisted.
Of 15 parseable heading points, the mean predicted X was the correct
middle-column X=480, but mean Y was ≈441 rather than the visible
header Y≈182. An actuator supplied with owner/DOM reference start
and visible-heading points had separately succeeded 8/8; that
**was not model success**. The previously sealed H140 unseen OOD
BLIND32 is still NOT eligible for this phase.

## Fresh frozen H142 source, qualified on owner Windows

Original unmodified open-source React Kanban Board, MIT-licensed,
pinned upstream Git revision
`fa047f7a32afb57750ed5246456dc282ea7befcd`.
The unmodified native Chromium application produced:
- **TRAIN64** (32 AR,32 EN; half Doing and half Done destinations).
- **DEV16** (8 AR,8 EN; half Doing and half Done destinations).
- All **80 actual source-reference browser drags survived reload**.
- 80 unique original 960×720 PNG SHA values; disjoint from
  1,013 previous excluded H136–H141, old failed-H142 and one-shot
  source hashes. TRAIN and DEV are mutually SHA-disjoint.
- Frozen TRAIN manifest:
  `988e4f0b716c574c6fa11a70519102003d4230f66bb0a6f89402ea877b8cc664`.
- Frozen DEV manifest:
  `22776558cf8c18865b715faaa33ea3b6d0ee8f58e4049a9151222f86dedaa285`.

**Important qualification limitation:** On the unchanged React layout,
the original visible heading center remains fixed at X≈480 for Doing
or X≈749 for Done and Y≈182 for both. DEV is a new task/content/source
sample but **is not a new layout or application**. Any learned gain
could partly reflect positional memorization and must NOT be described
as cross-application generalization.

The source material preserves the donor's MIT licensing requirements;
none of the upstream tracked implementation is modified. Failed source
collection attempts are quarantined and SHA-excluded, not recycled.

## Trained arm, predeclared DEV protocol

The actual past H140 V2 external research rank-8 LoRA is input via its
owner SHA-verified private checkpoint
`9595e463f2b621e308beaaefffd9ade8c38210bdc59550cb87d8e5f66d2a4bd0`.
This does NOT belong to or modify the main Hala 35B weights.

`h142/train_gpu.py` provides **64 genuine optimizer steps ×
accumulation 3 = 192 real image/role examples** on a private FREE
Tesla T4: every source START task once (64) and every visibly named
destination heading twice (128). No invisible DOM target and no images
from H141's already consumed neural evaluation cohort.

Frozen external Qwen3.5 2B model revision, exact private TRAIN/DEV
manifest SHA, adapter weights SHA, package runtime, and source Git
revision are separately verified before running.

`h142/dev_gate.py` precommits criteria **before seeing H142 results**:
START≥13/16, heading≥10/16, both roles≥10/16, each destination
and language heading≥4/8, heading accuracy increases over V2, and
START degrades by no more than 2 cases. Failure means NO promotion and
H140 OOD BLIND remains untouched.

A passing DEV gate only qualifies an **entirely new** browser-action
holdout, not a deployment, model release, or generalization claim.

## Private links and state

Private source dataset:
`abdulazizshehri/hala-h142-private-visible-heading-sft-v1`.
Private free-T4 training kernel:
`abdulazizshehri/hala-h142-private-original-visible-heading-t4`.

The kernel was submitted successfully with status **RUNNING** at last
recorded observation. This README **does not claim** any completed
H142 optimizer steps, DEV gain, independent browser task completion,
or model promotion until a verified owner-local Kaggle receipt exists.

No main Hala 35B update, H140 independent BLIND calls, paid compute,
public data/model release, main merge, or deployment.
