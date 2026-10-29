# H203 — Bind native browser action to the model-visible screenshot

Status: **source-native original GUI research, not a model improvement or
TRAIN/DEV/OOD release**. Stacks on H202 exact head
`c322353a7097ee3e1dc15145a630796fd7fea940`, not protected main.

## Why this changes correctness

H201/H202 validated screenshot-shaped JSON commands, viewport and the
pinned original URL; they did NOT require that the screen remain the
same as the image used to select an action. A valid action may therefore
be stale by the time the browser executes it.

The new `execute_attested_native_action` takes the **actual in-memory
PNG bytes that its caller supplied to an inference step**, validates
bounded 960x720 PNG header/size, verifies the exact current rendered PNG
SHA-256 via constant-time digest comparison, rechecks registered loopback
URL and viewport, then dispatches through H202's guarded native pointer
and keyboard executor. Duplicate-key/invalid JSON fails before the
screenshot. PNG bytes are never written or uploaded by this code.

The public original SNAP positive smoke now uses this attested entrypoint
with its genuine preaction screenshot. A new independent original-app
negative test changes the genuine original slider **300→301** by test-only
native keyboard input after capture and verifies the original canvas
actually redraws. An old model-shaped action is then rejected before
any additional keyboard/pointer input; value remains **301**. These
are scripted fixture actions, **not an actual learned Hala policy**.

## Actual Windows qualification, October 29, 2026

On the owner's authorized Windows device, CPython **3.11.15** and
**3.12.13**, both normal and `python -O` modes:

- H201 original executor tests: **13/13** PASS in every mode.
- H202 native transition tests: **9/9** PASS in every mode.
- H203 synthetic PNG identity/malformed input/change/redirect tests:
  **11/11** PASS in every mode.
- **33/33** PASS per environment, four environments (132 observed passes).
  `compileall` PASS.
- Pinned original interact.js source Vite build with existing unchanged
  Vite config and original modules: **PASS**.
- Actual installed Microsoft Edge headless original SNAP, H203 attested
  positive execution: **2 bilingual tasks × 2 independent contexts = 4**
  original native keyboard rewards; slider 300→301 and independent
  actual grid canvas redraws; no JS errors. Exact receipt:
  `H201_ORIGINAL_NATIVE_JSON_ACTION_EXECUTOR_REWARD_PASS original_tasks=2 contexts=4 native_events=8 redrawn_canvases=4 NO_MODEL_NO_TRAINING_NO_MEDIA_NO_OOD`.
- Actual original Edge stale-screen negative: **2 bilingual tasks × 2
  independent contexts = 4** changed original canvases and **4/4**
  rejected stale actions; no extra original slider increments:
  `H203_ORIGINAL_NATIVE_STALE_PREACTION_REJECTION_PASS original_tasks=2 fresh_contexts=4 blocked_stale_actions=4 NO_MODEL_NO_TRAINING_NO_MEDIA_NO_OOD`.

GitHub CI is separate: no hosted pass is claimed before its actual result.
The amended H201 Linux workflow runs both positive and negative on
the original pinned source, and H203's CPU workflow covers Python 3.11/3.12.

## What this does NOT prove

The caller is responsible for passing **exactly the same PNG** shown
to a model; the attested entrypoint cannot prove that an external
inference component used that image. The direct H201 low-level executor
still exists for isolated tests and must not be deployed directly.
Identical-image matching does not make a screenshot-and-click atomic;
UI changes in the tiny interval after verification remain possible.
The PNG header check is a size/dimension prefilter, not a complete
PNG decoder or comprehensive browser permission system.

No model inference, new LoRA weights, learning-step, GPU, accuracy
claim, 20 quarantined private screenshot/goal reads, old consumed H173
OOD, tentative Gantt OOD, TRAIN/DEV/OOD admission, private uploads,
payment, merge, release, or deployment. H199's training release SHA
allowlist remains empty pending historical source and decoded-RGBA
provenance qualification.
