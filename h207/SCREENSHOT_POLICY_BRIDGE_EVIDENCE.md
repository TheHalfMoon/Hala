# H207 — screenshot-only policy adapter for original native GUI actions

Status: **engineering integration only**. No model weights, fine-tuning,
scores, dataset release, media publication or OOD evaluation.

## Previously missing link

H199 established a strict multimodal prompt/action JSON grammar.
H201-H203 proved native browser action dispatch with hardcoded research
fixtures; they did NOT connect a screenshot-only predictor to that
execution pipeline. H207 adds a small injectable adapter
`h207.screenshot_prediction_bridge.run_screenshot_only_policy`.

The adapter accepts a pinned original local browser page, a bounded
user-facing instruction, an exact registered original SNAP URL and a
callback `predict(png: bytes, prompt: str) -> str`. It:

1. Checks pinned registered local URL and exact 960×720 viewport before
   taking any screenshot or invoking a predictor.
2. Constructs the H199 bounded screenshot-only instruction and captures
   **one actual viewport PNG**.
3. Calls the predictor with **only immutable PNG bytes and the prompt**.
   It never supplies page/context objects, HTML, DOM, selectors or private
   owner references.
4. Rejects malformed/oversized/duplicate-key/non-finite JSON via
   H199/H200's action parser.
5. Re-checks the same pinned browser, fresh PNG hash and viewport via
   H203 before H202's guarded native click, drag or focus+key dispatch.
6. Returns a receipt that explicitly states model identity, dataset
   admission and held-out evaluation are **NOT attested**.

A callback can itself perform side effects via its own closures or
network access; this harness is not a sandbox and does not prove model
identity. A page can change in the tiny interval between fresh PNG
verification and native action; execution is not atomic.

## Actual owner Windows checks (October 29, 2026)

On the owner's Windows CPython 3.11.15 isolated QA environment, both
normal and optimized (`python -O`) source-only synthetic tests passed:

- H201 existing action executor: **13/13** per mode.
- H202 guarded native transition: **9/9** per mode.
- H203 screenshot-freshness rejection: **11/11** per mode.
- H207 newly added screenshot-only predictor-bridge tests:
  **14/14** per mode (one-step click, drag, keyboard, abstain, bad JSON,
  unsupported keys, nonstring output, stale image, navigation, bad
  viewport, unsafe tasks, exceptions, malformed screenshot).
- **47/47** per mode; **94 successful observed synthetic test results**
  across both modes. `compileall` passed.

**Observed actual owner Windows Edge source-native proof**, based on
existing pinned original `taye/interact.js` SNAP Vite build
(no source modifications): H207 fixture-policy callback on two bilingual
original slider tasks, each in two independent fresh Edge contexts
(**4/4 genuine browser executions, 8 actual native input events,
4 actual original canvas redraws**). Original slider 300→301,
independent grid source canvas digest changed and zero JS errors:
`H201_ORIGINAL_NATIVE_JSON_ACTION_EXECUTOR_REWARD_PASS
original_tasks=2 contexts=4 native_events=8 redrawn_canvases=4
NO_MODEL_NO_TRAINING_NO_MEDIA_NO_OOD`.

The new H207 callback received only a PNG and screenshot-only prompt;
it returned the exact scripted original fixture-known target action.
It is **NOT** a learned Hala policy and has no generalization/accuracy
claim. A separate real Microsoft Edge H203 stale-image negative replay
also passed: two tasks × two fresh contexts = **4/4 original source
screens deliberately changed after screenshot, 4/4 stale commands
rejected**, no extra target increment:
`H203_ORIGINAL_NATIVE_STALE_PREACTION_REJECTION_PASS
original_tasks=2 fresh_contexts=4 blocked_stale_actions=4
NO_MODEL_NO_TRAINING_NO_MEDIA_NO_OOD`.

The source-native Edge outcomes, synthetic tests and hosted GitHub
Actions are separate evidence classes. **No hosted H207 pass is
claimed** until actually observed at the exact commit.

## What still blocks actual model development

There is no independently validated Hala checkpoint wired into H207,
no model processor pixel/token parity certificate and no successful
new LoRA update for this action interface. The original 20 prospect
screenshots are still unadmitted; H206 protects source metadata but
does not make historical old source/fork/RGBA provenance complete.
H199 has an empty official TRAIN allowlist. Do not open consumed
H173 OOD or tentative sealed Gantt; do not upload owner-private media.
No paid compute, merge to main, release or publication.
