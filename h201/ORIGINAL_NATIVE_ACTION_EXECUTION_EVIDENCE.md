# H201 — action JSON to **actual original browser input** and visual reward

**Status:** real native interaction research only; **NO actual Hala model call,
no private media, no TRAIN/DEV/OOD admission, no GPU, no publication**.

## Problem discovered and repaired by real owner browser testing

H199/H200 established a strict screenshot-coordinate action JSON grammar
(`click`, `drag`, `key_press`, `abstain`) but the training smoke did
**not** execute its model-shaped JSON in a real browser.
H201 creates `execute_one_native_action(page, raw, expected_url)`,
which parses through H200's duplicate-key- and NaN-rejecting parser
and issues only **native Playwright pointer and keyboard events**.
It explicitly refuses any page URL except the allowlisted original
`http://127.0.0.1:<port>/snap/index.html`, rejects viewport
mismatch, and does not use hidden DOM selectors/JS values as an
executor. Source-native DOM/Canvas accesses exist only in the
**independent verification harness** to assert reward.

The initial original browser smoke FAILED despite 13/13 synthetic
unit tests passing. Why: using the center of the native original
range input's *bounding box* as a focus click changed its actual
value from **300 to 260**, then `ArrowRight` resulted in **261**.
This is a true action-execution error, not merely a test nuisance.
An independent original pinned donor mouse-position diagnostic
found that a click on the **visible slider thumb** (7–17 pixels
right of bbox center in this original default configuration)
preserved value **300**, and the subsequently dispatched
`ArrowRight` changed it to **301**. H201's test-only scripted
source-coordinate fixture uses midpoint +12 px; this is
**not** a model prediction or generalized visual grounding.

## Actual original source and native reward

Original unchanged `taye/interact.js` MIT upstream at pinned SHA
`a993eb8cd5cbc65f3a19e45ee24b7fedc026174f`.
Original `examples/snap/index.html`, JS, CSS and LICENSE verified
as exact Git blob identities against pinned clean source before
rebuilding the original demo with its existing Vite 5 source
configuration and unchanged module aliases in an isolated scratch
directory. The original input `grid-x` and `grid-y` redraw
the real source `#grid` canvas from native `input/change` events.

The two preregistered bilingual H193 visual tasks are replayed:
English target `grid-x`, Arabic target `grid-y`, with the other
slider first seeded to 499 through original End/ArrowLeft keyboard
events. Two fresh Chromium contexts per task. Model-shaped JSON
with `key_press`, `point`, `ArrowRight` is then passed to the
**actual H201 executor**, which makes the native thumb pointer
click then the native keyboard press. Acceptance requires the original
target input to update 300 → 301, other slider still 499, full scene
PNG/RGBA before/after actually changes, and native original canvas
PNG SHA before/after actually changes with no JavaScript errors.
Fresh context replay requires exact source screenshot SHA and
visible original target click geometry; two source task states must
be visually distinct.

**Real owner Windows Edge native receipt (October 29, 2026):**

    H201_ORIGINAL_NATIVE_JSON_ACTION_EXECUTOR_REWARD_PASS
    original_tasks=2 contexts=4 native_events=8
    redrawn_canvases=4 NO_MODEL_NO_TRAINING_NO_MEDIA_NO_OOD

H201 owner CPython 3.11 unit regressions:
**13/13** normal and **13/13 under `python -O`**.
The unit mutations reject non-loopback redirect, wrong original source
URL, wrong viewport, duplicated JSON keys, NaN, out-of-range action,
hidden selector, wrong original native slider value, focus-only
visual change without original canvas redraw, differing screenshot
replay and inappropriate model action shape.

The no-cost dedicated hosted
`.github/workflows/h201-original-snap-native-executor-linux.yml`
builds and tests pinned original source in separate pristine scratch,
and runs real headless Chromium action/Canvas reward checks. Hosted CI
must independently PASS before claiming full qualification; owner
Edge pass is a distinct environment.

## Remaining science and product gap

The H201 JSON is a **known task and handpicked original visible
thumb coordinate**, not an inferred Hala model prediction.
H201 is **not** live computer-use policy quality evaluation.
There is no general camera/OCR/localization engine or arbitrary app
permission UX in this experiment.

All 20 H188/H190/H191/H194/H197 screenshots and owner gold remain
private, unadmitted and untouched. H199's admitted TRAIN manifest
SHA allowlist remains empty, so no real image/LoRA training is
enabled. Earlier H164 free T4 192-step research adapter completed,
but genuine unseen H173 v3 OOD showed no cross-app improvement;
H173 is consumed and cannot be reused. Tentative new Frappe
Gantt OOD remains sealed and unopened. Historical source,
fork/npm and old decoded-RGBA coverage remain incomplete.

No new Hala checkpoint, optimizer step, comparison accuracy,
public media, paid inference, protected-main merge, or release
results from H201.
