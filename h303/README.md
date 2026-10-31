# H303 — source-owned wrong-focus stop before native typing

## Reproducible safety defect

The original H296 two-action FieldPort evaluator accepted a **real native
click on the wrong form input**, then still invoked the second model-time
generation and dispatched `type_text` into that wrong editable input.
The final four-field oracle correctly scored the task as a failure, but
the incorrect write had already occurred. H263's generic editable-focus
check cannot distinguish the authorized source field from a different
editable field. This is a concrete unwanted-write safety gap, not a
demonstrated model-quality defect.

## Narrow source-only repair

After the first actual native click and the existing all-four-field
unchanged-value check, the **separate source-owned evaluator** checks
whether `document.activeElement` is the exact originally requested
form input. This target comparison stays inside the browser scorer:
**no DOM, source target ID, reference coordinate or gold is returned to
the neural policy**. On a wrong or absent focus it:

- Counts the already-dispatched first click as one genuine native action.
- Scores the case as **failed**, never drops it from a denominator.
- Refuses a second screenshot/model generation or native `type_text`.
- Seals H298's preloaded two-stage policy with a terminal failure latch.
- Emits only a categorical `source_focus_guard_refused` flag.

A correctly focused field still follows the original strict second-stage
model-time grammar, pixel freshness lock, native type action and all-field
postaction oracle. First-step abstention and wrong-code failures remain
independent negative cases. This is a source-specific safety oracle,
**not** a generic DOM-assisted model predictor. The target is never
consulted to choose the model's first action.

## Qualification and scientific limits

Invented CPU tests cover illegal early/late sealing and no retry after
wrong focus. Real Linux Chromium and Windows Edge source-only browser
tests must demonstrate EN and Arabic RTL wrong-neighbor clicks stop after
one native click/one fake-model generation, while correct focus+type
continues and all existing negative cases remain failures. Hosted CI
results should be recorded only after the exact final PR head succeeds.

Historical H299/H300 private pretrained Qwen experiments were pinned to
an **older H298 source commit**, so this repair does not retroactively
change their results. No new genuine Qwen inference, LoRA optimization,
official TRAIN/DEV admission, independent blind OOD or model improvement
is claimed. H199's official admitted TRAIN manifest remains empty;
previous H292 adapted-versus-base results remain unchanged at 6/8
versus 6/8. Do not use the publicly exposed FieldPort source as blind
OOD, and do not publish private screenshots, task text, points or weights.

All changes remain on a stacked **DRAFT PR**. No main merge, release,
deployment, paid GPU or owner-machine model download.
