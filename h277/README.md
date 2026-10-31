# H277 — Filter-independent full Queueboard state qualification

**STATUS: source-level safety repair, pending exact-head hosted browser CI.
NOT a Hala model score, improvement, training dataset, independent OOD, or
release.** This branch is stacked on H276 Draft PR #297 exact head
`952f91c649a4fbf48d434bb99eaaa2e27b0a06c6`. Keep it DRAFT and
unmerged. The protected `main` remains untouched.

## Why

The H276 first-party state oracle inspected rendered `.ticket-row` elements.
When the public Queueboard priority filter is enabled, only two of its six
fictional tickets are rendered; offscreen ticket states were omitted.
An oracle that checks only visible rows is unnecessarily weak for a claim
about exact state isolation across the whole fictional source app.

## Repair

`h276/queueboard_native_research.py` now reads the six-item first-party
`tickets` state through Playwright **inside the evaluator only**. It does
not modify filters, leak DOM or target coordinates to the screenshot-only
policy, save screenshots, or change the pinned H275 original HTML.

- Require exactly six source-owned tickets and the expected six unique IDs.
- Require an exact boolean `done` for every item; reject extra, missing,
  duplicate, altered, or malformed identities fail-closed.
- The same full-state oracle is checked before the model callback and after
  the single native action, including every ticket hidden by priority.
- Existing strict screenshot lock, native action dispatcher, task-target
  isolation, completed-counter check, and public source identity pin remain.
- Native browser regression tests assert hidden ticket visibility in the
  **oracle** despite only two rendered rows, mutated invisible state
  detection, unexpected extra IDs, and duplicate identities.

## Scientific boundaries

The Queueboard original source and IDs are PUBLIC. H277 test policy is a
fixed-action stub, not pretrained or adapted Qwen. No model inference is
attested, no weights are changed, and no TRAIN/DEV or sealed OOD samples are
admitted by this patch. H199 training manifest remains EMPTY; H173 scored
original OOD stays consumed/SEALED, H212 previously scored rows barred,
and H250/H251 external assets remain quarantined. Future generalization
claims require a genuinely independent, source-disjoint blind evaluation,
paired base/adapted native task-state scoring, and legal data provenance.

## Qualification

Required: H276 browser workflow Linux Chromium + Windows Edge against the
H277 head, plus CPU workflow. Until those complete successfully, this is
an implemented change, **not** a browser-qualified one. GitHub Actions
must not upload screenshots, private text, model weights, or coordinates.
No paid GPU, owner Desktop Commander use, merge, deployment, or release.
