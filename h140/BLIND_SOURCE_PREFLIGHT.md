# H140 one-shot BLIND — ORIGINAL source runtime QA only

**Date:** 2026-10-26
**Status:** Two original MIT codebases independently verified on owner Windows,
offline, without changing tracked donor code. NOT a frozen BLIND image corpus,
NOT a trained-model score and NOT authorization to optimize on blind images.

## Pinned independent sources

| Original author / application | Frozen exact revision | Source | Offline source QA |
| --- | --- | --- | --- |
| Aisha Abrar Chunawala — `AishaChunawala/interactive-kanban-board-or-study-kanban-board` | `f89d065e2cf16f91918bf1a72ae9ac132150197e` | Vanilla HTML5/CSS/JS Kanban with native drag, `kanbanTasks` | Original mouse task moved `todo` → `in-progress`, persisted after full page reload |
| Michael Germini — `michaelgermini/Kanban-Board` | `a68c1d9d6281a793835ceaa5e4f620068d95200a` | Different multi-board Vanilla JS task/swimlane engine, `kanbanBoards` | Original mouse task moved `todo` → `inprogress`, persisted after full page reload |

Both source revisions are complete, include their original MIT LICENSE, and
have no tracked modifications. Neither original codebase is one of the
three H140 TRAIN/DEV pilot apps.

## Exact reproduction contract

Run owner-only source QA:

`python -m h140.blind_source_preflight --sources <owner-private-h140/upstream>`

The script checks each original SHA, clean tracked diff, original LICENSE
and runnable JS/HTML/CSS files, then serves original files locally in a
private Chromium context. **ALL HTTP(S) requests to hosts other than
localhost/127.0.0.1 are denied.** Initial owner-authored state is
seeded only if storage is empty. Real pointer down/move/up events must
move the original task to the intended column, reflected in
`localStorage`, AND still be present after a genuine page reload.

Owner Windows result:
- `qualified_independent_apps: 2`
- `status: SOURCE_ONLY_OOD_RUNTIME_QUALIFIED`
- Each real mouse transition PASS; each reload state PASS
- Exactly ZERO blind model inference calls
- Exactly ZERO blind screenshots collected/uploaded here
- No checkpoint selection, optimizer updates, cloud spend or Hala 35B change.

The first Aisha preflight had an unrelated localhost navigation timeout.
A second using `wait_until=commit` and strictly denied external CDN
requests passed, independently reproduced by the shared QA script.

## Previous source rejection and honest limits

`PixelPenguin94/kanban-board-vanilla-js` at
`652d763c61f59a23fbd64179e82bd363cda337e0` is REJECTED. It
contains only README/license/git metadata without any runnable Kanban
source despite its README claims. The `realAhmedAnwer/kanban`
repository `67c2d76784685da0d50f5c8e7fe8bd2bd1358866`
is complete but has button-based task status transitions instead of
the original Kanban drag acceptance target; not selected.

Passing an individual owner QA drag does not establish a geometry-diverse,
32-case frozen one-shot BLIND benchmark, balanced bilingual real tasks or
any model generalization. Never add these original application snippets
or their future BLIND screenshot/DOM gold answers to TRAIN/DEV, prompts,
gradient updates or checkpoint selection.

Next H140 gates are still FULL TRAIN256/DEV64 original images with
cross-stage disjointness, then a separately SHA-frozen 32-case BLIND
split across these two totally independent source apps and one-time
frozen-model inference. The source-only QA did not capture any of that.

Keep Draft PR #130 unmerged. No deployment, release, paid GPU or model
accuracy claims.
