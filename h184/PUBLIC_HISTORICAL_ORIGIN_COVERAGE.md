# H184 — extend the public historical original-app source exclusion fence

**No new data collected, no scored evaluation, no training or model improvement.**

## Problem fixed

The H178 owner Git-only census inspected 22 surviving old original source
checkouts, but deleting/moving a checkout could hide that source from the
current owner-local inventory. H177's public hardcoded earlier-source
ledger attested nine repositories. Earlier publicly committed H139,
H140 and H154 records themselves name additional independently pinned
original Git donor families, including some once used and others only
proposed, investigated or rejected.

H184 adds **nine disjoint previously disclosed source families**, with
their exact public original Git SHA-40 and evidence file, to a conservative
never-claim-new source ledger:

| Earlier phase | Original Git upstream | Publicly evidenced status |
| --- | --- | --- |
| H139 | `CodeWithSindhu/kanban-board-js` | prior TRAIN/DEV |
| H139 | `arjundroid12/todo-drag-drop` | prior TRAIN/DEV |
| H139 | `CodeWithGionatha-Labs/dnd-kanban-board` | prior BLIND |
| H140 | `N3XT3R1337/kanban-board` | proposed only |
| H140 | `AishaChunawala/interactive-kanban-board-or-study-kanban-board` | source pilot |
| H140 | `aashwinshukla/FaceTask` | proposed only |
| H140 | `PixelPenguin94/kanban-board-vanilla-js` | **rejected** |
| H140 | `michaelgermini/Kanban-Board` | source pilot |
| H154 | `Farama-Foundation/miniwob-plusplus` | source-only native pilot |

The distinction is intentional: a proposal or rejected repository is **not**
represented as trained data, successful OOD, or any model score. Conservatively,
a source previously seen in research cannot be promoted as a wholly unseen
new original GUI just by selecting a different commit or app display name.

Every row is grounded in previously public Hala
`h139/README.md`, `h139/registry.py`,
`h140/SOURCE_CANDIDATES.md`, `h140/blind_source_preflight.py`,
`h154/VERIFIED_RESULTS.md`, and `h154/miniwob_source_probe.py`.
These are *public original source identities only*: never import owner
datasets, prior screenshots, predictions, gold, private models or history.

## Runtime integration and defensive file handling

`h179/source_novelty.py` imports H184's static public origin ledger;
its `screen_pins_against_history` therefore rejects new source candidates
with any of these original repository names (case-insensitive, any Git
revision), or any of these exact previously pinned Git revisions (even under
an invented GitHub repo name). The H181 tentative OOD source and H182
prospective TRAIN/DEV source plan transitively reuse this gate.

`h178/owner_git_inventory.py` also rejects historical phase and source
entries that are broken symlinks or Windows reparse points. Previously,
`Path.is_dir()` could silently report False for an unresolved symlink,
causing the old source to disappear from the census. The change fails
closed *before* testing whether a child is a directory.

## Qualification and limits

Windows CPython 3.11 owner synthetic regressions: **13 tests, 11 passed,
2 skipped** where Windows does not permit creating symlinks. The normal
and `-O` runs both succeed. The Linux CI tests exercise the symlink
cases where creation is supported; do not claim that Windows tested them.

The 9 newly pinned public prior source families plus H177's 9 earlier
public source families constitute a stronger **18-family static public
lower bound** (without counting the separate H173/npm roots). The live
22-checkout owner census is another *incomplete* source of information;
these sets must not be added as though disjoint.

Owner Windows has 37 historical phase folders and many legacy private
manifests/receipts. Identifying filenames is not proof of a complete
prior owner PNG/decoded-RGBA SHA-only registry. Do **not** open older
screenshots, gold, prediction results or consumed OOD just to fill it.

**Not authorized by H184:** source completeness, formal TRAIN/DEV
case admission, free/paid GPU training, new model performance, OOD
capture/evaluation, protected-main merge, publication, or release.
Old H173 v3 OOD is permanently consumed. Keep this PR Draft.
