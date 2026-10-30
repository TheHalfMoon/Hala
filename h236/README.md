# H236 — Original Dragula native drag on Windows (source-only research)

**No dataset admission, no model prediction, no weight update, no release.**
This stage extends H235 with a genuinely different source-family drag
interaction from the H232 autocomplete origin. It does not re-label H168
Choices.js, H173 OOD or any owner-private research cohort as new.

## Source, provenance and scope

- Upstream original MIT GitHub source: `bevacqua/dragula`; default
  branch `master`, pinned HEAD
  `09ab978dd550490f1f8e456551a1a39ed0d7f984`.
- Original public root license `license` SHA-256
  `13c2292db05054bffd25180c9e696cf798a033b0fcdc0806b7222aedfd3c2cf8`.
- Original bundled JS `dist/dragula.min.js` SHA-256
  `7a08f9c71b63e340ecd0651b39b559ee6d8b10eb27beef62e4e82d99dd48fa39`.
- Original index `index.html` SHA-256
  `ac94a69fbf7b7833bc0b4b13daef74eeb6695c474508b6fa24d0b2c0cfb86795`.
  This original demo includes third-party assets/links and is NOT loaded
  in the browser. The source file is pinned for provenance only.
- The source files remain unchanged in a clean Git checkout with a
  literal expected upstream Git origin.
- **Important**: H168 recorded Choices.js before this study, and
  H176 already excludes SortableJS, tingle, tiny-slider and other old
  source families. No textual match for `dragula` was found in the
  tracked H235 public source/docs; this is **not** proof of exhaustive
  old private source/fork/npm history or screenshot disjointness.

## What is actually executed

`original_dragula_native.py` loads original pinned public Dragula JS,
unmodified, and invokes its actual `dragula([left,right])` constructor
in an isolated real browser. A **self-authored tiny HTML/CSS fixture**
provides two visible panels and a draggable card. Actual Playwright
mouse drag moves a card from left to right. The outcome is checked
against actual post-action native DOM state, and visual PNG and decoded
RGBA receipts are captured privately for two independent contexts.

The action is a real runtime browser interaction, **not** an Hala model
prediction and not a replay of an untouched original application demo.
English and Arabic are context tags for authored QA, not proof of
multilingual instruction-following or a bilingual trained model.

Browser network requests are aborted. A failed/missing/modified
source, attempted network, Git origin mismatch, dirty checkout or
unadmitted source triggers a fail-closed error. Capture output is
create-only, never under any Git checkout, with
`INCOMPLETE_NO_ADMISSION` on interrupted/failed replay.

## Local Windows reproduction

Run in Hala repository root, with Python 3.11+, Pillow, Playwright,
Git and existing Edge/Chromium; do not install large dependencies:

```powershell
python -m pytest -q tests/test_h236_original_dragula_native.py
python -O -m pytest -q tests/test_h236_original_dragula_native.py
python -m h236.original_dragula_native --source-root D:\Hala-H236-Dragula-Original-20261030 --output D:\Hala-H236-Private\original-dragula-new
```

Each output directory must be fresh. The original clone is
`D:\Hala-H236-Dragula-Original-20261030`; owner-only replay
screenshots and image/decoded-RGBA fingerprints remain on Windows in
`D:\Hala-H236-Private`, **never** inside public GitHub PRs.

## Honest gates

This is a new *research-only original-library QA candidate*. An MIT
root LICENSE is not an audit of third-party assets, bundles, fonts or
transitive dependencies. Complete private old source/fork/npm inventory
and old visual RGBA hashes remain unproven, as does absence of hidden
source overlap. Therefore H199's
`ADMITTED_TRAIN_MANIFEST_SHAS=frozenset()` remains empty; do not
enable optimizer updates, source labels or new OOD. H212 procedural
candidates remain excluded from training; H173 consumed OOD remains
permanently sealed. Do not use real medical/patient data.

No pretrained model, cloud GPU, paid APIs, private gold, OOD access,
production release or main-branch merge is part of H236.
