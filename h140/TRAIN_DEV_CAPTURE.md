# H140 owner-private source data acquisition — TRAIN256 / DEV64

**Never upload screenshot PNGs, owner DOM truth, reference mouse coordinates
or native localStorage task seed states to public GitHub.** This document
describes reproducibility and truthful gating, not model quality.

## Exact original application revisions

- `vanilla-native-board` — Anant Kumar Agarwal / MIT,
  `47cfba359a71b70a6a23e511e554416aec006474`
- `react-native-board` — Danny Rodriguez / MIT,
  `fa047f7a32afb57750ed5246456dc282ea7befcd`
- `react-dndkit-board` — Thomas Cani / MIT,
  `f3e129998e3140a386f1884faff314345e0eb10b`

All sources stay original, unmodified, and owner-private. Two separately
qualified OOD source runtimes are **NOT** listed above and may never be
used for optimizer updates or checkpoint selection.

## Preregistered acquisition grain (not automatically training-qualified)

| Source | TRAIN new genuine cases | DEV new genuine cases |
| --- | ---: | ---: |
| Original Vanilla HTML5 | 86 | 22 |
| Original React HTML5 drag | 86 | 22 |
| Original React+dnd-kit | 84 | 20 |
| **Total** | **256** | **64** |

Every new source capture run uses a separate fixed, even case-index
offset (**TRAIN 10000**, **DEV 20000**), balancing AR/EN per app,
rather than recycling previous H140 diagnostic pilots. All new captures
use a real Chromium original screenshot, real pointer gesture, actual
source localStorage state transition and a complete browser reload
with persisted state before admitting the private example. Reference
START/END DOM boxes and gold actions are OWNER-ONLY, not model inputs.

Run the three original `h140/vanilla_pilot.py`,
`h140/react_pilot.py`, `h140/dndkit_pilot.py` CLIs with:
- `--attempt-offset 10000` for the TRAIN role, exact counts 86/86/84.
- `--attempt-offset 20000` for DEV, exact counts 22/22/20.
- Output in private `train-candidate-{vanilla86,react86,dndkit84}`
  and `dev-candidate-{vanilla22,react22,dndkit20}`.
- Use enough `--max-attempts` to reject offscreen/refused cases and
  still achieve EXACT balanced source-positive counts; do not fake them.

A diagnostic pilot manifest MUST continue to say
`training_eligible: false`. Merely running a capture is NOT
permission to optimize on it.

## Final source qualification, once all six outputs exist

`python -m h140.freeze_train_dev --private-root <owner-private-h140>
--experiments-root <owner-experiment-root>`

This **create-only** conversion executes in two phases: validate
all 320 genuine source cases before writing anything, then independently
copy native PNG bytes into a brand-new private frozen TRAIN and DEV
package, preserving exact role labels and SHA-qualified original
reference bboxes. It requires 15 prior H136-H139 historical source
manifests and ALL prior original H140 pilot manifest hashes, not just
their filenames. All six input applications/roles must pass PNG decoding,
SHA uniqueness, MIT source revision, persisted browser-reference
actions, bilingual balance, and at least 8 true native START/END
position bins per registered source in each role.

H140 `h140.qualification.load_split` provides an independent second
admission check for exact TRAIN256, DEV64, original source provenance,
per-app languages and geometry. No directory overwrite, ambiguous
version fallback or historical source exclusion bypass is allowed.

**Important:** PRE-QUALIFIED test successes or images on the three
training apps are NOT Hala 35B model improvements. This source
freeze will not itself train any neural model, and it must not
bundle any BLIND images with the TRAIN/DEV upload.

## One-time BLIND boundary

Two entirely separate original MIT sources have passed source-only
offline mouse drag and page-reload checks in
`h140/BLIND_SOURCE_PREFLIGHT.md`. They still need original 32-case
balanced geometric source capture, SHA freeze and **redacted model
input upload only after all new model checkpoints/prompt thresholds
are frozen**. Never tune on H139 OOD or future H140 BLIND cases.
