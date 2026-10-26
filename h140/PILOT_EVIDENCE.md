# H140 genuine owner-original source pilot evidence — 2026-10-26

**Status: reproducible owner-private source diagnostic. NOT TRAIN/DEV/BLIND.**

## Original app and legal provenance

- `anantagarwal1307/Drag-Drop-Kanban-Board` — MIT; preserve its LICENSE.
- Source is checked out on the owner Windows device at exact Git revision:
  `47cfba359a71b70a6a23e511e554416aec006474`.
- No tracked source modification and no third-party cloud, paid GPU,
  published dataset, or model training. Original native HTML5 drag-and-drop,
  localStorage and manual column card layout.

## Genuine reference run

`python -m h140.vanilla_pilot --source <private original checkout>
--output <private create-only destination> --cases 32`

The H140 pilot generator varies original app seeded card count, task selection,
target index and browser scroll without editing the application code. It
captures a native 960×720 browser screenshot **before** the action and saves
owner-side private bbox geometry. Then it issues a real mouse drag, requires
the identified task to move from `todo` to `progress` in localStorage, reloads
the app and independently checks that the native app persisted the task.
The init script seeds only if localStorage is empty to avoid falsifying
the reload-persistence result.

32/32 accepted genuine reference browser actions persisted across reload.
There was **one additional rejected offscreen card geometry proposal**.
No unsuccessful proposal is represented as valid source evidence.

| Native owner-only measure | First pilot (16 cases) | Expanded pilot (32 cases) |
| --- | ---: | ---: |
| Independently verified original PNG/image hash count | 16/16 | 32/32 |
| Distinct 24-pixel START point bins | 14 | 19 |
| Distinct 24-pixel END point bins | 10 | 15 |
| English cases | 8 | 16 |
| Arabic cases | 8 | 16 |
| Persisted reference drags that survive reload | 16/16 | 32/32 |
| Real neural inference calls | 0 | 0 |
| Hala flagship weights updated | No | No |
| Training eligible | No | No |

**Data-grain warning:** First pilot 16 and expanded pilot 32 intentionally
restart the same deterministic case schedule; all **16** first-pilot image
hashes occur in expanded pilot. They must not be concatenated or cross-split
as independent data. Owner-private source dirs remain outside GitHub.
The expanded pilot is ONE 32-example diagnostic group, NOT 48 distinct cases.

The source-side geometric variance gates are now exercised on a real,
unmodified licensed app rather than manufactured rectangles.

## Second independent source — original React, 16 genuine references

- Independent complete MIT application:
  `dandemslay/React-KanbanBoard`, copyright Danny Rodriguez.
- Pinned original Git source SHA:
  `fa047f7a32afb57750ed5246456dc282ea7befcd`.
- Local owner Windows `npm ci` followed by unmodified Vite `npm run build`
  **PASS**. The original app's tracked source remains clean; `dist`
  is a local build artifact and does not alter the Git donor.
- `h140.react_pilot` creates original 960x720 Chromium snapshots with
  position-diverse To Do / Doing task cards and native React drag zones,
  then performs **real** browser drag-and-drop via Playwright.
- **16/16** unique screenshot-hash source references moved the chosen
  task from `todo` to `doing`; all changes survived browser reload.
- **13 distinct START bins and 11 distinct END bins**, measured with
  the same 24-pixel grid. 8 English / 8 Arabic instructions.
- Owner-independent SHA/PNG/dynamic geometry audit **PASS**. Image SHA
  overlap with the **expanded 32-case vanilla pilot: exactly 0**.
- The React 16 and vanilla 32 therefore give **48 distinct diagnostic
  images across TWO distinct independent source applications**. The
  previously recorded vanilla first-pilot 16 duplicates MUST NOT be added.

Only owner-private original screenshots and gold bboxes exist; public
GitHub contains code and aggregate counts only, not media or labels.

**This is diagnostic, not a qualified TRAIN dataset.** In particular, H140
still requires 256 TRAIN, 64 DEV and 32 one-time BLIND scenes from >=3
real training apps and >=2 **completely different** BLIND app codebases,
independent Git/license audits, prior H136–H139 SHA exclusion, and truly
disjoint snapshots across training, development and BLIND.

## Qualification boundary

These pilot samples are not authorized for optimization, model selection or
public sharing. They prove only that these two independent original apps can provide diverse,
persisted original reference gestures and real screenshot geometry. The
H140 research adapter, 35B flagship update, GPU training and full browser
benchmark remain unstarted. Preserve the negative H139 independent React
result in PR #128 and issue #129; H140 PR #130 remains Draft.
