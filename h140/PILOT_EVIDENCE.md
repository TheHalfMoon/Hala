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

## Third independent source — dnd-kit/TypeScript, 32 genuine references

- Complete original MIT app `TomCani/kanban-dndkit-localstorage`
  (copyright Thomas Cani) at pinned Git SHA:
  `f3e129998e3140a386f1884faff314345e0eb10b`.
- Original tracked source clean. Local `npm ci` + `npm run build`
  (TypeScript and Vite) **PASS**; no source code changes.
- Genuine Chromium `PointerSensor` START/END mouse movements, observed
  card transition from `col_todo` to `col_doing` in `kanban:v1`
  localStorage, and original app page-reload state verification.
- The FIRST 16-task schedule achieved only 7 START and 7 END geometry
  bins, and was correctly recorded **NOT QUALIFIED**. No threshold was
  lowered. The second, position-diverse 32-task diagnostic run achieved:
  - 32/32 real persisted browser drags that survived page reload
  - 32/32 unique private original screenshot PNG SHA values
  - 12 START and 12 END true native 24-pixel geometry bins
  - 16 English and 16 Arabic tasks; zero offscreen or failed actions
- The owner-independent integrity audit verified all 32 PNG/geometry
  receipts and cross-app hash disjointness. The initial 16 from the
  FAILED schedule are not concatenated or represented as independent
  training examples.

| Qualified diagnostic sample only | Original scenes | START bins | END bins |
| --- | ---: | ---: | ---: |
| Vanilla HTML5 | 32 | 19 | 15 |
| React drag/drop | 16 | 13 | 11 |
| React + dnd-kit | 32 | 12 | 12 |
| **Total cross-application disjoint** | **80** | — | — |

All three pairwise cross-app PNG SHA overlaps are zero. Original apps
remain unmodified and independently MIT-provenanced. Owner-private PNG,
seed states, source labels and bbox coordinates are **never** pushed to
GitHub, Kaggle or public Hugging Face. Public repo contains source
collectors, verification tests and these truthful aggregate receipts.

**These 80 real samples are STILL diagnostic, not a qualified TRAIN
dataset or neural-model improvement.** In particular, H140
still requires 256 TRAIN, 64 DEV and 32 one-time BLIND scenes from >=3
real training apps and >=2 **completely different** BLIND app codebases,
independent Git/license audits, prior H136–H139 SHA exclusion, and truly
disjoint snapshots across training, development and BLIND.

## Qualification boundary

These pilot samples are not authorized for optimization, model selection or
public sharing. They prove only that these three independent original apps can provide diverse,
persisted original reference gestures and real screenshot geometry. The
H140 research adapter, 35B flagship update, GPU training and full browser
benchmark remain unstarted. Preserve the negative H139 independent React
result in PR #128 and issue #129; H140 PR #130 remains Draft.
