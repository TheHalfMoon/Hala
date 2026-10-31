# H277 — Actual two-family source visual disjointness (research-only)

## Reason and scope

H276 [real Chromium qualification](https://github.com/TheHalfMoon/Hala/pull/298)
proved six original Queueboard native completion targets and English/Arabic
UI behavior work on free hosted GitHub Actions. H262 original Stockroom
had previously been verified in actual browser research. H277 now tests
these **two separately authored, Apache-2.0 first-party original GUI
families together**, including exact and semantic-pixel duplicate
detection. This is a source-qualification milestone on the path to
legal, diverse GUI training data, not an already trained model or OOD.

No screenshots, original HTML source from third parties, raw model
predictions or labels are made into datasets or persisted here.

## Real hosted Chromium visual identity gate

`tests/test_h277_two_family_pixels.py` uses actual source HTML on disk,
a fresh offline Chromium at 960×720 1× with no external network or
JavaScript errors, and a strict original H262 Stockroom source SHA256
`3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a`.

The exact **8 predeclared scenarios** are:

| Source family | English | Arabic RTL |
| --- | --- | --- |
| H262 original Stockroom | Default and stock-sorted | Stock-sorted and search “Gauze” |
| H275 original Queueboard | Default and priority-filtered | Priority-filtered and search “Q-137” |

For each scenario, decode the actual Chromium screenshot in memory
with pinned Pillow into exact **960×720 RGBA** pixels. Require:
- **8/8 distinct SHA-256 hashes of PNG byte streams**, to exclude
  identical capture bytes;
- **8/8 distinct SHA-256 hashes of decoded RGBA pixels**, to exclude
  true visual duplicates even if their PNG encoding differs;
- **two distinct source families**, each in English and Arabic.
- No remote request, no screenshot file, no screenshot artifact upload,
  no client data, no model weights or paid GPU.

The same real browser checks original source app post-click state
without any model: in Stockroom, clicking actual `+1` for a visible
fictional Gauze inventory row increments the count by exactly one;
in Queueboard, clicking Q-137 completes only that fictional row
and increments completed count by exactly one without modifying Q-104.
These are only **functional native browser tests**; reading DOM
for source QA must NEVER be conflated with model seeing it at inference.

The expanded dedicated
`.github/workflows/h276-browser-qualification.yml` includes
both the H276 actual Queueboard tests and H277 two-family source/pixel
tests, on a read-only free hosted Ubuntu Chromium job. Source tests
remain optional in the general dependency-free Hala CPU workflow;
the dedicated workflow installs exact Playwright 1.58.0 and
Pillow 11.3.0 and must execute actual browser checks. DO NOT
claim qualified until the exact H277 head's hosted browser job
**COMPLETED SUCCESS** and unit tests are not skipped.

## Data rights and benchmark exclusion

Both HTML files are authored first-party in the public Hala repository,
with no external screenshots or donors. The additional app family
is source-diverse from H262, and these *already-exposed* UI scenes
can help **future lawful TRAIN/DEV research** after independent
data/grain/duplicate rights admission. They are NOT untouched
source-blind OOD: both sources, scenario descriptions and controls
are now public and their screenshots have been inspected during QA.
No PNG or decoded RGBA artifact is published or admitted by H277.

**H199 ADMITTED_TRAIN_MANIFEST_SHAS remains EMPTY**, H173 scored
32-case OOD consumed/SEALED, H212 barred, other original
third-party app media H250/H251 quarantined pending rights/forks/npm
and encoded/decoded image duplicate checks. No new Hala optimizer
steps, scored native model improvement, model/adapter release, or
publication is claimed. Completing a real model requires a
legally admitted multi-family TRAIN/DEV dataset, an independent
new untouched blind OOD and genuine paired native task-state gains.

## Real hosted browser qualification result — 2026-10-31

**VERIFIED**: the exact original H277 implementation head
`314643d3942e58250b0f6594c4cab9294caa8bac` passed
the [read-only GitHub Actions browser job #41210533702](https://github.com/TheHalfMoon/Hala/actions/runs/41210533702)
**COMPLETED SUCCESS** on hosted Ubuntu 24.04 with Python 3.11,
pinned Playwright 1.58.0, Pillow 11.3.0 and actual installed Chromium.

The real job log explicitly reports:
- H276 Queueboard native English/RTL/control/stable-PNG tests
  **5/5 PASSED**, actual original source browser, not mocks.
- H277 actual original Stockroom+Queueboard combined tests
  **2/2 PASSED**, including **8/8 PNG byte-distinct and 8/8
  decoded RGBA pixel-distinct screenshots** across the two
  original source families, EN/Arabic, real live native clicks
  and independent post-action source state changes.
- The workflow's no-persisted-screenshots/no-upload step
  **COMPLETED SUCCESS**. No PNG files, private gold, raw model
  output, third-party media or weights were retained by this
  workflow.

This source-family visual QA is experimentally proven for
the audited Linux/Chromium version only. **Not** an owner
Windows Edge confirmation, **not** a trained Hala skill
measurement, **not** legal admission to official TRAIN/DEV
and **not** an untouched independent blind OOD evaluation.
The H199 admitted manifests remain EMPTY, and all preexisting
scientific restrictions above continue unchanged.

The broader CPU source suite has a separate hosted workflow
and must be verified independently at the final result head;
do not infer its status from this Chromium success.
