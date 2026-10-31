# H276 — Actual browser qualification of original Queueboard

**Source-and-browser engineering milestone, not model training, scored
OOD, independent evaluation, or a model release.**

## Why this is useful

The H275 original author-owned Queueboard provides a second
bilingual synthetic GUI family alongside H262 Stockroom, but its
initial source-only tests could not show whether any requested
control really fits inside 960×720, whether an RTL button click
changes only the intended row, or whether a screenshot changes
during model inference. These were important blockers because
H264/H274 screenshot-staleness protection correctly refused
an earlier private GPU experiment when the Stockroom screenshot
changed while the model generated output.

The user's Windows Remote Desktop Commander connection has
**exhausted its monthly usage quota** and explicitly prohibits
retry/reconnection for now. H276 instead uses the existing
PUBLIC GitHub repository's zero-cost hosted Ubuntu CI
to execute a new isolated headless Playwright Chromium suite.
No owner PC dependencies or paid inference are required.

## Real Chromium actions to qualify

`tests/test_h276_queueboard_browser.py` is a real browser
suite, not a DOM mock. Each test opens the **actual unmodified
source HTML** from `h275/queueboard.html` with a fresh
960×720, 1× device-pixel-ratio Chromium context. It blocks
any HTTP/HTTPS/WebSocket request, records JavaScript errors,
and forbids screenshots or pixels being persisted or uploaded.

The suite tests:
1. Six independent row-specific Complete buttons are visibly
   bounded inside the exact 960×720 screenshot, and initial
   English screenshot bytes are identical before/after an
   idle delay. Native button geometry is read for **source
   functional qualification only**, not exposed to an AI model.
2. Clicking the **real** Complete button for fictional Q-137
   changes only that exact row's `data-status`, disables
   that specific button, increments completed count once,
   and decrements the waiting-priority count once; every other
   row remains waiting.
3. Priority filter yields exactly Q-118 and Q-137, and
   typing Q-137 yields exactly one row; after the search
   field is blurred, screenshot bytes must remain identical
   over time. Native click and counters must work through
   filtered state.
4. Arabic language toggle produces `lang=ar,dir=rtl`
   and Arabic labels; every Complete button remains fully
   visible in the screenshot. Q-118 real native completion
   changes only that target; Reset restores waiting states
   and all counts; switching back gives `lang=en,dir=ltr`.
5. English initial, filtered, Arabic filtered, and Arabic full
   scenes produce **four different rendered pixel hashes**.
   PNG signatures and exact 960×720 dimensions are checked
   from in-memory PNG bytes. No PNG or image artifact is saved.

The browser may inspect DOM/target rectangles during **these
functional app-qualification tests**, which are **not** Hala
agent inference or evaluation. Future learned-agent trials
must keep the model strictly screenshot-only and check
state/DOM only AFTER model inference and actuation. Never
promote functional UI tests into claims of model skill.

## Dedicated read-only hosted workflow

`.github/workflows/h276-browser-qualification.yml`
runs only on relevant PR changes or pushes to this dedicated
H276 branch. It uses Ubuntu 24.04, pinned Python 3.11
and Playwright 1.58.0, installs Chromium with its OS
dependencies, and executes only the actual original-browser
suite. It needs `contents: read` only, with
`persist-credentials: false` and no model, dataset, API keys,
remote sites, screenshots/artifacts upload or paid GPU.
The general Hala CPU qualification workflow remains
dependency-free: H276's test module skips only in environments
without the optional Playwright module. **The dedicated H276
browser workflow MUST have Playwright installed and run its
real tests, otherwise this milestone is not qualified.**

Hosted CPU tests proving source contracts are distinct from
hosted real Chromium tests. Do not say browser qualification
passed until the dedicated GitHub Actions browser job
completes SUCCESS at the exact H276 HEAD.

## Data rights and scientific gates

All six fictional Queueboard tickets and bilingual labels
are first-party Apache-2.0 source from H275, without external
media, healthcare records or accounts. There are no outside
dependencies in the application itself.

The repository's H199
`ADMITTED_TRAIN_MANIFEST_SHAS` stays EMPTY. H173's scored
32-case OOD is permanently CONSUMED and SEALED; H212
procedural scored cases are barred; H250/H251 third-party
media remain excluded until full source and transitive asset
rights/duplicate audits. Both Stockroom and Queueboard source
are already publicly visible, so neither serves as a freshly
unseen independent blind OOD evaluation. No Qwen adapter
weights are trained, persisted or released by H276; existing
negative native GUI quality outcomes remain negative.
