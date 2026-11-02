# H379 — RailKit first-capture readiness, original public source only

**Zero-cost source-browser engineering. NO legally admitted TRAIN, no model
inference, no private screenshot/gold, no blinded OOD and no release.**
Stacked on H378 Draft #408 exact parent a6ec0ccda20b6c10d5316bcbada244d4b0b29b3d.

## Observed, reproduced CI failure

At H378 HEAD, **24 of 25 GitHub workflow runs succeeded**, but the inherited
H325 original two-step RailKit Ubuntu Chromium source test failed:
[run 41536551883, job 125589739186](https://github.com/TheHalfMoon/Hala/actions/runs/41536551883/job/125589739186).
Exact traceback: the second H325 16-scene one-shot test reached the first
screenshot `one_source_case -> source_capture_png(page)` and raised the bounded
`H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED`. The first H325 16-scene source test
passed and Windows Edge H325 job passed. H322 already has one exact-signature
bounded retry; both capture attempts failed or a different capture exception
occurred. Raw Playwright exception was deliberately suppressed, so **we cannot
claim the browser's internal root cause from this traceback alone**.

## Narrow, falsifiable mitigation experiment

In H325 `open_original`, only **after** pinned public source URL, viewport,
rendered direction and exactly four source-owned buttons are checked, an
`await_railkit_first_paint` helper performs one 3,000-ms-bounded JavaScript
Promise waiting for **two sequential requestAnimationFrame callbacks**. It
also requires document.readyState complete and exactly four source route
buttons at the callback. The result must be literal True or the source
candidate is rejected before yielding its screenshot-capable page. The
original file URL is rechecked after the barrier. Any browser error
propagates, and the context is always closed.

This is not a screenshot retry, fixed sleep, external network access,
source HTML rewrite, media change, change to H322's existing bounded
capture behavior, weakening of native two-step validation or new model data.
The barrier resembles the separately qualified H375 BeaconBoard source
experiment but applies independently to RailKit's previously unprotected
first-frame source path. It may reduce a race after scene configuration,
but **does not prove why Chromium's protocol capture failed**, nor
guarantee a future CI run will be green.

## Evidence plan

- Invented-only source tests verify exactly ONE `evaluate` call, two
  animation callbacks, hard 3-second deadline, source-owned four buttons,
  literal True, fail-closed false/nonboolean, no retry, context cleanup,
  and actual barrier placement after configureRailKit.
- New 4-way Windows/Ubuntu Python3.11/3.12 source-only CI reuses inherited
  H325 tests. Original H325 real Chromium/Edge jobs run independently;
  any next red is a real red, not silently retried for a green badge.
- The source candidate role remains
  ORIGINAL_RAILKIT_TWO_STEP_TRAIN_CANDIDATES_ONLY_NOT_ADMITTED; 16 scenes,
  32 synthetic frame/label candidates, no rights approval or source OOD.

## Model Hala remains NOT TRAIN/RELEASE ADMITTED

H199 official TRAIN SHA allowlist EMPTY, H355 independent rights-review
trust keys EMPTY (GitHub [Issue #384](https://github.com/TheHalfMoon/Hala/issues/384)),
H356 truly NEW never-exposed source-family-disjoint sealed bilingual EN/AR
OOD custodian keys EMPTY ([Issue #385](https://github.com/TheHalfMoon/Hala/issues/385)).
Public original source-candidate examples are exposed development/review,
not blind OOD or legally approved training. No model weights, LoRA,
adapter-vs-base gain, actual held-out evaluation or release.
