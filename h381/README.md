# H381 — ShiftDesk first-frame source browser readiness

**NO TRAIN, legal admission, model weights, private media, or new blind OOD.**
Source-backed fix stacked on H380 #410; previous H380 status remains RED
at its own exact commit due unrelated inherited Linux H285 source screenshot.

## Actual parent-CI failure

On H380 HEAD `c68df4fc1987c65e4c7abe416ce634eb48f8fd38`,
the inherited [H285 original source browser run 41546694965,
job 125623883621](https://github.com/TheHalfMoon/Hala/actions/runs/41546694965)
failed on Ubuntu Chromium while Windows Edge succeeded.

The real trace shows `test_all_24_actual_browser_source_supervisions`
entered `_candidate_one -> source_capture_png(page)` at H285 line 134
and H322 raised its redacted `H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED`.
This is the first frame, after the source was configured but before an
action or any candidate was handed to a sink. The old H322 shared capture
helper **already has one strictly bounded retry** for a known exact CDP
protocol error; terminal error may be a second matching error or a
different browser failure. The sanitized trace does not prove which.
A prior H325 first-paint experiment passed cross-OS on one run, but that
does **not** prove any general Chromium race root cause.

## Narrow H381 experiment

In `h285/train_only_browser_candidates.py`, after the original
Git SHA source identity, URL/960×720 viewport, exactly eight source cards,
and HTML direction are checked, invoke ONE bounded browser evaluation
with TWO sequential `requestAnimationFrame` callbacks, max 3 seconds.
Require `document.readyState === 'complete'` and exactly eight
`article.card` elements before yielding the page. A literal True is
required; timeout/false/nonboolean refuse with a bounded static error,
while a Playwright exception propagates and the offline context closes.
Recheck URL after the first-render barrier. There is no screenshot
retry added, no arbitrary sleep, no source HTML mutation, no network
permission changes, no weakening of H322, no same-frame substitution or
partial sink write.

## Tests and limits

- Invented-only unit tests assert one evaluate call, exactly two frame
  callbacks and a 3-second deadline, eight-card readiness, failure on
  false/nonboolean/timeout, a browser error without retry, correct
  placement after original scene configuration, and context closure.
- New four-way Python3.11/3.12 Windows/Ubuntu stdlib source contract CI
  plus inherited H285 real source-browser jobs are needed before any
  GitHub qualification claim. No PNGs, source-owned action point labels,
  source pixels or gold are saved by the new tests.
- A green bounded run is evidence for that run, not proof the intermittent
  Chromium protocol capture bug is permanently eliminated.

All 24 H285 source scenes are public *exposed* first-party
TRAIN-CANDIDATE ONLY, not legally qualified TRAIN, DEV or blind OOD.
H199 official lawful TRAIN SHA allowlist EMPTY, H355 independent
rendered-screen/derivative weight rights reviewer key registry EMPTY,
Issue #384 OPEN; H356 independent truly new never-exposed
source-family-disjoint sealed bilingual EN/AR blind OOD custodian keys
EMPTY, Issue #385 OPEN. No lawful LoRA training, learned model gains,
adapted weight release or main merge.
