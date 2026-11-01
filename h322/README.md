# H322 — Bounded original-source screenshot capture recovery

A recurring **Ubuntu Chromium** test failure was independently observed on
immutable H321 HEAD `4a845ac9b2207d2daa2e6cc8df773a3c2ec54f4b`:
Playwright raised `Page.screenshot: Protocol error (Page.captureScreenshot):
Unable to capture screenshot` in the public/original H285 source-browser
candidate loop. H309 inherits H285 and uses an additional screenshot capture.
These are browser-infrastructure failures, **not failed Qwen predictions**.

H322 permits **one additional capture attempt only for that exact
Playwright-style CDP failure**, before any native GUI action or sample output.
A second capture failure or any unrelated failure terminates with a fixed
non-identifying error; there is no cached/stale/placeholder frame or partial
TRAIN candidate emission. The successful image must still pass all H264
original 960x720 PNG decoding, source pin, screenshot byte consistency,
model-visible pixel uniqueness, browser/native task-state and data-custody
checks. H319/H320's model screenshot freshness remains untouched.

Changes: `h322/source_capture.py`, exact call sites in
`h285/train_only_browser_candidates.py` and
`h309/exposed_source_rgb_split.py`, and invented-only
`tests/test_h322_source_capture_recovery.py`. All existing original
Linux Chromium / Windows Edge and broad Python 3.11/3.12 CI remain required.

This is a bounded attempt to recover an intermittent source-browser
capture failure, **not a guarantee of CI success**; do not mark H285/H309
qualified without terminal hosted CI at this exact head. No new model,
GPU, training, adapter gain, private data, blind OOD, publication, or
release. H199 admitted TRAIN remains EMPTY, H173 sealed OOD remains
untouched, H250/H251 rights quarantined and H293/H294 safety blocks
remain enforced.
