# H383 — FieldPort original screenshot-lock bounded CDP recovery

**Source-only first-party EXPOSED DEV research. NO TRAIN, blind OOD,
real model or adapted weights.** Stacked on H382 Draft #412.

## Evidence: actual H382 inherited CI red

At H382 exact head `ff44d0d60e1f92d03d5de7c8d72350a47bff8934`,
inherited [H298 original two-stage FieldPort browser run 41639259840,
Ubuntu Chromium job 125928276777](https://github.com/TheHalfMoon/Hala/actions/runs/41639259840)
failed on public scene `field_dev_00`, **the first source screenshot**.
The exact traceback points to `h296.fieldport_sequence_research._screen_locked_predict`
calling `h264.screenshot_locked_research._observed_png` directly,
where Chromium raised:

`Page.screenshot: Protocol error (Page.captureScreenshot): Unable to capture screenshot`

Windows Edge H298 job 125928276360 succeeded. The failure happened
before invoking the injected fake model or native browser action in the
affected scene. The original H296 flow had **no bounded retry**, unlike
H322's already qualified source-only exact-signature helper. This is
evidence of the error signature, not proof of why Chromium refused it.

## Narrow change

H296 now calls the existing `h322.source_capture_png` with its original
`h264._observed_png` as the explicit capture function at BOTH points
where the model-visible pre-action PNG is captured and subsequently
rechecked for stale-frame protection.

- Only the exact known Playwright `Error` with the complete matched
  screenshot protocol prefix may be retried **once**. Any other
  exception, repeated error or non-PNG continues to fail closed.
- Every successful attempt still runs the original H264 fully
  decoded PNG/viewport and byte-budget validation. No stale cached,
  substituted or fabricated frame is used.
- The source URL and PNG SHA screenshot-lock still must be equal
  before and after the fake/source-only model policy callback; the
  callback sees PNG+task only, never DOM or evaluator gold.
- No change to the original FieldPort HTML or its pinned Git blob,
  browser offline network routing, native focus/type validation,
  original H264 reference implementation, development data, rights
  admission or training gate.

## Reproducible verification

Invented-only tests check one exact transient then successful validated
model-facing flow, two known errors rejecting before any policy call,
unrecognized error rejecting after **one** attempt, stale changed
screenshot refusal, source URL drift refusal and malformed PNG header
denial. New four-way Ubuntu/Windows Python3.11/3.12 source CI includes
inherited H322 protocol-matcher tests.

The inherited H298 real Chrome/Edge test is the necessary system check
on this exact change. An eventual green CI run is bounded evidence
only, **not a demonstrated permanent fix** for Chromium screenshot
protocol intermittence. Do not selectively rerun a failure just to
claim a green result.

## Hala model release blocker unchanged

H199 official lawful admitted TRAIN SHA registry empty; H355 qualified
independent screenshot/title and derivative weights rights-review
key registry empty, rights Issue #384 open. H356 independent sealed
family-disjoint truly new never-exposed bilingual EN/AR blind OOD
custodian key registry empty, OOD Issue #385 open. Original public
FieldPort H296 is exposed DEV *research* and H298 uses a stub model,
not genuine pretrained/adapter inference, held-out novel OOD or model
learning. No data, model, GPU, optimizer or private screenshots saved
by this change. Keep draft, no main merge or release.
