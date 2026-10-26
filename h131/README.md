# H131 — actual pretrained neural predictions drive Chromium

This is a private *experimental* two-stage browser-control experiment. It is NOT
a new Hala checkpoint, not fine-tuning, and not evidence of general
computer-use capability. The four tasks use synthetic H130 browser fixtures.
No original H71 training images or human personal data are uploaded.

- Owner Windows captures actual Chromium 960×720 screenshots and instructions
  in a create-only PRIVATE directory without DOM target coordinates.
- Private Kaggle dataset stores these screenshots under a private owner slug;
  free Kaggle T4 calls the frozen pretrained ShowUI-2B model (MIT, fixed
  revision), asks for START and END independently and saves 8 raw responses.
- Owner checks image bytes, source manifest SHA, model revision and native
  [0,1] point parsing; only then sends the predicted points into actual
  Playwright mouse events on fresh deterministic Chromium fixture pages.
- Browser outcomes are measured from real local UI state. Owner generates
  another private screenshot dataset AFTER these actions, without using
  target boxes, to give the same frozen model one corrective inference round.
- The corrective Kaggle run is a *separate* private T4 job. Owner recreates
  the first state using the original real model points and confirms the
  intermediate screenshot SHA matches the corrective model input, then
  executes the second predictions and records actual terminal success.
- Failed or missing model responses fail closed: there is no oracle geometry
  substitution, no free-form browser/site access, no model or fixture
  deployment, no paid compute, no public source images or model weights.

H130 owner-only oracle E2E passed 4/4 executor tasks; that does NOT count
toward H131 neural success. H127 independent blind visual-identifiability
review remains open for genuine H71 drags.

Local contracts: `python -m unittest discover -s tests -p 'test_h131*.py' -v`.
Private Kaggle launch is manual/source-SHA pinned and requires a PRIVATE Kaggle
dataset. Any model success here describes only four synthetic GUI interactions.

## Owner-private experiment outcome (2026-10-26)

Both private free T4 stages completed with **16 actual pinned ShowUI-2B
inferences** and SHA-qualified outputs. Owner Chromium replay applied model
START/END points (4 tasks, 2 attempts each). Tasks succeeded **0/4**
on pass 1 and **0/4** after feedback; no model updates occurred. Every initial
post-action screenshot was byte-identical to its original.

Owner-only after-inference DOM scoring (h131/owner_diagnostics.py) found **0/4
START predictions inside draggable handles**, with median distance **277.41 px**
to the nearest handle boundary. This audit reads the fixture DOM only AFTER
model inference and NEVER feeds DOM positions to the model or executor.

This is a negative result on four synthetic GUI tasks, NOT a qualified
computer-use benchmark. The next experiment must address global visual target
identification rather than reuse failed local zoom or static-marker refinement.
H127 independent blinded human review remains open for authentic H71 source
semantics. Keep raw image and per-case prediction files owner-private and PR Draft.
