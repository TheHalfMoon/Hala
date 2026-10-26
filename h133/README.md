# H133 — preregistered unseen 8-scene Chromium GUI comparison

Stage: new synthetic browser tasks, not yet neural evaluation or training.

## Research question

H132 found a promising *external* GUI-specialist model: 2/4 START handle
hits and 1/4 Chromium tasks completed, versus frozen ShowUI-2B at 0/4.
Those four scenes were selected after prior inspection, so we cannot
claim out-of-sample reliability. H133 freezes eight **new** offline Chromium
GUI drag scenes *before* model inference; the two pretrained model arms
must see exactly the same images and instructions.

Four interaction types, each with two distinct visual positions/layouts:
slider-north/south; resize-west/east; column-left/right; drop-high/low.
The browser's pointer events are real; the tasks are synthetic (not
third-party applications or clinical workflows). Owner-only Playwright
QA uses DOM geometry to establish that all eight scenes are solvable,
but that geometry is NEVER provided to either neural model.

## Immutable model arms and scoring

- Frozen ShowUI-2B revision per H131.
- Frozen external GUI Assist Grounder 2B revision per H132.
- Each model predicts START and END independently for each of 8 pictures:
  16 neural inference calls/arm, separate **private free Kaggle T4** jobs.
- Both arms have strict native-coordinate parsers: ShowUI [0,1], specialist
  JSON point_2d[0..1000]. Parse failures are recorded as missing predictions
  and are never replaced with guesses.
- Owner Windows independently checks private model text, pinned revisions,
  input manifest SHA + 8 media SHA, then executes exactly the predicted
  coordinates with actual Playwright mouse moves / down/up in fresh Chromium.
- Exact native-valid pairs, START handle hits, full GUI task successes, visible
  state changes and median pixel distance to nearest draggable handle.
  DOM target geometry may be inspected by owner **only after model inference**
  to calculate diagnostic score, never for prompting or choosing actions.
- Report both models *even if both fail*. No post-hoc prompt changes, no
  model update, fine-tuning, or selective case dropping.

## Guardrails

This is a more independent test than H132 but only 8 synthetic tasks;
not a powered real-world benchmark. This is still **not Hala's trained
model** and any external pretrained model must retain licensing/attribution.
No H71 original sources, human data, EVAL/TEST, paid GPU, deployment,
merged PR or release. Private screenshots and raw responses are stored
owner-private, outside GitHub.

H127 human visually-blinded identifiability review remains outstanding for
actual H71 drag examples; only after source qualification should legal,
original START-target training data or LoRA adaptation proceed.

Frozen owner screenshot manifest byte SHA-256: `01de0f06ca176286f8119aa2fbf177691f458bb910f88bcbb82a7e6d8cfd868e`. All 8 screenshots were disjoint from H131's four original image hashes; each case identity, instruction and PNG SHA were validated before model execution.
