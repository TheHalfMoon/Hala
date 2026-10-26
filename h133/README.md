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

## Owner evidence — frozen H133 result (2026-10-26)

Both private free T4 arms completed with **16 real pretrained neural calls each**,
all eight START/END pairs native-parse valid. Owner Windows independently
verified each raw response JSON SHA against the corresponding private Kaggle
attempt and validated source image / manifest SHA and frozen model revisions.
The two models were evaluated on the exact same preselected eight NEW
Chromium scenes, and the final point predictions were executed via actual
Playwright mouse movements (no oracle point correction or model training).

| Metric | ShowUI-2B | GUI Assist Grounder 2B |
| --- | ---: | ---: |
| START points inside real draggable handles | 0/8 | **5/8** |
| Completed full browser tasks | 0/8 | **4/8** |
| Visual state changes | 4/8 | 5/8 |
| Median initial START distance to handle | 264.46 px | **0 px** |

Specialist full successes were slider-south, resize-east, column-right,
and drop-high; drop-low hit START but missed the endpoint, while
slider-north, resize-west and column-left missed the START handle.
An exact two-sided paired McNemar/binomial test over discordant pairs gives
p=0.125 for task success (4 specialist-only successes, 0 ShowUI-only)
and p=0.0625 for START hits (5 specialist-only, 0 ShowUI-only). This small,
synthetic, nonrandom sample does **not** establish generalized superiority
or trained Hala improvement. The positive direction supports a larger
predeclared independent sample and focused START-vs-END analysis; it does
**not** justify release/merging or silently adopting third-party weights.

Original eight private PNG screenshots and raw neural outputs remain outside
GitHub. Neither model underwent optimization/training; both used free private
Kaggle T4 only. H127 genuine H71 source human review is still open.
