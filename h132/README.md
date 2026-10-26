# H132 — independent small-model GUI handle grounding baseline

This is an actual *alternative pretrained 2B visual point-grounder* comparison,
not a fine-tuning of the Hala model. No Hala weights are changed.

## Rationale

H131 confirmed ShowUI-2B returned points successfully on four controlled
synthetic Chromium GUI scenes but its first START missed every draggable
handle (0/4, median error 277.41 px), causing zero interface transitions.
A second visual feedback round gave 0/4 task success. Identifying a
task-relevant component is the bottleneck, so H132 tests a point specialist
trained for click grounding rather than adding coordinate/zoom prompting.

## Frozen pretrained arm and evidence

- Model `patrickballou/gui-assist-grounder` frozen SHA
  `31632edaa20749c7f532868a42635f7240033106`.
  It is an Apache-2.0 Qwen3.5-2B derived checkpoint, distinct from Hala.
- Exact model-card user prompt: screenshot first, then
  `This is a screenshot. Where should I click to accomplish this instruction:
  "{instruction}"? Output the click point in JSON format as
  {"point_2d": [x, y]}.`
- The independent START and END role target descriptions are constructed
  from the same original task instruction; model sees no DOM/ground-truth
  coordinates/visual oracle in prompts or images.
- Output is strict `point_2d` JSON on a 0–1000 grid; invalid outputs are
  recorded as invalid, never replaced with guessed coordinates.
- Same four owner-private original H131 Chromium screenshots are used
  for *paired exploratory* pretrained model comparison. Since prior H131
  failures already influenced model choice, this set is **not an unseen
  holdout** and no generalized accuracy claim can use it.
- Kaggle NVIDIA T4 free/private only; no paid service, optimizer steps,
  H71 TRAIN/EVAL/TEST, model publication or release.
- Independent owner Windows verifies source SHA, model SHA, strict native
  point parsing and invokes the actual Chromium mouse action. A task
  counts only when its real UI state reaches success, not when a coordinate
  parses or the image appears visually plausible.
- The model is an external teacher/reference candidate; do NOT rebrand
  the third-party weights as Hala or claim a Hala checkpoint improved.

Licensing note: verify external-model Apache-2.0 attribution and required
notices before derivative integration; model-card training data has separate
licensing, so copying training data is *not* authorized merely by the model
license. This pilot reuses no external training images.

## Validation

`python -m unittest discover -s tests -p 'test_h132*.py' -v`

Keep this PR Draft, no merge, deployment or public Kaggle notebook.
Future Hala training requires original legally usable data and independent
validation after genuine human H127 visual-identifiability verification.

## Actual private experimental outcome — 2026-10-26

The owner-private **free Kaggle T4** run completed at frozen model SHA
31632edaa20749c7f532868a42635f7240033106. All **8 actual neural calls**
returned native-valid 0–1000 grid coordinate JSON (4 complete pairs).
Original image/manifest hashes, model provenance and private raw prediction
file SHA were independently confirmed on owner Windows.

Owner Chromium executed the specialist model's raw predicted START/END points
(no browser oracle input). Results: **2/4 START points hit actual draggable
handles**, **1/4 complete real browser GUI tasks succeeded**, and **3/4
scenes showed visible state changes**. Per-task results: slider handle hit but
final task unsuccessful; resize handle miss; column divider handle miss; card
drag/drop succeeded. The private median START-to-handle error was **22.5 px**.

Paired prior ShowUI-2B on the **same four synthetic scenes** was 0/4
handle hits, 0/4 task successes and median START error 277.41 px. These
results are promising **external-model selection evidence**, not a trained
Hala model improvement nor an independent holdout (the four scenes were
already inspected in H131). Do NOT release or rebrand the specialist model
as Hala. Preserve private original/model outputs and record the result in
PR #121. Next: evaluate on a *new preregistered unseen synthetic set* and,
if source license/independent visual identifiability are satisfied, build
original lawful START-grounding training data for Hala LoRA distillation.
