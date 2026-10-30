# H252 — audited native action diagnostics (five action types)

**Source-only invented synthetic CPU diagnostic. No model inference, training,
real measured accuracy, approved source dataset, official DEV or sealed OOD.**

## What it solves

H249 introduced `type_text` and `scroll`, H250 exercised real source-native
typing/scroll and H251 exercised real original Arabic RTL UI. The H199 action
parser already validates exact one-step native JSON actions, but a future
Qwen3-VL baseline/LoRA comparison needs **per-action diagnostics** that
do not confuse a near-miss or a valid abstention with task success.

This pure CPU module composes, rather than weakens, H199's strict parser:
- Exact JSON single-action agreement and kind match across `click`, `drag`,
  `key_press`, `type_text`, `scroll`.
- Click, keyboard focus-point and wheel pointer: Euclidean distance within
  an explicitly diagnostic 12-pixel radius. Drag requires BOTH endpoints
  within that radius.
- `type_text` requires exact bounded Unicode string equality; changing
  Arabic combining marks is **not** silently normalized into a match.
- Scroll separately reports correct direction and delta within 120 pixels;
  this does **not** imply that the browser scrolled or task succeeded.
- Invalid or malformed predictions count as invalid, unlike a valid
  explicit `abstain`, which counts as abstention but not as correctness.
- References are strictly parsed as non-abstention training *shapes*, with
  no approval for being real training data. No disk/path/data loading.
- Aggregate reports contain only counts, boolean non-claims and kind
  distribution; they never include user instructions, source URLs, raw
  screenshot pixels, coordinates, typed text, predictions or gold.

Neither a 12-pixel coordinate agreement nor an exact action string proves
true native GUI state change. The native state/oracle verification demonstrated
in H250/H251 and a future blind model output need their own independent checks.

## Runnable invented CPU example

    python -m h252.native_action_diagnostics
    python -m pytest -q tests/test_h252_native_action_diagnostics.py
    python -O -m pytest -q tests/test_h252_native_action_diagnostics.py

Owner Windows verification: **33** focused H252/H199/H249 tests PASSED,
**9** H252 optimized Python -O tests PASSED (standard pytest optimized
assertion warning), and broad H174/H176/H178/H185/H198/H199/H204/H236/
H238/H241/H242/H243/H244/H245/H246/H247/H249/H250/H251/H252 tests
**390 PASSED, 6 SKIPPED**, 2598 unrelated deselected. Initial pytest
scratch parent directory was absent (WinError3); corrected by creating
only a new H252 private parent on D and rerunning with a fresh basetemp.
No test gates disabled, all final tests zero failures; compileall and
Git whitespace check passed.

The CLI uses **five hard-coded invented** action/answer pairs only, one per
kind. The resulting 5/5 exact diagnostics are a code-format smoke,
**not** model accuracy or baseline evidence. The pair API is bounded to
32 invented pairs and intentionally has no CLI for loading dataset files.
Its output contains explicit false flags for model prediction, GUI task
success, gold admission, DEV/OOD evaluation and measured model improvement.
Do not input the consumed H173 32-case scored OOD or owner-private receipts.

## Why no new LoRA job here

H15 previously recorded a genuine two-step FP16 rank-4 Qwen3-VL-2B LoRA
Colab T4 engineering smoke, including adapter updates and resume equivalence
(see `training/README.md`). That is proof that a bounded training
mechanism has operated on remote GPU, NOT that the adapted model improved
real original computer-use accuracy. The current Windows checkout has
neither CUDA nor imported Torch/Transformers/PEFT; this module claims no
new GPU progress.

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` is still EMPTY. H250 and H251 real
source-native original app PNG and reference actions remain quarantined
because historical source/visual identity and transitive visual rights are
not complete. The old H173 scored OOD is consumed and SEALED; H212
procedural source remains barred. A scientific release still needs
independently authorized source-disjoint TRAIN/DEV, a genuinely fresh blind
OOD, actual model predictions from immutable unadapted and adapted
checkpoints and genuine task-state success evaluation.

**No merge to main, model/weight release, paid GPU, real-media export,
unsanctioned dataset admission or private gold publicity is warranted.**
