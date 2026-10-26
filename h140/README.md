# H140 — original-app positional-diversity recovery (SOURCE QUALIFICATION ONLY)

**Why:** H139 64-step rank-8 LoRA achieved 16/16 START and 16/16 END point-box
hits on same-app DEV, but only 6/8 START and **0/8 END** on a third
independent React app. Its END predictions collapsed to a single point.
That is a **negative transfer** result; Hala 35B flagship did not change.

## Predeclared data contract — no existing examples qualify automatically

The `h140.qualification` CPU admission gate requires all of:

- PRIVATE original 960×720 browser screenshots with their exact image SHA and
  originally licensed, complete, unmodified MIT app source Git SHA.
- TRAIN 256 and DEV 64 new source cases across **at least three** genuine
  applications, plus blind OOD 32 across **at least two OTHER**
  independently implemented applications. No app codebase overlap between
  TRAIN/DEV and BLIND.
- Equal English/Arabic cases per application. Descriptions may name targets
  but never carry gold DOM coordinates or bounding boxes as model inputs.
- Genuine owner-observed mouse START/END reference gestures that actually
  changed and persisted application state. This gate checks the private
  receipt; a separate Playwright/source execution audit must substantiate it.
- Native START and END bbox centers visible within the screenshot, genuinely
  diversified by real app layout, task position, list population,
  destination column, scroll, responsive viewport content or legally usable
  UI controls (NOT artificially painted target cues).
- 24-pixel geometry bins: at least **8 different START and 8 different END
  centers per app** on TRAIN and BLIND; at least 4 each per DEV app.
- Globally unique source-image hashes across TRAIN/DEV/BLIND, excluding all
  provided prior original experiment manifest hashes (especially H136–H139
  DEV/SEALED/BLIND). The caller must supply historical manifests and
  independent reviewer verification for admission.

A separate native output collapse gate expects a predeclared 16-case
dynamic-geometry slice with **at least 8 distinct neural points** and no
single native point used for more than 25% of outputs. Invalid/abstained
points fail this diversity qualification; no oracle fallback.

**No H140 real source dataset, LoRA training, GPU run or improvement claim
exists yet.** The unit tests use deliberately artificial minimal PNG headers
to falsify acceptance logic; they are not H140 training or evaluation data.

## Next engineering work

1. Identify genuinely independent complete legally usable apps and pin source
   MIT LICENSE file, creator attribution and commit SHA before acquisition.
2. Implement offline owner Windows source captures with *real* positional
   perturbations and browser-persisted reference gestures; keep all source
   pixels/labels private and never use historical blind cases for training.
3. Run independent source/audit tests and freeze actual TRAIN/DEV/BLIND split
   manifests before selecting an optimizer or any models.
4. Use free private T4 only after source qualification. Compare frozen
   external 2B vs new adapter; never call this trained flagship Hala 35B.
5. Freeze checkpoint/prompt before ONE blind OOD inference and independent
   owner-browser replay. Record all failures. Do not tune on exposed blind data.

No paid cloud, public model or dataset, deployment, merge or release.
Parent: H139 Draft PR #128, H140 tracking issue #129.
