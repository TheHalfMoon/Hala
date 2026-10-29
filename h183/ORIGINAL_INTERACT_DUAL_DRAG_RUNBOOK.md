# H183 — original interact.js dual-scene native drop proof (research-only)

**No TRAIN/DEV/one-shot OOD dataset. No model, GPU, optimizer or release.**

## Original source and two preregistered bilingual drags

The original pinned public MIT donor is `taye/interact.js`, Git
`a993eb8cd5cbc65f3a19e45ee24b7fedc026174f`; its tracked original
`examples/dropzones/index.js` defines the following accepted drop rules:

- `#drag1` into `#drop1`: accepted by the original source rule.
  Research instruction (English): "Drag the first blue element into the
  first drop zone."
- `#drag2` into `#drop2`: accepted by the original source rule.
  Research instruction (Arabic): "اسحب العنصر الأزرق الثاني إلى منطقة
  الإفلات الثانية."

These are new native *source-only research scenes* following H174/H175
proof of a third draggable, not an approved dataset. They deliberately
use the exposed interact.js family already marked PROSPECTIVE_DEV in H182,
**never the reserved Frappe Gantt OOD source**.

## Actual runtime evidence contract

Reuse H174's exact original pinned donor Git/LICENSE/HTML verification,
original Vite monorepo aliases and source-tracked dropzones Vite build.
Build only in a disposable scratch copy and serve only the original
built HTML/JS/CSS over HTTP 127.0.0.1; leave the pristine donor clean.

`h183/interact_dual_native_smoke.py` runs two independent browser
contexts per scene, with Service Workers blocked, WebSockets closed,
external HTTP requests aborted, downloads disabled, and no screenshots
written to disk. It requires all four actual native interactions to
modify the original dragged element to `Dropped`, leave the unrelated
drag element unchanged, and produce 960×720 screenshots decoded to RGBA
in memory. It compares both PNG and decoded RGBA hashes across independent
contexts and rejects identical visual outcomes between the two scenes.
No coordinate gold, reward model, labels or dataset manifest is produced.

The hosted workflow `.github/workflows/h183-interact-dual-drag-linux.yml`
reuses the fully pinned H174 original Linux source/build workflow, runs
its existing `#drag3 -> #drop1` smoke first, then the new two-scene
Playwright Chromium proof. **A committed workflow is not a verified pass.**
Inspect exact-head hosted job SUCCESS and actual output string
`H183_INTERACT_TWO_ORIGINAL_NATIVE_DRAGS_SOURCE_ONLY` before claiming
the browser test succeeded.

## Admission blockers

- The 22 surviving owner Git historical source families are only a
  partial census, not the entire history of used original apps.
- The prior owner PNG/decoded RGBA hash registry and all missing archived
  source/npm/fork origins remain incompletely audited.
- No TRAIN/DEV images or model supervision are admitted. H182 roles are
  source-only suggestions; H183 images remain in memory for research QA.
- Do not open/capture Gantt source OOD until training/model/evaluation
  protocol freeze and independent full source-history audit.
- H173 one-shot v3 (2/32 vs 3/32; exact McNemar p=1.0) is consumed.
  Never use it for model selection or tuning.

No main merge, release/publication, paid inference or human review gate;
keep the work Draft until all independent qualifications pass.
