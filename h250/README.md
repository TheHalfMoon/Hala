# H250: actual original MIT standalone To-Do app — native Arabic/English text and scroll

**Research only. Not TRAIN, DEV, OOD, a neural inference, an admitted dataset,
a new source-independent benchmark or evidence of a LoRA gain.**

## Why this matters

H249 added `type_text` and `scroll` to Hala's strictly single-action
JSON output vocabulary, but synthetic schema tests did not demonstrate that
those actions matched any **genuine original complete application**.

H250 replays [Serkanbyx/simple-to-do-list](https://github.com/Serkanbyx/simple-to-do-list)
as its own unchanged offline HTML/CSS/JS application, NOT Hala-authored
wrapper markup. It operates on exactly the author's original checkout
at Git SHA `009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff`. Original source
files and their SHA256 digests are pinned in `native_original_todo_app.py`:

- Original standalone `LICENSE` (MIT)
- Original `index.html`
- Original `css/style.css`
- Original `js/app.js`

No donor source bytes or third-party images are committed to Hala. The
original Git checkout must have the expected remote, immutable SHA, pristine
status and exact original source hashes. H238's public minimum known previous
source-family blocklist must not include the donor. This does **NOT** prove
complete prior deleted Git/fork/npm or source identity histories.

## Genuine owner Windows source-native replay

Headless installed Edge navigates to the untouched original local HTML page
with Playwright `context.set_offline(True)` BEFORE navigation and route-aborts
external font/network requests. It operates in isolated per-language contexts.
It uses genuine native pointer coordinates and `keyboard.insert_text`, rather
than setting a fake DOM input value for the studied text action.

For each of two cases (one English task and one Arabic task), it records:

1. `before_focus` screenshot → source-native coordinate click to focus field
2. `after_focus` screenshot → bounded `type_text` exact Arabic/English literal
3. `after_type` screenshot → source-native coordinate click on original Add
4. `after_add` screenshot → original task text visible and persisted after reload
5. `before_scroll` screenshot after setup with 20 extra UI-created tasks
6. `after_scroll` screenshot after actual `page.mouse.wheel(0,480)`; positive
   browser scroll offset independently verified

The 20 extra tasks are **setup by original app UI**, not 20 new model
supervised trajectories. Each studied step has ONE H199/H249-valid JSON
action and one source/result screenshot pair. Source/target click coordinates,
private task text/reference labels, all PNG and RGBA hashes, original browser
state oracle and remote font request counts stay in a receipt outside Git.
Image files remain owner PRIVATE, never uploaded or model-fed.
The original UI itself is English in both cases; only the task/action
text and browser locale differ, so this is **not** native Arabic UI
localization or an independent Arabic-interface benchmark.

**Verified owner Windows result:** 2/2 real native complete-app journeys
(EN/AR), 8 real single-action steps, 12 private PNGs, actual input/reload/scroll
oracles passed; no incomplete marker. No remote font or other web resource
was allowed to load. This result is SOURCE-NATIVE APP RESEARCH ONLY.

Owner Windows focused H250/H249/H199/H247 synthetic source-contract tests:
**40 PASSED**. H250 under `python -O`: **5 PASSED** (standard pytest
optimized assertion warning). Broad original-source/history/action tests
H174/H176/H178/H185/H198/H199/H204/H236/H238/H241/H242/H243/H244/
H245/H246/H247/H249/H250: **377 PASSED, 6 SKIPPED**, 2598 unrelated tests
excluded, no failures, with a NEW writable D pytest temp (C default ACL
is broken). Source `compileall` and Git whitespace check passed.

## Boundaries, blockers and next stage

- Single independent complete app family, not diverse production work; 2
  cases from one app are NOT two independent source families.
- Transitive rights for visual assets and any externally sourced styles, and
  global historical Git/forks/old PNG+RGBA coverage are unproven. A separately
  pinned MIT license covers the project source only, not blanket approval
  of third-party application screenshots.
- H173's scored 32-case OOD stays **consumed and SEALED**, never reopened.
  H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains **EMPTY**.
- H250 is **not** admitted TRAIN/DEV/OOD, any new blind model TEST, Qwen3-VL
  inference, fine-tuning, adapter weights, or measured accuracy progress.
- Offline browser emulation and route aborts are defense-in-depth,
  **not** OS-level proof that every network path was inaccessible.
- Next substantial gate is lawful cross-source clean TRAIN/DEV with full
  history and a separately sealed new blind OOD, followed by free
  noncommercial real processor/LoRA training and independent evaluation.

## Reproduce safely

On authorized Windows, owner ORIGINAL upstream is a separately cloned,
clean Git checkout **outside Hala** at exactly the pinned SHA. Output must be
a NEW owner-private directory outside EVERY Git repository:

```powershell
python -m h250.native_original_todo_app \
  --source-root D:\Hala-H250-Original-simple-to-do-list \
  --output D:\Hala-H250-Private\h250-owner-private-research-original-todo-NEW
python -m pytest -q tests/test_h250_original_app_native.py
```

The second run must use a *different* absent output directory; no overwrite
or append of private histories or prior scored OOD is allowed.
