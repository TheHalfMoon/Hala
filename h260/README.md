# H260 — two additional independent original-app family candidates

**RESEARCH ONLY — no TRAIN/DEV/OOD admission, no model inference, no optimizer
steps, no scored H173 access.** All public H260 files are Hala-authored
source, documentation, and invented synthetic tests. No original app source,
third-party icons, screenshots, visual receipts, or raw owner-private hashes
are committed.

## New owner-Windows source and browser evidence (2026-10-31)

Unlike prior repeated H250/H251 screens, this screen covers **two distinct
third-party upstream repositories**:

- [dacheson/pomodoro-timer](https://github.com/dacheson/pomodoro-timer):
  MIT license naming David Acheson; 9 commits reachable through locally
  fetched refs; one historical PNG path under docs (not loaded by the app);
  five inline SVG elements whose separate rights/origin are not established.
- [dinnocenzo/pomodoro-timer-app](https://github.com/dinnocenzo/pomodoro-timer-app):
  MIT license naming dinnocenzo; 2 commits reachable through locally fetched
  refs; no historical media paths found in those local refs.

Both original repositories were cloned into a **new owner-private D: directory
outside Hala**, pinned to locally verified commits, and left clean. Current
HTML/CSS/JS source had no external HTTP URL references, CSS `url()`
references, or image/media tags in this bounded screen. An actual offline
owner Microsoft Edge (960×720) ran both original apps without external HTTP
requests or page JavaScript errors. For each app, the original initial timer
was 25:00; a real native Short Break click changed it to 05:00; native Start
and Pause clicks changed their actual GUI control states. The second app
additionally accepted an **invented** task via its original form and showed
that task in the GUI. **Nine state checks passed across two families**,
not model task-success accuracy.

The dacheson docs screenshot was not opened, copied, or displayed. No
screenshots were captured, saved, or uploaded. The public script prints only
aggregate source/history counts and booleans; pinned commit identifiers and
private owner receipts remain off GitHub. It checks only locally reachable
history and does not claim global Git/fork/npm/deleted-history completeness.
Nor does an upstream MIT license alone establish rights for every inline SVG,
every historical asset, or the authorship chain.

## Owner-only reproduction

On the authorized Windows device, use clean original upstream clones outside
the Hala checkout and record their exact expected commit IDs privately:

```powershell
python -m h260.independent_family_screen --family dacheson --repo PRIVATE_CLONE_A --expected-commit PRIVATE_PIN_A
python -m h260.independent_family_screen --family dinnocenzo --repo PRIVATE_CLONE_B --expected-commit PRIVATE_PIN_B
```

The script uses installed Microsoft Edge and Playwright, keeps the browser
offline, reads source locally, executes original native clicks, and never
writes frames or receipts. A missing pin, dirty repo, remote mismatch, external
reference, broken GUI state, or unexpected media causes fail-closed rejection.

Hosted CI runs **invented synthetic source-screen tests only**, not original
apps, private history, Edge replay, dataset admission, or any model.

## Next scientific gates

These are two **candidate families**, not independent TRAIN/DEV/OOD examples.
Check original authorship and transitive SVG rights, investigate complete
historical/fork/npm aliases and image collision against private exclusions,
and prepare genuinely new sealed blind tasks. H199's admitted manifest remains
empty. H173's consumed 32-case scored OOD remains permanently sealed.
Only after independent qualification should authorized zero-cost GPU
Qwen3-VL baseline and LoRA be compared using real held-out GUI state success.
