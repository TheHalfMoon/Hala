# H276 — Queueboard screenshot-locked native research and live Chromium qualification

**Research only. No trained-model gain, official TRAIN/DEV admission, sealed
OOD result, model attestation, or release claim.**

## What is new

This is the first native screenshot-to-action research bridge for the
second, author-owned Hala GUI family `h275/queueboard.html`. H262
Stockroom is not reused. The Queueboard source is pinned to its exact
public Git blob identity; altered source bytes, symlinks, wrong URL or
960×720 viewport drift fail closed.

`h276/queueboard_native_research.py` supplies a one-step evaluator:

1. A fresh offline, download-disabled, service-worker-disabled browser
   context loads only the pinned first-party Queueboard.
2. Optional Arabic RTL and priority filter are set **before** the image
   is observed. The active element is blurred to avoid a blinking caret.
3. The injected policy sees only the fully decoded in-memory screenshot
   and H199 screenshot-only user task. It does **not** receive page, DOM,
   ticket oracle, target coordinates or source path.
4. A strict H199 native JSON action is parsed. URL, viewport, source
   identity and exact model-visible screenshot bytes are rechecked
   before dispatch through the H263 real native browser actuator.
5. **Only after prediction/action**, a separate first-party DOM state
   oracle checks that exactly the requested fictional ticket became
   completed, no neighboring visible row changed and the completed
   counter incremented once. It returns only safe aggregate flags.

This is a source-development oracle, not a blind independent OOD
oracle. The invented Queueboard family and ticket IDs are public.
A policy may be a stub or a genuinely preloaded H208 vision model;
the function **does not attest model identity**. Never interpret
a stub-policy test as Qwen accuracy.

## Browser qualification

`tests/test_h276_queueboard_browser.py` runs real headless Chromium
when `HALA_H276_BROWSER_QUALIFY=1`. It tests:

- Native click changing exactly the intended English ticket row.
- A wrong-neighbor click being rejected by the private state oracle.
- A real Arabic RTL priority-filtered click.
- Native abstention being a no-op.
- A page mutation during the injected policy call being blocked by the
  stale-screenshot guard **before** actuation.
- Two stable fully decoded screenshots in memory, with no saved PNGs.
- Altered first-party source bytes being rejected.

The positive click test deliberately uses an invented fixed-action
stub whose point is determined in a separate fixture page. **This is
a browser/actuator correctness test, not a model prediction or
scientific task success.** It does not use old H173/H212 scored data,
owner-private media, original third-party assets, or model weights.

`.github/workflows/h276-queueboard-browser.yml` runs the same tests
on standard GitHub-hosted Ubuntu Chromium and Windows Microsoft Edge
runners. The Windows job uses machine-installed Edge, not the owner's
remote computer. Neither job uploads artifacts or screenshots. These
are real browser/actuator engineering checks, not model inference.
The owner's Windows Remote Desktop Commander monthly quota is not
touched. The existing CPU workflow skips this opt-in browser suite;
its results must not be misreported as live browser qualification.

## Scientific limits

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains EMPTY. H173 scored
32-case OOD is permanently consumed and sealed, H212 scored procedural
cases are barred, and H250/H251 third-party media are quarantined.
Queueboard is a second first-party source family but is **publicly
exposed**, not a source-blind sealed OOD. No new free T4 experiment,
LoRA optimizer step, trained model improvement or external-family
generalization is asserted by this PR.

Next: verify exact-head live Chromium CI; when the owner device quota
resets, separately qualify Windows Edge and private Kaggle inference
using lawful source-disjoint data. For scientific progress, build a
genuinely fresh never-exposed blind OOD and paired pretrained/adapted
native task-state measurements before any quality/release claim.
