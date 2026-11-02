# H375 — one bounded compositor-readiness barrier for original first screenshot

**Hypothesis-driven engineering only; no measured model improvement.** Stacked on
H374 Draft #404. Do not merge main, train, release or publish source screenshots.

## Actual observed failure and why this change exists

H374 exact-head signed-metadata UTC expiry workflow **4/4 passed**, but
inherited original H337 Ubuntu Chromium failed on the publicly exposed
synthetic EN abstain case with redacted
`stage=first_image_capture kind=browser_error capture=protocol_capture`
([run 41524506740](https://github.com/TheHalfMoon/Hala/actions/runs/41524506740)).
H372 previously caught the identical first screenshot capture stage on
H319 Ubuntu. Native action validation, model tokens/weights, language,
geometry and OOD have nothing to do with that browser-protocol error.

This is **not a proven root-cause diagnosis**. Public Playwright report
[38103](https://github.com/microsoft/playwright/issues/38103) documents
intermittent CDP `Page.captureScreenshot` failures following navigation,
and [MDN requestAnimationFrame](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame)
documents scheduling a callback before a repaint. It does not prove two
render callbacks will repair Chromium's internal screenshot protocol.
Fixed time sleeps and silent screenshot retries would hide genuine
failures, so we do not use them.

## Minimal, bounded, fail-closed candidate mitigation

After original source file/role validation, page navigation, JS
`configureBeaconBoard`, viewport, text direction and **exact four
source-owned buttons** have been checked, `original_scene` now waits
for two sequential `requestAnimationFrame` callbacks before returning
the FIRST screenshot-capable page. The single JS Promise has a fixed
3,000-ms deadline: if the page does not render (background/hidden
tabs can throttle callbacks), it returns **False** and the original
fail-closed source evaluator rejects. A nonboolean response is also
rejected. After the wait, the original pinned file URL is rechecked.
No additional screenshot, screenshot retry, page reload, network
permissions change, DOM element mutation, source hash change, or
model-side prompt/action modification occurs.

Source-only invented tests assert exactly ONE `page.evaluate` call
containing exactly two RAF calls, a bounded deadline and pinned source
button invariant; rejects False/None/truthy nonboolean and propagates
the original browser error without retry. The existing original-source
H319/H337 CI tests (and the separate H371 three-trial per-OS study)
will test the actual Chromium/Edge screenshot path independently.
Do **not** call the intermittent issue fixed from a single green
attempt. A failed page-render deadline or screenshot remains a hard
failure to investigate. The absence of failures in a bounded sample
can be reported as just that, not a proven browser-internal fix.

## Independent model TRAIN/OOD gates unchanged

H199 official lawful TRAIN SHA allowlist EMPTY, H355 independently
qualified legal reviewer keys EMPTY and rights [Issue #384](https://github.com/TheHalfMoon/Hala/issues/384)
OPEN, H356 new genuinely independent never-exposed source-disjoint EN/AR
blind OOD custodian keys EMPTY and [Issue #385](https://github.com/TheHalfMoon/Hala/issues/385)
OPEN. The H374 UTC correctness fix is only signed-metadata validation,
not real benchmark or approved training. No model forward, LoRA, adapter
weights, new evaluation score, learned gain, deployment or release.
