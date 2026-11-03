# H407 — explicit source-reviewed click boundary (research-only)

## Why

Previous **real** private Kaggle source-only Qwen3-VL-2B LoRA H406, after 96 additional training steps on legally generated toy squares, reduced duplicate-target wrong clicks from **13/16 to 2/16** but still sent **two wrong clicks**. It also regressed single-target correct clicks **15/16 → 11/16**. The H406 safety gate was **FALSE**. Further optimization of completion-token loss cannot independently establish the uniqueness or correctness of a real on-screen target.

The existing H207 screenshot-only bridge and H201/H203 attested executor correctly bind model-visible original screenshot bytes, page URL and viewport to a native action, but **do not prove the model picked a unique correct target**. H207 is explicitly an injected **research fixture** interface, not a deployed safe agent. This H407 module adds an **additional separate opt-in source-review boundary** before invoking H201/H203's original real native click executor.

## Fail-closed contract

`h407.reviewed_action_bridge.run_source_reviewed_policy(page,task,url,predict,review)`:

1. Requires exact registered local original source URL and viewport, validates bounded task; refuses missing screenshot-only predictor or independent source reviewer BEFORE taking an image.
2. Captures a fresh bounded model-visible PNG **in memory only**. Only image bytes + H199 task prompt reach the predictor; no DOM, page, hidden app state or reference labels.
3. Strict H199 parser accepts only one JSON-native `click` or `abstain`; drag, keyboard, scroll and other actions are **not authorized** through this interface.
4. A click triggers a separately supplied local reviewer (NOT the model), whose structured proof must specify **exactly one screenshot-visible target**, a finite strict visible target box containing the proposed click and SHA-256 bindings to the **exact screenshot**, exact UTF-8 user task, and **canonical parsed native action**. Zero, two, invalid, missing, stale or mismatched approvals **fail before any native event**. A reviewer exception is redacted; an altered page or PNG fails closed.
5. The unchanged H201/H203 `execute_attested_native_action` takes a fresh browser screenshot immediately before dispatch, validates the exact original registered page/viewport and SHA match, then dispatches the click. A model abstention never dispatches browser input.

**Trust limit:** `VisualTargetReview` is a local data structure, **not** a cryptographic credential. A malicious or model-controlled reviewer can forge its contents. H407 **does not authenticate reviewer identity**, enforce callback-locality/no-network, prove genuinely independent source recognition, solve post-screenshot TOCTOU race, or certify medical/real GUI action safety. Only connect a separately authenticated, approved, strictly local human/source-observation component after independent review. Do **NOT** enable automatic production action dispatch merely because a callback constructs this proof. The separate H207 research path remains test-only and unchanged.

**Privacy:** H407 itself never writes PNGs/model output or calls external services, but an injected reviewer/predictor receives screenshot bytes. Callers must ensure BOTH are authorized, local/sandboxed and network confined before processing sensitive data. No patient data or private app screenshots were used here.

## Verified original first-party evidence

Test-first **RED** before implementation (missing H407 gated module), then **GREEN** after.

- `python -B -m unittest discover -s tests -p 'test_h407_source_reviewed_action.py' -v`: **11/11 PASS**. Covers unique-target proof and native click, zero/two/malformed target reviews, bounding-box/NaN/negative geometry rejection, exact PNG/task/action SHA bindings, stale screenshots, changed page URL, reviewer secrets/exceptions, missing reviewer, abstention, and unsupported input kinds.
- Previous H207 screenshot-only policy tests **14/14 PASS**, H203 preaction screenshot/native executor tests **11/11 PASS**, H208 preloaded VLM-policy safety tests **15/15 PASS**, unchanged old fixture behavior. H407 stdlib-only `python -B -S ...` **11/11 PASS**, no ML packages or model weights needed.
- **ACTUAL OWNER Windows Edge browser** executed `python -B -m h407.original_reviewed_browser_smoke`, returned exact `H407_FIRST_PARTY_FIXTURE_REVIEW_BOUNDARY_PASS`. It uses two isolated original local HTTP/HTML visual fixtures at 960×720 and restricted loopback-only Playwright network routing:
  - **One** red 40×40 square: a deliberately injected hardcoded screenshot-only fixture proposes click (200,170); separate source-known fixture reviewer proves count1 and bound box; browser actually clicks and DOM counter changes to 1.
  - **Two** red squares: the identical injected proposal is rejected because source owner reports count2; DOM counter remains 0, zero native events.
- Both first-party fixtures are owned by this project, never general websites, copyrighted user GUIs, patient data or external benchmark. **NO actual Hala model inference** and **NO externally authenticated human reviewer** participated. Real browser run on this owner's Windows took approximately 175 seconds including Edge startup and teardown; not a throughput claim or evidence of broad cross-platform validation.

Reproducible real-browser opt-in unittest: `HALA_H407_BROWSER_QUALIFY=1`, `HALA_H407_BROWSER_CHANNEL=msedge`, `python -B -m unittest discover -s tests -p 'test_h407_real_original_browser.py' -v`. Default general CI SKIPS this expensive/optional real browser integration; full fake-browser source guards stay always-on.

## Hosted H289 Chromium screenshot follow-up

The initial H407 private PR source commit `25f76d06159ae5ad348bae2f0766e9a26189b081` completed 21/22 primary PR workflows successfully, but **H289 Ubuntu Chromium** [run #41738421868](https://github.com/TheHalfMoon/Hala/actions/runs/41738421868) failed while taking its first original first-party screenshot with the exact known `Page.captureScreenshot: Unable to capture screenshot` CDP transport error. H289 already had a two-attempt fail-closed exact-error retry, but retried **immediately**, independently of H404's repaired H322 path used by H296.

A separate **test-first** H289 update therefore mirrors H404's minimal bounded renderer settle: only after FIRST exact known CDP error AND revalidation of the original static source, URL and 960×720 viewport, invoke `page.wait_for_timeout(150)` if supplied, revalidate the source URL/viewport again and take only the SECOND and LAST **fresh** screenshot. If waiting fails, abort with a sanitized error and no retry; unrelated errors or two failures remain terminal; no cached frame, changed page, model inference or input event can bypass the source checks. The helper is still DEV-only/public-first-party research, not a model or product authority.

Two new adversarial H289 tests **RED before** the change, **GREEN after**: 150 ms settlement on exact error, and fail-closed/redacted abort if settle itself fails. H289 source tests on owner Windows **14/14 OK with 4 real-browser opt-in SKIPs**, H407 new source tests **12/12 OK with 1 real-browser opt-in SKIP**. The original H407 actual Windows Edge two-HTML-fixture source-review smoke remains verified independently. The H289 Ubuntu Chromium runner must demonstrate the fix on the new pushed exact HEAD before claiming all hosted CI green. The existing H407 original source-only auto-click interface remains unchanged.

## Original Hala release remains NO-GO

Actual H199 original rendered GUI screenshot rights-approved TRAIN SHA allowlist EMPTY; independent legal/ML derivative licensing reviewer #384 OPEN and genuinely independent never-exposed bilingual EN/AR OOD custodian #385 OPEN. H407 cannot qualify source rights, model weights, autonomous action safety or legitimate external OOD. It is a **new optional supervised safety interface only**, not proof that Hala is finished or ready. Private draft PR only, no protected `main` merge, no model weight publication or patient screenshots.
