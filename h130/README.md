# H130 — isolated closed-loop browser actuator and feedback contract

**Status: browser engineering validation; NOT Hala neural improvement.**

H125, H128 and H129 showed no statistically credible strict GUI grounding
improvement. Do not repeat their one-pass prompt/crop/marker attempts. H127
has two owner-private blind annotation viewers but **zero independently
completed human reviews**, so visual target identifiability is still open.

## What this implements

H130 supplies an **actual Chromium/browser pointer-event environment** with
four controlled deterministic GUI tasks: move slider, resize panel, resize
table column, drag/drop a card. It is a **synthetic offline browser fixture**,
not a real-world app benchmark and not a claim of real-world GUI skill.

The model-facing `Observation` includes screenshot PNG, instruction,
viewport size and attempt index only. A policy supplies two screenshot-
pixel coordinates (START and END); `BrowserDragEnv.execute_drag` uses
Playwright's *actual mouse move/down/move/up* events. The browser itself
updates the UI. The feedback is a fresh screenshot, observed success/failure,
terminal flag and count, so an external model can use multiple actions in
a closed loop. `run_episode` bounds actions to <=3, does not open tools
for the policy, and stores only receipt hashes, not private screenshots.
The fixture's DOM-based correctness checker and DOM selectors are accessible
to the owner-only E2E QA driver but **never to the policy**.

This architecture separates:
1. visual/task understanding from image and language,
2. pixel-coordinate generation by the external model,
3. real pointer execution and browser state transition,
4. outcome feedback and independent evaluation.

External HTTP(S) requests are blocked; browser launch uses installed
Chromium without downloading models or binaries. No model is connected
in H130: **all successful E2E trajectories use QA/oracle DOM coordinates
to validate only the browser executor**, not neural predictions.

## Validation

Run portable contracts (Playwright browser not needed):

`python -m unittest discover -s tests -p "test_h130*.py" -v`

Run actual Chromium owner-only E2E on a device with installed Playwright
and Chromium, without downloads:

`$env:H130_BROWSER_E2E='1'; python -m unittest discover -s tests -p "test_h130_browser_closed_loop.py" -v`

The controlled QA E2E deliberately executes an invalid first action and
then a valid pointer drag for each of 4 scenes. It checks actual DOM
success state, changed screenshots, terminal behavior, zero external
requests and zero JS page errors.

## Next research gate

The next **model** experiment must integrate a frozen pretrained GUI model
as an untrusted screenshot-only `propose_action` callback in an isolated
private noncommercial GPU or permitted owner local runtime, provide actual
model inference SHA evidence, and evaluate source-disjoint real interactive
trajectories. A successful scripted/oracle UI test does NOT count as Hala
improvement. In parallel, independently complete H127's human source
identifiability review before further model optimization.

No training, optimizer updates, paid cloud, deployment, PR merge, public
original screenshots, user data or model weights are involved.
