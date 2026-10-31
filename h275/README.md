# H275 — Queueboard: second original Hala GUI source family

**Status: original source-only research asset; NOT a trained model,
approved TRAIN/DEV dataset, sealed OOD benchmark, or released app.**

## Why this source exists

All real free Kaggle NVIDIA T4 H268–H273 native task experiments
were limited to the previously exposed **Hala first-party H262
Stockroom** source family. Genuine Qwen model generation, valid JSON,
native browser clicks, and 2/8-step language, vision-only, or joint
LoRA optimizer updates were achieved, but none showed a reliable
first-button localization or native task-state gain. H274 is still
investigating strict relative-output format failures.

One exposed inventory app cannot establish learned cross-application
GUI grounding. Additional legally clean original visual interface
families are prerequisites for future source-diverse training and
unseen evaluation. Do not keep reusing H262 to imply independent OOD.

## Original source and legal provenance

`h275/queueboard.html` is **newly authored, first-party Hala code**,
not copied or translated from H262 or a third-party application.
It is a stand-alone original offline fictional service-counter
dashboard. It uses only vanilla HTML/CSS/JavaScript, system fonts,
text/colored shapes and invented ticket IDs `Q-104` through
`Q-153`. No donor repositories, images, icons, stock photography,
patient records, external media, npm packages, network fonts, backend,
remote calls, credentials, persistent storage, or real customer
data are involved. The source has an Apache-2.0 SPDX marker and is
covered by this repository's existing Apache-2.0 LICENSE.

Its security-oriented Content Security Policy disables connections,
forms, third-party resources and external assets. There are no
animations, timers, real-world dates, or random initial values;
the initial queue is deterministic. Switching Arabic/English,
filtering priority, searching a fictional ticket ID, completing a
ticket and resetting the sample are all local simulated actions.

This **does not** authorize third-party screenshot copying and does
not relax transitive license checks for other original app families.

## Research GUI mechanics

- Original visual layout: distinct service-desk dashboard with
  counters and six ticket rows (not Stockroom inventory cards),
  designed compactly for a 960×720 screenshot viewport.
- Two language modes: **English and Arabic RTL**, including
  distinct translated request labels.
- Selectable filters: `All`, `Priority`, and typed fictional
  ticket-ID search.
- Each ticket has a `Complete` button with a
  source-owned `data-ticket-id`. Clicking only the intended row
  changes its DOM `data-status` from `waiting` to `done`,
  disables that button and increments
  `#completed-count` exactly once.
- The source state oracle for a **future, separately qualified
  research trial** is the changed `data-status` for the
  requested fictional ticket, checked ONLY after model inference
  and native action. The model must see pixels and user task only,
  **never selectors, DOM IDs, app script, target coordinates or
  the oracle**. Never choose a fallback target from the DOM.
- All provided samples are fictional. This application makes no
  clinical decisions and is not linked to government or hospital
  systems.

Example hypothetical task description for future unscored source
development: “Complete fictional ticket Q-137.” This wording
and these known IDs are now PUBLICLY EXPOSED and therefore
**not acceptable as a new blind sealed OOD prompt**.

## Qualification already possible without remote desktop

`tests/test_h275_original_queueboard_source.py` checks the
deterministic six-ticket source, bilingual RTL, visible action
contracts, post-click status oracle, offline CSP, Apache SPDX and
absence of external media, persistent data or animation timers.
These tests use only Python's standard library and DO NOT test
a running browser, screenshot pixels or model performance.

A true future browser qualification must separately verify actual
headless Chromium and Windows Edge behavior at 960×720,
click isolation, keyboard/RTL/filter behavior, screenshot stability,
and source hash; it should be performed when the owner's computer
access quota becomes available. Do not assert those tests passed
based solely on the static source suite.

## Permanent science and release gates

This is an additional first-party UI **source family**, but there
is no admitted training manifest or new sealed evaluation yet.
Do not admit its screenshots automatically. First establish source
identity, exact visual duplicate exclusion (both PNG and decoded
RGBA), split membership and test isolation. Its original app
is publicly visible in this PR, so it **cannot itself be claimed
as a source-blind independently unseen OOD test** merely because
it is visually different from H262.

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` is still EMPTY; H173's
32-case scored OOD remains consumed and permanently SEALED;
H212 scored procedural rows remain barred; H250/H251 third-party
media need complete historical and transitive licensing. A real
Hala release still requires legally admitted multi-family
TRAIN/DEV, genuinely new sealed source-disjoint blind OOD,
paired base/adapted native GUI task-state gains, and safety
qualification. No model/adapter weights released.
