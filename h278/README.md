# H278 — explicit task-to-target binding for the original Queueboard probe

**Status:** DRAFT research safety hardening; no Qwen accuracy, model gain, training admission, OOD result, or release.

Stacked on H277 full-state oracle Draft PR #299 at exact base
`f96c14b5764640b6b78d7ffa66d40993be813c86`. The original
H275 Queueboard HTML and model-visible screenshot/prompt/action API are
not altered; protected `main` remains unchanged.

## Problem

A research caller could provide a task prompt saying to complete one
fictional ticket while giving the post-action oracle a different
`target_ticket`, or name two tickets in one supposedly single-step
trial. Even a perfect GUI policy would then be scored against the wrong
request. A single-step paired comparison is invalid with ambiguous task
labels.

## Fix

Before creating a browser context or calling `predict`, require a
bounded printable task with exactly one complete `Q-###` ticket ID,
matching the separate evaluator's already-approved `target_ticket`.
Further restrict wording to the two preregistered original-source
single-step templates: `Complete fictional ticket Q-###` and
`أكمل الطلب الخيالي Q-###`. Reject a missing ID, a different ID,
duplicate IDs, embedded suffixes, control characters, or appended
instructions even when the one correct ID is present. Other prompts
require an explicit new research protocol. The source-owned pre/post state oracle still
checks all six tickets, even those hidden by a priority filter. Neither
the oracle's target nor DOM/source code is supplied to the screenshot
policy.

Two opt-in native browser regression cases test invalid prompts
fail-closed before policy invocation and six legal bilingual task IDs.
The existing real English/Arabic native click tests qualify normal
paths. No private screenshots/targets or model weights are stored.

## Required qualification

Run the exact-head free GitHub-hosted H276 live Ubuntu Chromium and
Windows Edge browser workflows **plus** broad CPU tests before making
claims. The policy used in these regression tests is a fixed-action
fixture, NOT real Qwen inference. Browser success is only a correctness
proof of the source/actuator/test harness.

H199 admitted TRAIN manifest remains empty; H173 original scored
OOD consumed and SEALED; H212 previously scored rows barred;
H250/H251 external assets quarantined pending provenance. H262/H275
original public sources are not independent blind OOD. No merge,
deployment, release, GPU spend, or published private media.

## Real CI repair

Initial exact-head browser workflow
[run #41213806324](https://github.com/TheHalfMoon/Hala/actions/runs/41213806324)
**FAILED** on both Ubuntu Chromium and Windows Edge at one adversarial
case: a syntactically valid ID followed by a literal extra instruction
reached the fixed-action policy. Rejecting only malformed IDs was too
weak for a preregistered one-step task. Repair explicitly requires one
of two preregistered bilingual task templates; the screenshot/actuator
and source oracle remain unchanged. Re-run exact-head live browser and
CPU checks before calling the H278 repair qualified.
