# H247 — fix H246's two-click vs one-action training-horizon mismatch

**Actual source-native browser progress — owner PRIVATE research-only.**
No H199 TRAIN admission, no neural inference, optimizer steps, new OOD
scores, weights, merge, or model release.

## Why the H246 snapshots could not be one-step SFT

H199's `parse_actions(..., training_target=True)` requires **exactly
one** native action per multimodal screenshot. H246's eight original
MCDatepicker tasks say "choose day N and confirm" and require TWO
different native UI clicks:

1. Click the visible day. The calendar stays open; the text input
   remains empty. This action **does not** commit the selected date.
2. Click the real original **OK** control; only then does the text input
   show the selected ISO date.

H246 recorded only a pre-action screenshot after opening the calendar
and a post-action screenshot after pressing OK, **without the
intermediate visual observation**. Training Hala to answer this combined
task with a single action would be a wrong, unverifiable target, even
if these images were independently legally admitted in the future.

## Actual H247 repair

On the owner Windows machine H247 replays the SAME originally pinned
MIT [mikecoj/MCDatepicker](https://github.com/mikecoj/MCDatepicker)
library (`17dcf1ac963f38395a310f9198b427ea0d36af71`)
in offline Playwright/Edge. Its original library provenance SHA pins
and H238/H178 historical source-family, H241 old SHA-only and H198/
H236/H242/H245 research-exclusion preflight are inherited verbatim from
H246. H247 then separately checks H246's own eight-case PRIVATE receipt
for exact original source Git, source SHA, reference instructions,
native date oracle, SHA metadata and nonadmitted role.

For each of the 8 tasks (four Arabic and four English), H247 captures
**THREE real source-native screenshots**:

- `before_select`: opened original date picker, day not yet clicked;
- `after_select_before_confirm`: target day clicked but input still
  empty and picker still open;
- `after_confirm`: original OK clicked; input has matching ISO date.

It records **TWO individual click actions with different screenshot
targets and instructions**. Each action is independently accepted by
H199's **strict one-action JSON training-target parser**, but this
is a structural parser check ONLY, not training data admission.
Every step refers to exactly one source screenshot and one result
screenshot, with no DOM tokens, hidden test labels or reference images
in model-visible instruction text. Confirming screenshot coordinates
are bounded to 960x720 pixels.

Owner Windows H247 real re-execution: **8/8 native calendar journeys,
16 individual action steps, 24 new private PNG frames**. The browser
network guard is fail-closed and no model/GPU/network calls are made.

The H247 frames represent **replay of the existing H246 original
source family**, NOT a novel donor family or wholly independent
TRAIN/DEV/OOD image population. Some frames may be visually identical
to H246's prior frames, and their hashes must NOT be counted as new
independent training evidence. The only scientific gain is repairing
the missing intermediate screenshot and action horizon.

## Source/test-only public code

```powershell
python -m h247.two_step_calendar_replay --source-root D:\Hala-H246-MCDatepicker-Original-20261030 --output D:\Hala-H247-Private\mcdatepicker-private-research-h247-NEW --historical-parent D:\ --owner-root D:\ --h236-private D:\Hala-H236-Private\original-dragula-01 --h242-private D:\Hala-H242-Private\dragula-native-prospects-01 --h245-private D:\Hala-H245-Private\html5sortable-research-01 --h246-private D:\Hala-H246-Private\mcdatepicker-private-research-01
python -m pytest -q tests/test_h247_two_step_calendar_replay.py tests/test_h246_original_calendar_native.py
python -O -m pytest -q tests/test_h247_two_step_calendar_replay.py
```

The output MUST be an absent, fresh, owner-private directory outside
Git. Only the public Python source, documentation and invented synthetic
tests are committed. The 24 image frames and the private detailed
two-step reference/oracle/digest JSON receipt stay **outside Git** and
must NEVER be published, sent to a connected app, or used with a model
before full legal/media/history and disjoint split qualification.

## H248 follow-up: model-visible pixel/action fidelity

The previous source-native replay computed screenshot click labels from
Playwright target boxes but executed the original control through
`locator.click()`. That could mask a wrongly occluded or off-screen
recorded pixel: Playwright's locator click may scroll, re-aim, or wait
instead of actually executing the **recorded screenshot coordinate**.

H248 now computes the finite positive-area center coordinate, requires it
to be in the fixed 960x720 screenshot bounds and verifies with the
browser's `document.elementFromPoint(x,y)` that this exact pixel hits
the original element or its descendant, refusing occluded targets.
The original calendar selection and original OK confirmation are both
executed through `page.mouse.click(x,y)` at precisely their recorded
one-action H199 target coordinates, with the pre/post native state
assertions still required. Synthetic unit regression tests use mock
locators/mice, require exact executed pixel equality and reject occluded,
off-screen, nonfinite, or zero-size targets.

This is **action-label fidelity only**, not an independent source,
noncontamination proof, admitted TRAIN/DEV/OOD, measured model improvement,
or model release. Actual owner-private screenshots/receipts are NEVER
committed or exposed in PRs.

Owner Windows H248 verification: the actual pixel-executed original
MCDatepicker replay produced **8/8 successful bilingual native cases,
16 coordinate-executed one-action steps, and 24 owner-private PNGs**;
no incomplete marker, receipt outside the repository, no network/model/GPU.
Focused H246/H247: **20 passed**; H247 under `python -O`: **11 passed**;
broader source/history/split/action QA with isolated writable pytest temp:
**364 passed, 6 skipped** (2598 out-of-scope deselected). This validates
source-native replay and CPU contracts, not training-set admission.

## Still blocked

- H199 `ADMITTED_TRAIN_MANIFEST_SHAS` is EMPTY and remains unchanged.
- Incomplete historical original source (including deleted Git/forks/npm)
  and consumed decoded RGBA SHA history means all-time independence
  cannot be proven from known 130 prior PNG SHA-only exclusions.
- H242, H245, H246, H247 are original JS **library** runs inside
  author-created HTML fixtures, not independent unchanged real
  production-app datasets. Rights to all transitive media, independent
  real TRAIN/DEV and fresh OOD remain unqualified.
- Original scored 32-case H173 OOD was consumed, SEALED, and was NOT
  accessed. H212 procedural research cases remain optimizer-blocked.
- No real Qwen3-VL processor/model, adapter optimizer, LoRA weights,
  changed model accuracy, or release is claimed.
