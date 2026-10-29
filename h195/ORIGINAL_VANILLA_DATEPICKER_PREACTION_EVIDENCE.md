# H195 — original bilingual datepicker PRE-action proof

**Status: original source-only GUI research. No TRAIN, DEV, OOD
admission; no model weights/training, no old benchmark reopening.**

## Why the donor changed

The first proposed original interact.js `examples/transform`
resize showed an unchanged element size despite real browser
edge dragging, plus multiple source-side JavaScript errors. That
is NOT resize proof: it was discarded, with no screenshots or
private data saved.

The next source, `SortableJS/Sortable`, had valid pinned MIT
HTML and original JavaScript but **failed the H179 owner historical
source identity check**: its upstream was already observed.
It is NOT a novel donor and was never used to collect source images.

A different source implementation —
`mymth/vanillajs-datepicker` — passed the same conservative
H179 historical source-only preflight on Windows against **22
surviving original Git checkouts**. This still does NOT establish
history completeness or source-family eligibility for TRAIN/DEV/OOD.

## Genuine original source pin and browser tasks

Original public MIT upstream
`mymth/vanillajs-datepicker@358dfaaa3e2b8772e313d71191720f5415c9275c`,
`demo/index.html` original HTML SHA-256
`f2eeaa6393ed504765fa124991d57ae9bb8aeed5057865517b898e2af5aa247a`
and original LICENSE SHA-256
`fc128e54c03aa96a27de02d0bb9c468a6869faca42c62942abcdffa25eaa3810`.

The immutable source verifier also confirms the original demo
JavaScript and original built datepicker JS and CSS are tracked
Git blobs, not replacements; checks clean Git root and official
upstream URL, with source path reparse restrictions.

Two preregistered actual original GUI actions, each in two
fresh isolated browser contexts:

- EN: **open calendar**, capture entire GUI *before* selecting
  October 15, 2026, then click the original day and require the
  original input to contain `10/15/2026`.
- AR: **open calendar**, click the original next-month navigation
  button, capture entire GUI *before* selecting November 22,
  2026, then click the original day and require input
  `11/22/2026`.

Browser clock frozen to 2026-10-10 UTC in Riyadh time zone
**only for reproducible source QA**, not for model evaluation;
original source/date JavaScript is unchanged.
The original application page requires a CDN syntax highlighter
only for unrelated code examples. To keep browser networking
**strictly loopback-only**, the native QA supplies a no-op
`hljs.highlightAll()` in the isolated browser, which touches
no application calendar, input, click target or reward semantics.
External browser requests are aborted. The source is served only
from localhost and source files are never modified.

Each capture is before the requested action, each successful
native source reward requires the actual date input to change
and the full GUI raster to change. Both original days and both
month states must yield distinct PNG and decoded RGBA SHA.
Independent fresh contexts must replay identical input screenshots,
reference centers and native selected values. All screenshots
and owner reference positions remain in memory, NEVER saved or
uploaded. Historical H185 SHA-only screening can reject known
old PNG/RGBA collisions on owner Windows, without reading old
private screenshots.

## Actual Windows source QA — 2026-10-29

Original source clone pinned and clean. First real source smoke
selected 15 October from the app's original calendar with
zero JavaScript errors. Exact H195 full source runner on
Windows Microsoft Edge then completed:

    H195_ORIGINAL_DATEPICKER_TWO_PREACTION_DATE_SELECTIONS_PASS
    native_actions=10 preaction_scenes=2
    old_git_checkouts=22 old_sha_screened=1
    NO_DATASET_ADMISSION_NO_MODEL_NO_OOD_NO_MEDIA

Nine H195 synthetic negative/provenance tests passed on
owner CPython 3.11 normally and with optimized `-O`.
The dedicated free GitHub Linux Chromium job must be checked
at the exact final PR SHA; owner Windows success does not
automatically guarantee hosted success.

## Limits — no unearned model claim

The H188/H190/H191/H194 18 private original-source PNGs remain
untouched, quarantined, and unadmitted. **H195 adds zero stored
PNG or private gold**. This app has not been assigned TRAIN,
DEV or OOD and is not a new held-out evaluation.
The tentative Frappe Gantt source-only OOD remains unopened,
and H173 v3 scored OOD is permanently consumed.
No new model weights or accuracy metric exists.
Incomplete historical source family/fork/npm and old
decoded RGBA SHA coverage still blocks promotion and free
GPU training. All PRs must remain Draft, no main merge,
publication, paid models or cloud upload of user media.
