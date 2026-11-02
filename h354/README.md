# H354 — Preserve valid visual button clicks; correct only near misses

**Draft source-exposed research. No H354 GPU result yet.**

## Motivating observed model result

Independent owner-private Kaggle H352 (baseline-first) and H353 (order-
balanced) real pretrained Qwen3-VL-2B experiments on twelve PUBLIC,
EXPOSED original H319 BeaconBoard scenes both showed EN **6/6 to
6/6** and Arabic RTL **4/6 to 6/6** after screenshot-pixel-only
nearest-button-center snapping (2 Arabic paired wins, 0 losses).
These are repeated first-party synthetic source scenes, not blind OOD
and not trained model gains. H353 established that deterministic
baseline-first sequencing alone was not responsible for the result,
but 6 per language is far too small for generalizable statistics.

H353's exact protocol counted **6 changed first-click actions per
language**, including all ten already-successful first clicks. H354
pre-registers a conservative safety reduction: **if the Qwen
point already lies inside ANY of the four pixel-detected button
rectangles, preserve its native action exactly.** If it is outside
every rectangle and within 48 pixels of one unambiguous button,
reuse frozen H352 snap-to-center. Ambiguous/far points are unchanged,
abstention is preserved, unexpected images fail closed. This never
requires knowing which button is the task target. In particular,
the policy does not automatically change a click inside the wrong
button, since that would require a target oracle unavailable to
screenshot-only geometry.

## Evidence and integrity

- The policy receives only PNG bytes and the H199-parsed native
  model first-click. No DOM, target, label, task text, OCR, model
  reasoning, source script or CSS inside the action policy.
- H352 frozen screenshot detector and H353 counterbalanced
  two-original-browser, one-pretrained-Qwen-generation design are
  reused EXACTLY at source revision
  `37a7224f4886618e44788de92ce48ed0df2114f3`.
- H354 adds a small separately tested wrapper. Tests assert
  actions inside each of four buttons and across both light/dark
  themes are byte-for-byte preserved, whereas near-outside
  unambiguous clicks are moved and abstentions/far remain untouched.
  The private bundle embeds the frozen source and H354 wrapper
  together into ONE Kaggle script, with code SHA computed from
  actual uploaded disk bytes, no files containing screenshots,
  model weights, private data or labels.
- Only prospective owner-private free T4 pilot may produce an
  aggregate paired outcome. One-time distinct slug:
  `abdulazizshehri/hala-h354-preserve-valid-pixel-qwen-t4`.
  Output `hala-h354-private-conservative-paired.json` or one
  categorical stage-only failure, both with strict parsers.
- Frozen source Git commit `37a7224f4886618e44788de92ce48ed0df2114f3`,
  H319 HTML Git blob `6e8467e123c7f0fbac77ea553622ab8e659f9c16`,
  model rev `89644892e4d85e24eaac8bacfd4f463576704203`.

## Pre-registered success and limits

The primary **safety/process** goal is reducing
`pixel_snap_changes` from 12 total actions to fewer while
preserving Arabic native first-click outcomes and avoiding paired
losses. Exact output is unknown until true Kaggle Qwen runs. Treat
a non-identity within-button change as a failure in invented tests,
not an improvement. No training or weights are updated.

Success on 12 PUBLIC/EXPOSED original scenes would only validate
a known-UI postprocessing rule, not a model-training improvement,
unseen independent OOD or deployment readiness. H199 official TRAIN
manifest registry remains EMPTY, H343 rights/title review unsigned,
H173 sealed, H212 barred, H250/H251 quarantined, H293/H294 blocked;
no release or main merge. Only one private, actual FREE T4 Kaggle
run allowed after **full exact-head hosted CI**, verified remaining
free quota, no duplicate or concurrent GPU kernel, private-only
metadata and exact submitted code provenance checks. No paid compute,
no raw images/coordinates/gold/logs in GitHub or user-visible reports.
