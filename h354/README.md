# H354 — Preserve valid visual button clicks; correct only near misses

**Draft source-exposed research. H354's single private GPU pilot completed on 2026-11-02; no rerun is authorized. No training, blind OOD, or release.**

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
- The completed one-time owner-private free T4 pilot produced an
  aggregate paired outcome. One-time distinct slug:
  `abdulazizshehri/hala-h354-preserve-valid-pixel-qwen-t4`.
  Output `hala-h354-private-conservative-paired.json` was accepted
  by the strict parser; no raw individual outcomes or coordinates are
  disclosed.
- Frozen source Git commit `37a7224f4886618e44788de92ce48ed0df2114f3`,
  H319 HTML Git blob `6e8467e123c7f0fbac77ea553622ab8e659f9c16`,
  model rev `89644892e4d85e24eaac8bacfd4f463576704203`.

## Pre-registered success and limits

The primary **safety/process** goal is reducing
`pixel_snap_changes` from 12 total actions to fewer while
preserving Arabic native first-click outcomes and avoiding paired
losses. The exact pilot outcome is documented below; no additional GPU run is authorized. Treat
a non-identity within-button change as a failure in invented tests,
not an improvement. No training or weights are updated.

Success on 12 PUBLIC/EXPOSED original scenes would only validate
a known-UI postprocessing rule, not a model-training improvement,
unseen independent OOD or deployment readiness. H199 official TRAIN
manifest registry remains EMPTY, H343 rights/title review unsigned,
H173 sealed, H212 barred, H250/H251 quarantined, H293/H294 blocked;
no release or main merge. The one permitted private FREE T4 Kaggle run is complete, following
exact-head CI, free quota, private-only metadata and code provenance
checks. Do not rerun H354 or earlier H351-H353 jobs. No paid compute,
no raw images/coordinates/gold/logs in GitHub or user-visible reports.

## Observed H354 private pilot — 2026-11-02 (not model learning)

The **single** owner-private FREE NvidiaTeslaT4 H354 v1 kernel completed.
The uploaded code was independently pulled from the owner's private Kaggle
kernel and its SHA256 matched the submitted script exactly:
`337DED189A3327E304B705E3FFA2ABA86B10D9A92F6C29B7A1A7EFA2931204FD`.
The owner verified `is_private=true`, the T4 metadata, and the sole
959-byte allowlisted aggregate through `h354.verify_private_receipt.verify_file`.
No raw images, coordinates, per-scene predictions, gold, model logs or
weights are published. The original H319 source was **publicly exposed**
before evaluation.

| Exposed source language | Native pretrained Qwen | Conservative pixel policy | Paired wins | Paired losses | Native actions changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 6/6 | 6/6 | 0 | 0 | 0/6 |
| Arabic | 4/6 | 6/6 | 2 | 0 | 2/6 |
| Combined | 10/12 | 12/12 | 2 | 0 | 2/12 |

**Scientific interpretation:** The within-H354 paired comparison supports
only a preliminary *system-level* correction of two near-outside first
clicks on these twelve already-exposed scenes. With two discordant paired
outcomes (2 wins, 0 losses), two-sided exact McNemar/binomial `p=0.5`;
this is not a statistically established generalizable benefit. The
H353 naive snap pilot changed 12/12 actions while H354 changed 2/12,
but those are **separate model runs**, not a randomized within-output
head-to-head test of H353 versus H354; do not attribute their difference
causally to the policy alone. The 12 scenes come from **one source
family**, not twelve independently blinded families. No model weights
were trained or improved. No rights admission, H199 TRAIN whitelist,
new blind OOD, H340 qualification or release claim follows.

The historical H354 receipt is tied to draft source HEAD
`086992ba10f21e850745f5139f800fb1e8c386bb`. This README
update is documentation-only and **must not be represented as a new
GPU evaluation or an exact-head CI pass**. No H354 GPU rerun.
