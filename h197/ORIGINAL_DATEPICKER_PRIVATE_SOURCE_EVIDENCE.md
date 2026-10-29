# H197 — original Datepicker owner-private research quarantine

**Strictly private and strictly research-only. Zero TRAIN/DEV/OOD cases admitted.**

## Dependency and original source isolation

H197 combines the exact H196 Draft browser egress isolation head
`954fe5807984209c907c74a947844adbf1b47b4f`
and the qualified H195 original source head
`372a8d472c312906042bc51f5212b3160182a57f`
through a non-force, two-parent owner Git merge. It does **not**
change those Draft PR branches or protected main.

Original MIT app and immutable pin:
`mymth/vanillajs-datepicker@358dfaaa3e2b8772e313d71191720f5415c9275c`.
The original application, its JavaScript, CSS and license are untouched.
H195 fixes its browser clock to 2026-10-10 for deterministic
original on-screen month/day visibility, and the H196 browser
context disables Service Workers, downloads and WebSockets and
aborts non-loopback HTTP before any page is created.

H197's owner-private callback is opt-in. It is triggered **only after**
the original browser's date selection has yielded the expected
native input date and a changed GUI raster. In-memory callbacks alone
are not proof: the original source runner performs actual page actions
in two separate fresh browser contexts for each bilingual task
(October 15 and November 22, 2026), checks identical PNG and
decoded RGBA screenshots and reference coordinates, and screens
known historical SHA reuse. The owner writer independently repeats
the old SHA screening before writing anything.

## Cohort exclusion and output contract

The owner-only writer reads **metadata SHA and source identity only**
from H188 (2), H190 (2), H191 (12), H194 (2): 18 existing distinct
new original source captures. It NEVER opens their images, owner
references or private gold. It also screens the bounded old
historical H185/H186 manifest PNG and existing RGBA SHA-only fields.
This is known-old collision exclusion, NOT a complete historical
source or visual-collision proof.

On the owner device only, after all native browser and SHA checks:
`D:\Hala-H197-Private` holds two 960x720
original PRE-action PNGs, a public-code-compatible
`source-prospects-not-admitted.json` containing source
instruction/language/PNG/RGBA hashes and all eligibility flags false,
plus a SEPARATE
`OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json` with the true native
click target and date reference.

An independent H197 audit opens **only the two NEW H197 images**,
rehashes both encoded PNG and decoded RGBA, checks source pin and
language/instruction, detects metadata tampering, private reference
leakage and false DEV eligibility, and compares the hashes against
the previous 18 private new sources and known older historical
manifest SHA values. The public GitHub code and CI never contain
the screenshots or private target coordinates.

## Strict limits and remaining scientific work

The new Datepicker family remains
`RESEARCH_UNASSIGNED_NOT_ADMITTED`, not newly assigned TRAIN,
DEV, nor held-out OOD. Its novelty check passed against 22
surviving owner historical Git checkouts and public family
blacklists, but old source/fork/npm and decoded RGBA completeness
are NOT proven. Prior H173 scored OOD remains consumed, and
tentative original Frappe Gantt source-only OOD remains unopened.
Neither the current private images nor any model can be
submitted for training/scoring until the science and data gates
are truly satisfied. H197 runs **zero inference/model/GPU steps**.

Windows actual capture results, independent raster rehash results,
owner test counts and exact-head hosted CI should be recorded in
Draft PR once independently confirmed; synthetic tests cannot
substitute for live browser and private saved-file evidence.
