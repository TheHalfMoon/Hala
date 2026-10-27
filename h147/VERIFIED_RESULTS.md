# H147 — frozen screenshot-observed native drop entry evaluation

**Research only** on the original MIT React Kanban app, pinned donor
SHA `fa047f7a32afb57750ed5246456dc282ea7befcd`. The original
source remains unchanged; shifted column CSS is owner-browser-only.
No neural model weights were updated or used in this controller QA.

## Independent source and fixed visual entry rule

Use 16 entirely fresh owner original tasks with IDs >=114000
(4 Arabic/Doing, 4 Arabic/Done, 4 English/Doing, 4 English/Done,
four shifted layouts). Both arms use the SAME deliberately
imprecise visible destination-heading point still located inside
its native visible rectangle and perfect visible source card center.
The direct baseline releases at noisy heading center+27.5px.

The new visual-entry controller drags from above the visible heading,
scans downward at frozen offsets, observes actual native screenshots,
and looks for a sufficient pixel change within a coarse destination ROI
around the visible heading. It releases when the signal appears.
**No hidden DOM drop rectangle/class is inspected by the controller.**
The owner reads true persisted source status only after mouse release
and full page reload.

## Genuine Windows original browser outcome

| Arm | Persisted successful card move |
| --- | ---: |
| Direct drag from visible points | **0/16** |
| New visual-entry feedback controller | **7/16** |

There were **7 visual-only successes**, zero baseline-only successes.
All 16 visual-controller attempts observed a pixel-change signal, but
**only 7 tasks actually persisted**. Therefore the feedback signal is
not sufficient proof of a correct native drop.

In the private consumed 16-case diagnostic, successful visual-entry
releases occurred at an offset of **+18 px** from the noisy predicted
heading; failed releases happened at **+26 px**. This is a *post-hoc*
correlation **on a consumed cohort**, not an independently validated
position threshold; do not retest or tune H147 to report a better
percentage. A preregistered **NEW H148 cohort** can instead test the
explicit hypothesis that snapping back to a near-heading entry point
before mouse-up improves persistent task completion.

All original task results remain owner private at
`private-h147/visual-entry-new16-v1.json`.

The source-only pilot performed **0 neural calls**, **0 optimizer
steps**, **0 H140 sealed independent BLIND calls**, no paid GPU,
no flagship Hala 35B modifications, public weights, merge, or release.
