# H337 — first-click miss proximity, source-owned and privacy-bounded

**Stage: source-only qualification. No new Qwen model generation,
no GPU, no adaptation, no true isolated Goldfish review.**

## Grounded H336 result and diagnostic question

Owner-private, zero-cost FREE T4 **genuine pretrained** Qwen H336
ran exactly 12 original H319 first-party **public/exposed** screenshot
first clicks, 6 English and 6 Arabic RTL, using the original H323
model-time JSON grammar and native H263 browser click. Its 968-byte
owner-only categorical report was independently verified against the
frozen H336 SHA/model/source and then deleted:

| Source-only first-click outcome | EN | AR RTL |
| --- | ---: | ---: |
| correct button opened AND input focused | 6 | 4 |
| correct opened but input not focused | 0 | 0 |
| wrong station opened | 0 | 0 |
| no station opened | 0 | 2 |
| abstain | 0 | 0 |
| Total original exposed scenes | 6 | 6 |

These counts are for **first click**, not complete click+SIM code entry.
A previous genuine H333 matched pretrained prompt study showed **no
improvement** from static RTL guidance (EN2/2 vs2/2; AR0/2 vs0/2). H336
localizes failures more tightly: no wrongly opened station and no
correct-opened-but-unfocused case in this sample. It does not tell us
how far outside the intended button the two Arabic clicks landed.

**H337's falsifiable, prospective question**: when an original
source-native click opens no station, is the predicted click
(a) on the intended button but inert/occluded, (b) within 12 image
pixels of its edge, (c) 12–48 pixels away, or (d) more than 48 pixels
away? These are frozen coarse source-only classes, not per-case model
action coordinates. The distance threshold is measured on the original
source's **960×720** visible button rectangle, not on text direction.
A near-edge concentration would suggest coordinate/point calibration
to investigate; a distant miss would require genuine visual target
localization study. Both remain hypotheses, not conclusions.

## Trusted evaluator and boundaries

`h337/source_miss_distance_bands.py` uses the original pinned H319
HTML, offline fresh browser source context and first screenshot,
H319 screenshot integrity, H199 parsed native single action and H263
real click. The model-facing callback receives **only screenshot bytes
and user-visible task text**. The source-owned evaluator separately
knows the scene's correct station and may read the DOM to score the
result, but these must **never** enter a model prompt, policy, report,
or public PR artifact. It first uses H335's strict source-state/focus
classification; it consults visible source button bounding rectangles
only if no station opened after a native click. Source data and actual
action points stay in memory; the only returned value is a category.
Impossible rectangles, invalid native points and impossible focus/DOM
states fail closed.

The prospective 12-scene `audit_source_distance` exposes **only
aggregate two-language category counts** across eight whitelisted
classes: correct-open/focus, correct-open/wrong-focus, wrong station,
abstention, miss inside expected rectangle, miss within 12px,
miss 12–48px, miss farther than 48px. No per-scene model prediction,
target ID, point, box, task text, DOM, PNG, source gold, prompt text,
weights or private Kaggle report is written, read back, returned, or
persisted. Source-owned fixtures can demonstrate the evaluator's
correctness but **do not** attest Qwen performance.

## Test scope

```bash
python -m unittest discover -s tests -p test_h337_source_miss_distance_bands.py -v
```

Five invented-only CPU tests validate boundary classes, arithmetic,
impossible coordinates/boxes, privacy, wrong roles and strict aggregation.
Two opt-in real original browser tests under
`HALA_H337_SOURCE_BROWSER=1` run the 12 source-owned correct center
clicks plus first-party EN/AR invented near-edge, far, wrong-button
and abstain controls with Ubuntu Chromium and Windows Edge.
These test-driven source clicks are **not genuine model actions**.

To run a meaningful H337 model-time test in the future requires a
separately reviewed, owner-private, free-only pinned Qwen experiment
that uses this exact source evaluator and a strict receipt contract
built before any GPU dispatch, after verifying fresh quota, source
rights and protocol. **No such new job is created by H337.** Never
replay private H336 model points, inspect raw Kaggle logs or unseal
H173 OOD to populate distance buckets.

H199 admitted TRAIN is still empty. No lawful first-party independent
multi-source TRAIN/DEV admission, independent source-family-blind OOD,
or measurable paired BASE-vs-ADAPTED GUI learning gain exists. H293/H294
blocked private data retrieval/provenance, H250/H251 quarantined rights
and founder no-release/no-merge constraints remain intact.
