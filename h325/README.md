# H325 — Original synthetic two-step TRAIN-candidate factory

**TRAIN CANDIDATES ONLY. Not officially admitted, not DEV/OOD or proof
of an improved model.** This original author-written 960×720 RailKit
browser family has zero external fonts, media, images, accounts, real
records, patient data, downloads, APIs or network-dependent assets.
This source is **distinct** from the previously scored/publicly exposed
BeaconBoard H319/H324 family and the single-step ShiftDesk H285 family.

The source's exact pinned Git blob is
`743b5e1b54774a1deea4dbe6bfbe0613adead7e4`; `.gitattributes`
locks HTML checkout bytes across Windows and Linux. Sixteen frozen
fictional source scenes cover 8 English + 8 Arabic RTL, four orders,
light/dark styles, four target rail IDs and harmless invented `LAB-NN`
codes. Each scene has TWO different visual stages:

1. Screenshot of visible four-button RailKit. Trusted TRAIN-only
   browser extractor obtains the visible target BUTTON bounds and a
   strict native click label; it dispatches that click and checks that
   **only** the intended station opens and focuses its new input.
2. Fresh, decoded 960×720 RGB screenshot of that newly revealed field.
   The trusted TRAIN-only extractor builds a bounded fictional
   `type_text` label, dispatches it natively and verifies all four
   source state records reflect exactly the intended entry.

All 32 model-visible RGB images must be different from one another,
and BOTH screenshots must remain unchanged between the trusted oracle
and their native dispatches. **No candidate reaches any sink until
ALL 16 source scenes, 32 native actions and screenshot uniqueness checks
have succeeded.** A caller's sink must be in-memory and trusted,
never a model/OOD evaluator or a file/network writer. No images, gold
coordinates, prompts, completion targets, source artifacts or receipts
are persisted by the factory.

Run source-only tests with no browser opt-in:

```sh
python -m unittest discover -s tests -p test_h325_original_two_step_train_candidates.py -v
```

Real qualification requires browser/Pillow dependencies and
`HALA_H325_BROWSER_QUALIFY=1`; use
`HALA_H325_BROWSER_CHANNEL=msedge` for Windows Edge. CI includes both
Linux Chromium and Windows Edge. Failure of any screenshot capture,
source pin, native action, image distinctness, role or label parser
**terminates without producing candidate output**. This qualification
does not run a model or train any adapter.

H199 official `ADMITTED_TRAIN_MANIFEST_SHAS` remains EMPTY. The
authorial/source history and eligibility of the produced frames for
later offline training still require independent review. Keep all
candidate material disjoint from BeaconBoard public evaluation; no
blind OOD can be established by either public source. The historical
H173 OOD remains sealed, H212 barred, H250/H251 asset rights
quarantined, and H293/H294 private retrieval safety blocks unchanged.

Genuine H324 pretrained BeaconBoard ran on owner-private FREE T4, and
the *strictly validated aggregate* showed 1/2 source tasks, three
model-time generations, three native actions. That is an exposed
**pretrained baseline** only. This H325 source generation cannot claim
to improve Qwen or authorize new LoRA/GPU or release.
