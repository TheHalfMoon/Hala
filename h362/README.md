# H362 — source-candidate bilingual shortcut / confound repair

**SOURCE-ONLY: no lawful TRAIN admission, no learning, no blind OOD or release.**

H357/H358 are original from-scratch PUBLICLY EXPOSED browser sources
intended only as TRAIN/DEV review candidates. Earlier H360 repaired
ControlGarden's target-slot parity leakage, but a follow-up source-only
contingency audit found two remaining defects:

1. **H357 target shape leaked through language:** for English the 32
   target shapes had counts circle 7, square 11, triangle 5,
   diamond 9; for Arabic they were circle 10, square 3,
   triangle 13, diamond 6. Slot/color marginal balance alone did
   not protect language-conditional shape labels.
2. **H358 metadata leaked language:** English cases previously used
   *only odd* displayed workspace IDs, Arabic cases *only even* IDs.
   The first letter of each invented reference code also selected
   English versus Arabic by parity. That made task language predictable
   from nonlinguistic attributes without reading the localized UI.

## Concrete repair

ControlGarden now uses a deterministic constrained permutation
generator, keeping **all 64 source layouts distinct** while balancing
conditional distributions for **each language separately**:

| Label dimension | For each language |
| --- | --- |
| Target slot (0–3) | 8 occurrences each |
| Target shape (4 shapes) | 8 each |
| Target color (4 colors) | 8 each |
| Target (slot, shape) pair | 2 each |
| Target (slot, color) pair | 2 each |

The actual visible source application and all four individual CSS
shapes remain unchanged. Non-target tiles are permuted using a
deterministic seed; gold is obtained from the original browser and
never exposed to a model as DOM/source code.

NoteRelay now has **16 exact mirrored bilingual pairs**, each with
the **same nonlinguistic code, workspace and theme** while English
and Arabic source UI text/layout differ. There are still **32
distinct full rendered cases, 16 per language**, four native actions
each (128), 96 clicks + 32 text actions. There are intentionally
only **16 distinct invented codes**, each seen once in each
language; the previous 32 unique codes made language a function
of the exact synthetic value. Case IDs are still 32 unique.
No real account or patient data is present.

## Limits of the repair

- Conditional marginals and these pairwise contingency tests are
  mathematically balanced *for these 64 authored samples*.
  **They do not establish source-family diversity, full factorial
  independence, absence of all learnable shortcuts, good unseen
  generalization or actual trained-model improvements.**
- Both sources remain public and permanently exposed, not a
  new independent blind DEV/TEST set.
- H359 H199-native source audit, H361 processor-only contract and
  H199 source rights/data admission guards remain intact.
  The **real Qwen3-VL CPU processor parity previously observed in
  H361 pertains to the preceding generated scene configurations**.
  This H362 revision changes raster/case contents, so a
  separate real pinned-processor requalification on these new
  contents is required before claiming actual real processor
  parity for H362.
- Existing original source HTML Git SHA identities are unchanged;
  only original source candidate factory parameters are updated.
  No data/image/gold/weight files are committed.
- H199 official TRAIN manifest SHA list is still EMPTY; H355
  external reviewer trust and H356 independent OOD custody
  trust registries EMPTY. Open external issues #384 and #385
  still require independently qualified humans and provenance.
  No external legal permission, GPU model training, weights,
  optimizer or model evaluation is authorized.

## Qualification

```sh
python -m unittest tests.test_h362_source_balance -q

HALA_H357_BROWSER_QA=1 HALA_H358_BROWSER_QA=1 \
python -m unittest \
  tests.test_h357_original_controlgarden_candidate \
  tests.test_h358_original_noterelay_multistep -q
```

The dedicated H362 4-way GitHub CI runs actual original Edge/
Chromium native browser outcomes on Windows/Ubuntu with Python
3.11/3.12. Screenshots and source-only gold are generated under
`TemporaryDirectory` and removed immediately, never sent as
external datasets or stored in the repository.

Release readiness remains **NO-GO** until independently certified
training rights and an actual trained adapter with a new, truly
independent sealed source-family-disjoint bilingual benchmark exist.
