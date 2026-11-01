# H309 — actual exposed-source RGB split integrity audit

## Research question

H308 checks that the 24 public ShiftDesk TRAIN **candidates** have unique
model-visible RGB frames. It does not establish that frames from a
different public source family are disjoint. A PNG-byte or RGBA comparison
is insufficient: metadata and alpha-only changes can preserve the RGB
pixels actually supplied to Qwen.

This opt-in source-only audit runs the original H285 browser producer,
including all 24 real native source-state actions, and captures the 12
original H296 FieldPort initial-state screens (six English, six Arabic).
It compares only ephemeral decoded 960x720 RGB SHA-256 digests in memory,
rejects intra-role duplicates and cross-role RGB collisions, and returns
only bounded aggregate counts. It never stores or prints screenshots,
prompts, task text, labels, coordinates, case IDs, or hashes.

## Qualification

```bash
python -m unittest discover -s tests -p test_h309_exposed_source_rgb_split.py -v
```

The opt-in real-source test requires
`HALA_H309_BROWSER_QUALIFY=1` and Playwright Chromium, or
`HALA_H309_BROWSER_CHANNEL=msedge` on Windows. Dedicated GitHub Actions
runs on Linux Chromium and Windows Edge. All other tests use invented
in-memory pixels only.

## Scientific limits

This tests **source-only visual disjointness**, not semantic task
independence, licensing, official admission, model inference, adapted
performance, or blind OOD. Both source families and every task are
PUBLIC/EXPOSED. H199's admitted TRAIN manifest remains EMPTY; H173 old
scored OOD is sealed; H212 is barred; H250/H251 remain quarantined.
A passing 24+12 source audit cannot make these sources officially admitted
or support a generalization/release claim. The historical LoRA paired
BASE6/8 versus ADAPTED6/8 result remains zero gain.

No GPU, weights, owner-private Kaggle outputs, H293 blocked receipt, H294
blocked provenance write, or owner Windows device is used.
