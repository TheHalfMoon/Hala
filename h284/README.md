# H284 — match private H282 Kaggle kernel title to actual slug

During the first **genuine**, owner-private FREE noncommercial Kaggle
T4 H282 ClinicGrid pretrained experiment, the CLI warned that the
submitted metadata title would be normalized to a different URL than
its requested `id`. Kaggle successfully created **kernel V1** at:

`https://www.kaggle.com/code/abdulazizshehri/hala-h282-private-pretrained-clinicgrid-t4`

This exact V1 run COMPLETED and produced the original two-case
**real pretrained Qwen** ClinicGrid research result, validated
independently by H283: **2/2 true original source-owned task-state
successes** (English urgent C-214 and Arabic RTL C-308),
both true image-conditioned generations and genuine native clicks.
This is **previously exposed first-party research**, not hidden OOD,
trained Hala gain, admitted TRAIN/DEV or release qualification.

## Source-only repair, no duplicate GPU jobs

H284 corrects only the *future upload packager's* `KERNEL_ID`
to the URL **Kaggle actually created**:
`abdulazizshehri/hala-h282-private-pretrained-clinicgrid-t4`.

The unchanged title string is now declared as one explicit
`KERNEL_TITLE`. The CPU test checks its lowercase URL slugification
matches the actual owner kernel ID, preventing future accidental
second notebook creation or version confusion. It does NOT rename,
overwrite, re-submit, or re-run the completed V1 kernel.

The V1 source identity stays exactly as submitted from H282
`aa104c9f3405b67c757f14e28bcf37319ac4571e`, with original H281
source commit `ab6d4c77ee89a6abfb4375edbba4f1a2108db445`.
The 1,379-byte private receipt and empty CLI log were checked against
the H283 closed schema and then deleted from owner Windows D: scratch,
deletion verified; no private artifacts/coordinates/screenshots/
model weights were committed.

## Owner Windows qualification

```powershell
python -m unittest discover -s tests -p test_h282_private_clinicgrid_probe.py -v
python -m unittest discover -s tests -p test_h283_clinicgrid_evidence.py -v
python -m compileall -q h282 h283
```

At exact source fix head
`2aabbc38a594fcff206d7d7b6f1c55de25617fe5`,
owner Windows passed **10/10 H282** synthetic source/packager tests and
**10/10 H283** independent receipt validation tests; Python compileall
passed. These local tests are source-only and must never be called a
model accuracy measurement. Real browser and full hosted CPU
qualification is tracked separately.

This branch remains DRAFT, stacked over H283. Do not merge into
protected `main`, deploy, publish weights, use paid compute or
re-use the exposed first-party ClinicGrid tasks as blind OOD.
