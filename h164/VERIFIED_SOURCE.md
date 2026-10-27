# H164 — fully qualified original-application source cohorts

**2026-10-28: original-source/data qualification COMPLETE. The neural model is NOT YET TRAINED or scored on the independent OOD holdouts.**

## Independently verified split

| Split | Total | Distinct original apps | Source replay | Gold available to model |
| --- | ---: | --- | --- | --- |
| TRAIN | 384 | 5 | 96/96 + 288/288 exact Chrome PNG | Yes, TRAIN-only |
| DEV | 96 | 2 | 96/96 exact Chrome PNG | No |
| OOD | 64 | 2 new apps (Pell, Pikaday) | 64/64 exact Chrome PNG | No |

Each case includes the exact original unmodified application runtime and the actual successful reference browser action. The Calculator's GPU-shadow rasterization is normalized for new TRAIN/DEV browser screenshots only, with the donor source and native controls unchanged. See [VERIFIED_TRAIN.md](VERIFIED_TRAIN.md) for rejected candidate details and true train-source evidence.

The final independent owner-only auditor `h164.audit_full_qualify` passed **`H164_FINAL_384_TRAIN_96_DEV_64_TRUE_TWO_NEW_APP_OOD_ADMITTED`** and verified no overlap across splits, enough original app families, proper bilingual coverage, valid source scenes and separate owner reference records.

Source owner-private manifests (SHA-256):
- first 96 TRAIN: `887eddd886d7f0505fcca4afd79ee96ce0dadf2443c72c0e5932f418e751d577`
- additional 288 TRAIN: `d2d2b8231a2995fb2b9e7c0d9ee4048c5e21152fa558103e344dc10697572104`
- 96 DEV: `ffdcc3e20435ec9f9048ad23570f8cf1297ff430c37db36b33e75c4ec78b0f34`
- 64 OOD: `3991d8b5b200ef9e80c5567534a6637df3df3319686ae9fe2c05b85cd7e75fd4`

Frozen owner-private Kaggle ZIPs verified with absolute SHA (NO raw media or owner gold in GitHub):
- 384 TRAIN: `ed08ef5a7e8b38d6fe8c38c235d93d8fa0625437ca3c67a5578f4fc941db4495`
- 96 DEV: `17adcc78fae62d7de3ae73c2ae7d3863abd0b2f1105edf4f45db7f4c78b503b0`
- 64 OOD: `6056b16ef9f6ef86ec2a514e7d71a6d2840d0930dcf770ffa198e1367f444a85`

Parent external Qwen3.5-2B research H161 rank8 adapter SHA `72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c` remains frozen and separate from the flagship Hala 35B.

**Experimental scope:** `h164.train_gpu` defines 192 real optimizer steps / accumulation 2 using all 384 TRAIN images, one exposure each, with frozen language LoRA and strictly free Kaggle T4. DEV inference never includes owner gold, and OOD is NEVER opened during training. No model improvement claim is supported until an actual GPU receipt and real original-browser native app task reward are independently audited. Do not tune on consumed H140, H162, H163, or the new H164 DEV/OOD.

No publication, paid GPU, flagship Hala 35B changes, main merge or release. Keep PR Draft.
