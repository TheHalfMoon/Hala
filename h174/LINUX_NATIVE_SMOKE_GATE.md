# H174 pinned Pickr Linux native smoke — qualification protocol

**Purpose:** obtain an independent real Chromium source-native browser smoke on
a zero-cost public GitHub runner, while owner Windows Abdulaziz is offline.
This is NOT a Windows qualification, TRAIN/DEV capture, dataset admission,
OOD evaluation, model training, or release. This project must remain Draft.

## Immutable original source and isolated build

The workflow in `.github/workflows/h174-pickr-native-linux.yml` checks out
only the public Hala source at the event's test revision (no stored auth),
fetches the exact original MIT donor:
`simonwep/pickr@e8ff88d9161ffc6e3c46a4430be8c328c42d6415`.
It verifies tracked original HTML SHA-256
`4cc4f0dbcb26e79bbf09202ba3896d5d66953c6d40fb6347eac2c15522712d1e`
and LICENSE SHA-256
`7129bc74b781d3db2979d4617940b00e75885914b8d636a3155a64e57481e5e5`.
A clean original Git checkout is archived into a **separate** disposable
scratch tree; only that scratch tree receives npm packages and generated
assets. The build uses Node 24, pnpm 10.27.0, frozen lockfile, pre-fetched
public package tarballs, offline install without dependency lifecycle scripts,
and the *original* `build:lib` and `build:demo`.

## What a successful new runtime would actually prove

If the new dedicated hosted workflow **truly runs to completion** and both
source and browser gates pass, it establishes only a reproducible
Linux/Chromium smoke of the original pinned Pickr donor: open its original
color picker, click an original color swatch, observe a real changed input
value, and match decoded 960×720 RGBA frames from two independent contexts
without saving screenshots. Network interception allows the ephemeral
loopback HTTP port; contexts block Service Workers, downloads and WebSockets.
This is NOT an operating-system firewall; other browser networking mechanisms
may need independent egress controls.

The workflow has no GPU, model or optimizer, has no access to owner-private
gold, screenshots, previous train/DEV/OOD predictions, and does not upload
artifacts. No historical H173/H172/OOD resource is part of the workflow.

**Only a concrete exact-head completed workflow success with inspectable
hosted job steps counts as a passing Linux result.** Existence of this
workflow YAML, success of the generic CPU unit-test workflow, or a run merely
queued/in-progress DOES NOT count.

## Remaining independent gates, even after a Linux PASS

- Re-run exact pinned source build, actual original swatch action and
  independently decoded RGBA replay on owner Windows Abdulaziz with Edge.
- Independently audit source repository history completeness and historical
  PNG/RGBA SHA-only registry. The H174 metadata checks alone cannot prove it.
- Qualify two entirely new source-disjoint TRAIN/DEV original apps with
  real actions and bilingual instructions; reserve a third genuinely
  untouched OOD source family **before** training. Do not capture reserved
  OOD screenshots during development.
- Only then train/compare with free approved compute on TRAIN and new DEV;
  freeze all artifacts; run newly reserved OOD **once** after qualification.
- Keep previous H173 v1/v2 quarantined, H173 v3 permanently consumed,
  and all previous DEV/OOD forbidden for further tuning, evaluation or model
  selection. The H173 2/32 to 3/32 result had McNemar p=1.0, not significant.

No protected-main merge, paid inference, public release or deployment.
