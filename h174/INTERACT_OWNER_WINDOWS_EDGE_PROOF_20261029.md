# H175 owner-Windows original interact.js Edge qualification — 2026-10-29

**Research-only proof; no model training, evaluation, or data admission.**

## Identity and scope

- Parent Hala Draft PR: [#191](https://github.com/TheHalfMoon/Hala/pull/191), exact tested Hala head `e50e917cf8e4cdf2b594e4e590009d4c0aaed5d4`.
- Original upstream family: `taye/interact.js`, pinned upstream Git SHA `a993eb8cd5cbc65f3a19e45ee24b7fedc026174f` from H174 immutable candidate catalog.
- Local owner device: **Abdulaziz**, Windows 11 Home, Microsoft Edge executable version `156.0.4314.50`, Python `3.11.15`, Node `24.19.0`.
- Isolated test environment: `Pillow==12.3.0`, `playwright==1.64.1` installed in a throwaway Python virtual environment on the D: drive.
- Source under test: an existing scratch build at `D:\Hala-H174-Interact-Win-Build-20261029\source\.h174-derived-dropzones` generated using the **pinned original Vite**, original config and original `examples/dropzones/index.html`. This run checked source HTML and license file SHA-256 against the H174 catalog; it did **not** independently rebuild the entire dependency graph from scratch. The latter is a separate CI reproducibility gate.

## Observed native run

- A Python loopback-only static server served the built original HTML on `http://127.0.0.1:18181/dropzones/index.html`. No HMR, outside network, uploaded media or saved screenshots.
- Invoked the existing Hala `h174.interact_native_smoke` against the installed **msedge** browser channel, not headless bundled Chromium.
- Browser context #1: native original drag `#drag3` into original `#drop1`, original `Dropped` event confirmed, 960×720 RGBA captured in memory.
- Browser context #2: independent isolated context, same real drop event and RGBA capture confirmed.
- Exact independent RGBA byte SHA-256 in both contexts: `5fc9042de19971666a156e628f25aac92649e7fa80db3ba681bb58cf745de628`.
- Output gate: `H174_INTERACT_ORIGINAL_NATIVE_DROP_PASS ... independent_contexts=2` plus `H175_OWNER_EDGE_NATIVE_QUALIFIED_NO_DATASET_ADMISSION`. Process returned **exit 0**; local static server explicitly stopped.

This is a **native Windows original-app behavior and replay pass**. It does *not* qualify the whole model, a checkpoint, all donor families, or the historical-source registry. Unlike the Linux GitHub-native CI, the existing Windows scratch build was reused, so do not claim independent clean Windows build reproducibility from this owner receipt alone. The separately proposed Windows GitHub-hosted workflow must establish that boundary independently.

## Reproduce without persisting private data

On Windows with the same Hala commit and a verified scratch source tree/build, run a loopback-only `python -m http.server` against the derived build directory and:

```powershell
$env:PYTHONPATH = 'D:\Hala-H174-PR191-Edge-QA-20261029'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\Hala-H175-OwnerEdge-20261029\.venv\Scripts\python.exe' -u -m h174.interact_native_smoke --port 18181 --browser-channel msedge
```

Never reuse this generated QA material as TRAIN/DEV/OOD samples. H173 OOD v3 is consumed and must never be rescored as unseen. No new performance claim, GPU execution, publication, release, or `main` merge follows from this result.

## Remaining gates

1. Clean exact-source and original-lock Windows-hosted build plus the same original Edge event/RGBA replay (separate workflow, no substitution of synthetic controls).
2. Original Pickr/medium-zoom owner Windows source-native qualification at exact independent evidence heads.
3. Complete historical source-family inventory, source-disjoint TRAIN/DEV and reserved unseen OOD, before any new data admission.
4. Fresh free-GPU model update, immutable adapter checks, genuinely untouched OOD and statistical comparison, only after source and split gates pass.
