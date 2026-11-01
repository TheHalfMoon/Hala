# H312 — pinned original-source rights and rendering dependency triage (public evidence only)

**Decision: BLOCKED / NO TRAIN, DEV, OOD ADMISSION.** This is a dated source-evidence inventory, not legal clearance, a browser-network trace, a complete Git/fork/npm history, a screenshot audit, a model result, or permission to publish media. No original third-party source/media bytes, screenshot, gold, private receipt, image hash, credential or model weight is included.

## Exact public sources independently inspected, 2026-11-01

| Original app | Immutable source commit | Public files inspected | Finding |
| --- | --- | --- | --- |
| H250 [simple-to-do-list](https://github.com/Serkanbyx/simple-to-do-list) | `009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff` | [tree](https://github.com/Serkanbyx/simple-to-do-list/tree/009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff), [index.html](https://github.com/Serkanbyx/simple-to-do-list/blob/009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff/index.html) | Original HTML directly references a Google Fonts CSS URL for **JetBrains Mono** and **Outfit** plus two preconnect hints. The source tree has no image/font asset files; inline SVG is not thereby automatically rights-cleared. |
| H251 [LIfe-Os-Free](https://github.com/3mmar404/LIfe-Os-Free) | `b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c` | [tree](https://github.com/3mmar404/LIfe-Os-Free/tree/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c), [Arabic index](https://github.com/3mmar404/LIfe-Os-Free/blob/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c/index.html), [English index](https://github.com/3mmar404/LIfe-Os-Free/blob/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c/index-en.html) | Both original HTML pages reference local `assests/icon.ico` and `assests/logo3.png`, remote **Font Awesome 6.5.2** CSS, and Google Fonts CSS. Arabic requests **IBM Plex Sans + Tajawal**; English requests **IBM Plex Sans + Inter**. These are *declared references*, not proof of network loading or legal permission. |

The H251 [pinned package.json](https://github.com/3mmar404/LIfe-Os-Free/blob/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c/package.json) has empty npm dependency maps but its `repository.url`, `bugs.url`, `homepage` and `funding.url` use generic `username` placeholders, not the actual pinned owner. **Therefore the package metadata is not an independent attestation of origin or complete npm lineage.** Its [README](https://github.com/3mmar404/LIfe-Os-Free/blob/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c/README.md) names Font Awesome and notes initial font/icon loading. The [root MIT LICENSE](https://github.com/3mmar404/LIfe-Os-Free/blob/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c/LICENSE) is evidence of the repository's declared source license, **not** proof that the repository author owns every bundled image/logo or remote font/icon. H250's existing H259 Heroicons path research remains partial, not blanket notice compliance.

## Why the source-native research cannot yet become TRAIN/DEV

1. **Visual-asset chain of title unresolved.** Obtain independent license/attribution and redistribution/training-rights evidence for each actual rendered asset and any derived icon. In particular H251 local logo/favicon and declared CDN icon/font families require per-asset review; the presence of an MIT root LICENSE is insufficient to certify third-party ownership. This is a rights-review request, not an assertion that these assets are unlawfully used.
2. **Rendered-state fidelity unresolved.** The prior H250/H251 source-native replays intentionally blocked external requests. The HTML nevertheless declares remote fonts/icons. No claim is made that the remote glyphs loaded, nor that a screenshot of offline fallback rendering represents the complete online app. A future admissible protocol must pin the exact *actually rendered* asset set, font fallback, viewport, browser and network policy without uploading media.
3. **Historical source independence unresolved.** Current exact commit and root LICENSE do not prove complete deleted Git history, forks, package mirrors, prior scored screenshots, or RGB lineage. H253's static parser is not an exhaustive JavaScript/runtime dependency graph; H259 is icon-specific. H309's 36/36 distinct **first-party exposed** RGB frames does not address these third-party families or semantic source-disjointness.
4. **Custody and outcome gates unresolved.** Any future candidate requires independent rights reviewer, source-family review, lawful TRAIN/DEV split admission, separate untouched blind OOD custodian, and paired BASE/ADAPTED native task-state evaluation. H199 `ADMITTED_TRAIN_MANIFEST_SHAS` stays empty. H173 OOD stays sealed; H212 barred; H293/H294 safety blocks are not retried.

## Concrete decision ledger

| Gate | H250 | H251 | Admission impact |
| --- | --- | --- | --- |
| Original pinned public source and root MIT declaration | Documented | Documented | Necessary, not sufficient |
| Declared external CSS/font/icon references | Google Fonts | Google Fonts + Font Awesome | Independently clear rendered dependencies |
| Local app-specific visual assets in original HTML | None referenced by HTML (inline SVG remains) | Logo PNG + favicon ICO | Review asset authorship and rights |
| Complete runtime dependency and rendered pixel inventory | Not proven | Not proven | **BLOCK** |
| Complete historical fork/npm/source and image-lineage independence | Not proven | Not proven | **BLOCK** |
| Independently reviewed lawful TRAIN/DEV admission | **NO** | **NO** | **BLOCK** |
| New untouched, separately custodied blind OOD | **NO** | **NO** | **BLOCK** |

**Research-only status:** `H312_PUBLIC_PINNED_RIGHTS_TRIAGE_INCOMPLETE_NO_ADMISSION`. This document is intentionally non-executable; it neither consumes private media nor accepts a self-declared checkbox as legal proof. No Qwen generation, optimizer step, GPU experiment, model improvement, or release is claimed.

**Next real evidence unit:** independent per-asset rights/attribution and complete original source/fork/npm/history review, plus a safe actual-rendered-resource manifest from a future authorized original-app session. Do not substitute synthetic examples or publicly exposed first-party tasks for blind OOD.
