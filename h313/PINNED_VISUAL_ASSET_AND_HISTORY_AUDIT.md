# H313 — pinned external visual-asset and commit-history audit (public metadata only)

**Decision: BLOCKED.** This is a read-only inspection of GitHub metadata at the immutable H250/H251 source pins. It is not legal clearance, proof of originality, exhaustive history/fork search, a runtime resource trace, model evaluation, or TRAIN/DEV/OOD admission. No donor media bytes, screenshot, raw prediction, gold, task text, image hash, credential, or model weight is copied here.

## Sources and scope (independently inspected 2026-11-01)

- H250: [Serkanbyx/simple-to-do-list at `009c7880`](https://github.com/Serkanbyx/simple-to-do-list/tree/009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff). [Pinned recursive Git tree](https://api.github.com/repos/Serkanbyx/simple-to-do-list/git/trees/009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff?recursive=1) returned `truncated=false`, 18 paths, **zero checked-in PNG/WebP/ICO/JPEG files**. Inline SVG, CSS, and remotely declared Google Fonts remain separate rights/runtime questions. The visible default-branch commit listing returned four commits, with a single initial root and a linear ancestry at the observed pin. This does **not** prove the absence of unlisted branches, mirrors, or earlier external origins.
- H251: [3mmar404/LIfe-Os-Free at `b5fcb78c`](https://github.com/3mmar404/LIfe-Os-Free/tree/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c). [Pinned recursive Git tree](https://api.github.com/repos/3mmar404/LIfe-Os-Free/git/trees/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c?recursive=1) returned `truncated=false`, 49 paths. **Eleven checked-in image-file paths represent ten distinct Git blob identities**: three under `assests/` (ICO, PNG, WebP) and eight under `screenshots/` (seven PNG and one WebP). `assests/poster.webp` and `screenshots/poster.webp` point to the same Git blob; count them as one byte-level asset, not two independent visual sources. This is a Git-blob identity observation, **not** a decoded-RGB or cross-family duplicate audit. No image contents were downloaded or decoded.

## Historical changes relevant to source custody

The [H251 default-branch commit listing](https://api.github.com/repos/3mmar404/LIfe-Os-Free/commits?per_page=100) returned **16 commits**, all within 2025-07-28, with a single root and a linear parent chain in the observed listing. The [initial commit](https://github.com/3mmar404/LIfe-Os-Free/commit/0e9cbb6190ca7d4817587d52acc16ec47782d17d) already contained the three `assests/` image paths. The [comparison from `b154dd80` to the pin](https://github.com/3mmar404/LIfe-Os-Free/compare/b154dd80a5151e87416fe2d3b7b64e4019ec173a...b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c) shows the eight `screenshots/` image paths added later in that visible history. The [`b154dd80` commit](https://github.com/3mmar404/LIfe-Os-Free/commit/b154dd80a5151e87416fe2d3b7b64e4019ec173a) removed `package.json`; the comparison shows `package.json` later added back. Thus a **single current package manifest cannot substitute for historical package/source custody**. These events alone do not imply misconduct or a license violation.

Both repository API metadata objects currently report `fork=false`, which means only that GitHub does not mark these repositories as forks; it **does not establish** independent authorship, absence of copy/paste, mirror history, asset ownership, or training permission. The observed commit-list endpoint is not an exhaustive audit of deleted refs, upstream mirrors, npm publications, CDN contents, or browser runtime requests.

## Independent-review gate and next evidence request

| Requirement | H250 | H251 | Decision |
| --- | --- | --- | --- |
| Immutable pinned tree and observed default-branch commit metadata | Inspected | Inspected | Evidence only |
| Checked-in image path inventory | No PNG/WebP/ICO/JPEG files at pin | 11 paths / 10 Git blob identities | Not a rights attestation |
| Original authorship/license of inline SVG, icon, font, and visual assets | Unverified | Unverified for local assets and remote dependencies | **BLOCK** |
| Historical source/package/fork and actual rendered-resource custody | Incomplete | Incomplete; observed manifest removal/re-addition | **BLOCK** |
| Rights-reviewed source-diverse TRAIN/DEV and untouched blind OOD | None admitted | None admitted | **BLOCK** |

**Next substantive unit:** obtain an independently reviewed per-asset provenance/attribution record (including the H251 image assets and declared external font/icon dependencies), followed by an authorized actual-rendered-resource inventory with explicit network/fallback policy and complete historical/source-disjointness review. Do not infer rights from a root MIT license or a `fork=false` API field. Do not run GPU or use exposed research screens as blind OOD.

**Invariant:** H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains empty; H173 sealed, H212 barred, H250/H251 quarantined; H293/H294 safety blocks remain in force. This report adds no code path to override admission, no training, no Qwen inference, and no release claim.
