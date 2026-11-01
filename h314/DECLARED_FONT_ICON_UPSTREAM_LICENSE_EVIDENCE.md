# H314 — upstream licenses for declared font/icon dependencies (public metadata only)

**Decision: H250/H251 remain QUARANTINED; no TRAIN/DEV/OOD admission.** This note independently checks upstream license **text**, not the licensing of a particular browser-fetched font file, the source application's logo, the actual rendered resources, image-model training rights, or the legal chain of title. It does not copy font, icon, screenshot, or third-party app media bytes. Inspected 2026-11-01.

## Exact declarations from H312's immutable original app pins

- **H250** [Serkanbyx/simple-to-do-list at `009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff`](https://github.com/Serkanbyx/simple-to-do-list/tree/009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff) declares Google Fonts **JetBrains Mono** and **Outfit** in original HTML.
- **H251** [3mmar404/LIfe-Os-Free at `b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c`](https://github.com/3mmar404/LIfe-Os-Free/tree/b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c) declares Google Fonts **IBM Plex Sans** and **Tajawal** in Arabic, **IBM Plex Sans** and **Inter** in English, plus Font Awesome Free **6.5.2** CSS. Local `assests/icon.ico` and `assests/logo3.png` are separate unverified app-specific assets.

## Independent primary-source license inspection

The five Google Fonts OFL texts below were fetched as public GitHub file contents, with the exact **Git blob SHA** recorded to make the inspected license text content-addressable. The `main` directory path is mutable; a blob identity proves only which license text was inspected, **not** which font binary was actually fetched or rendered at runtime.

| Declared font | Primary upstream OFL file | Inspected Git blob SHA | Observed declaration |
| --- | --- | --- | --- |
| JetBrains Mono (H250) | [google/fonts/ofl/jetbrainsmono/OFL.txt](https://github.com/google/fonts/blob/main/ofl/jetbrainsmono/OFL.txt) | `821a3dac22aff15a1f1c9689a1d79c45bb58ca39` | SIL Open Font License 1.1; JetBrains Mono Project Authors |
| Outfit (H250) | [google/fonts/ofl/outfit/OFL.txt](https://github.com/google/fonts/blob/main/ofl/outfit/OFL.txt) | `723cd447edee83ed3bb07c3343f478d352aa7440` | SIL Open Font License 1.1; Outfit Project Authors |
| IBM Plex Sans (H251) | [google/fonts/ofl/ibmplexsans/OFL.txt](https://github.com/google/fonts/blob/main/ofl/ibmplexsans/OFL.txt) | `c35c4c618fab33da8695177b3a6cefe0810b7b28` | SIL Open Font License 1.1; IBM Corp., Reserved Font Name "Plex" |
| Tajawal (H251 Arabic) | [google/fonts/ofl/tajawal/OFL.txt](https://github.com/google/fonts/blob/main/ofl/tajawal/OFL.txt) | `76c01b50a1da8b3ae44a6f05c2e0d6ccb1cac5b9` | SIL Open Font License 1.1; Boutros International |
| Inter (H251 English) | [google/fonts/ofl/inter/OFL.txt](https://github.com/google/fonts/blob/main/ofl/inter/OFL.txt) | `21f6aff961064c2e429f570995e446bcdd555422` | SIL Open Font License 1.1; Inter Project Authors |

**Font Awesome Free 6.5.2** was inspected at the exact [upstream `6.5.2` `LICENSE.txt`](https://github.com/FortAwesome/Font-Awesome/blob/6.5.2/LICENSE.txt), Git blob `e69c5e39a30ce8d112e012cac36289e77691a9b7`. Its license explicitly distinguishes **SVG/JS icon assets: CC BY 4.0**, **web/desktop font files: SIL OFL 1.1**, and **non-font/non-icon code (including CSS): MIT**. Do not flatten these distinct categories into "all icons are MIT." Brand icon trademark and attribution concerns remain separate from the package's source-license text. H251 declares a CDN CSS URL; no live network request, fetched CSS digest, CSS `@font-face` dependency, icon glyph list, or screenshot was inspected in H314.

## Narrow findings, still-open rights gates

1. **Resolved only as upstream declaration evidence:** the five named font families have accessible upstream SIL OFL 1.1 license texts, and Font Awesome Free 6.5.2 has a verified version-specific mixed license. These texts are not proof of compliance in H250/H251 or evidence of permission from an app-specific asset author.
2. **Unresolved runtime provenance:** the actual browser-fetched font/icon bytes, fallback glyphs, blocked network behavior, and any visual changes from external resources are unverified. The original offline replays do not establish online visual fidelity.
3. **Unresolved local and derived assets:** H251's local logo/favicon and its other checked-in images still need individual chain-of-title, attribution and permitted-use review; H250's inline SVG and any externally derived icon still require review.
4. **Unresolved scientific custody:** H199 official `ADMITTED_TRAIN_MANIFEST_SHAS` is empty. H173 scored OOD is sealed, H212 barred, H250/H251 quarantined. A license statement is not a new blind OOD split or measured BASE-vs-ADAPTED Qwen improvement.

**Next required independent evidence:** source-owner/per-asset authorship and licensing attestation, version/digest-bound actual-rendered-resource manifest from an authorized session, historical source/npm/fork provenance, and independent TRAIN/DEV plus untouched blind OOD custody review. No self-certification, no model run, no GPU, no admission or release. H293/H294 safety blocks remain untouched.
