# H253 — Original-app visual media rights dependency audit

**Public source/docs/synthetic tests only. No original donor bytes, owner screenshots, gold, SHA receipts, private historical identities, or model artifacts.**

H250 and H251 used real intact original apps in offline Edge. A root MIT license does not independently establish the authorship or licensing of every local image, icon, external stylesheet, font, or embedded SVG. This tool reads only a clean, full-SHA-pinned original public Git checkout's HTML/CSS, its root LICENSE text, and Git metadata. It never opens image/font bytes or owner-private media.

It records local referenced visual assets, external runtime URLs (separate from preconnect hints), missing local assets, unsafe/embedded URLs and inline SVG elements. It always returns INCOMPLETE_NO_ADMISSION: static references are not proof that resources loaded in a browser, nor proof of third-party rights or historical source/pixel independence. JavaScript-generated/dynamic resources, runtime CSS, srcset edge cases and inherited CSS are not exhaustively covered.

## Source-only audit examples (owner Windows, original clones already present)

    python -m h253.original_app_asset_rights --repo D:\Hala-H250-Original-simple-to-do-list --sha 009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff --entry index.html

    python -m h253.original_app_asset_rights --repo D:\Hala-H251-Original-LifeOS --sha b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c --entry index.html --entry index-en.html

H251 original index.html and index-en.html reference assests/logo3.png as an image and assests/icon.ico as favicon. Both also reference Google Fonts and Font Awesome remote CSS. The original README acknowledges Font Awesome, but a generic project MIT LICENSE does not establish each media asset's chain of title. H250 includes external Google Fonts and inline SVG elements.

No TRAIN/DEV/OOD may be admitted based on this report. H199 remains empty; H173 scored OOD remains sealed; H212 procedural research data remains optimizer-blocked. Prior H15 engineering LoRA does not imply a trained native GUI model or accuracy gain.


## Actual pinned public original-source results (2026-10-31, Windows CPU)

- H250 full original app at 009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff: one HTML entry, 0 referenced local image/font files, 1 remote Google Fonts stylesheet, 2 preconnect hints (not loads), 2 inline SVG elements. This does not establish SVG authorship.
- H251 full original LifeOS app at b5fcb78c0606cfb610c0ae0d7a3b99a901bb682c: both original AR and EN HTML entries, 2 local visual assets (assests/icon.ico, assests/logo3.png), 3 remote stylesheet URLs (Font Awesome and two language-specific Google Fonts), 2 preconnect hints (not loads).
- Both original checkouts matched their exact Git SHA, were clean, and contained root MIT LICENSE text. Neither scan read image/font bytes or any owner-private media. Neither passes rights or dataset admission.
- In H251 the logo image is explicitly referenced by the original app HTML; do not assert its authorship/rights or actual browser load without further evidence. Offline network blocking means remote fonts/icons should not be assumed to have rendered.
