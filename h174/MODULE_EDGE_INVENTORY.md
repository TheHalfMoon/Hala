# H174: immutable donor JavaScript/CSS import-edge inventory

**Research-only source preflight, NOT native GUI qualification.** This stage
uses exact pinned original MIT Git donor checkouts, after the H174 Git/HTML/
license provenance gate. It reads tracked local browser assets, but does not
build, execute JavaScript, start browsers, capture TRAIN/DEV/holdout images,
open historical benchmark media, score old results, call GPUs, or train.

## Owner Windows, 2026-10-29 — original donor receipts

| Exact-pinned donor | Source-only dependency findings | Native QA |
| --- | --- | --- |
| Pickr (simonwep/pickr) | Original `www/index.js` imports `/src/js/pickr.js`; latter imports **nanopop** (bare package) and 6 extensionless relative paths not directly browser-resolvable as files. Entry HTML references 3 original SCSS stylesheets and external Google Fonts. | **BLOCKED**: original package resolution/build and offline CSS needed |
| interact.js Dropzones (taye/interact.js) | Original `examples/dropzones/index.js` imports **@interactjs/interactjs** bare package; two entry HTML assets tracked. | **BLOCKED**: original dependency resolution/bundle needed |
| medium-zoom vanilla (francoischalifour/medium-zoom) | Sparse original checkout still missing 3 directly referenced files: `src/index.js`, `src/reset.css`, `src/style.css`. No tracked browser module available to descend into. | **BLOCKED**: restore source-only checkout from exact pinned upstream, not generate donor files |

The scanner intentionally reports package specifiers, missing/untracked local
imports, external imports and unsupported/dynamic forms. It follows a bounded
graph of tracked original JS/CSS references, checks lexical traversal and
reparse-point safety, and reports direct HTML missing assets separately.
It is a **conservative textual inventory, not a full ECMAScript/CSS parser**
and cannot prove all runtime imports, build transitive dependencies, module
resolution, native browser state or complete history-registry exclusion.

A bundler or compile output is a derived artifact and must be built in a
**separate local workspace**, never written to the pristine original Git
checkouts or misidentified as original Git-tracked media. Exact build sources
and source SHA must be retained in an auditable owner-only receipt.
Before model research, pick genuinely new TRAIN/DEV apps, reserve an independent
future OOD source family (no images), establish the full owner SHA-only
historical source registry, and pass genuine native-action + independent pixel
replay on fresh TRAIN/DEV. All H173 v3 and earlier consumed OOD/DEV remain
permanently forbidden. No model success claim or release.
