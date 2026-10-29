# H174 static original-entrypoint asset inventory (source-only)

This check is **not native GUI qualification**, a dataset, a model test, or a
permission to capture, train, select, or evaluate. It inspects only pinned
NEW candidate HTML and referenced script/stylesheet paths after the existing
Git/HTML/license provenance gate. It never reads prior benchmark images,
owner reference gold, GPU predictions, or consumed evaluation results.

Owner Windows Abdulaziz, 2026-10-29, pinned source-only checkouts:

| Candidate | Local HTML script/CSS references | Missing | Other constraints |
| --- | --- | --- | --- |
| Pickr | 2 present | 0 | 3 SCSS references require a build; external Google Fonts reference |
| interact.js Dropzones | 2 present | 0 | Module imports, browser behavior and real actions **not audited** |
| medium-zoom vanilla | 0 present | 3 | Sparse checkout lacks src/index.js, src/reset.css, src/style.css |

The inventory is deliberately limited to direct script[src] and
link[rel=stylesheet][href] references. It does not validate JS imports,
CSS @import, fetches, runtime bundles, native interactions, accessibility,
pixel replay, bilingual actions, or dataset admission. Even zero missing
references cannot qualify an original app. No files were added to or
removed from the pristine donor checkouts.

Next gates: reproduce complete immutable donor source/build dependencies
in a separate owner-only workspace, verify source-native behavior and
independent decoded RGBA replay for genuinely new TRAIN/DEV, and reserve
separate untouched OOD identities **before** training. H173 and every
historical DEV/OOD are permanently consumed.

## H174 HTML URL resolution and Git-backed asset evidence

A local file existing on disk is **not** evidence that it belongs to the
immutable original Git commit: clean Git status can omit ignored/generated
assets. `local_present` now requires `git ls-files --error-unmatch` to match
the exact original relative path. Existing but untracked/ignored references
are reported separately in `local_untracked`, never counted as pinned source.
HTML `<base href>` is rejected before reference inspection because browser
resolution may redirect an apparently local `script[src]` or stylesheet to
an external or different base URL. No assets are fetched or executed.

Three independent synthetic negative/contract tests reproduced both false
assumptions on the parent implementation before repair. Owner Windows H174
suite after repair: 75 passed; all three pristine source-only donor receipts
still have zero `local_untracked`. Pickr still requires SCSS build/external
font, and medium-zoom still lacks three sparse-checkout assets. No donor is
native-GUI qualified or assigned to TRAIN/DEV/OOD by this inventory.
