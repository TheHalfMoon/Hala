# H174 Pickr immutable build and original GUI smoke receipt

**Owner-only Windows Abdulaziz, 2026-10-29. Narrow native smoke only; NOT TRAIN/DEV/OOD admission.**

## Immutable source and build

- MIT donor: https://github.com/simonwep/pickr, pinned Git commit e8ff88d9161ffc6e3c46a4430be8c328c42d6415. Original tracked LICENSE and pnpm-lock.yaml retained; package @simonwep/pickr v1.10.2, packageManager pnpm v10.27.0.
- Original donor checkout clean. Exact commit exported with git archive --format=zip to a NEW separate owner-only D: scratch tree, never into the donor checkout. Source archive SHA-256: 060ec28b66e7c30b0298463ddeac5b4405079780b3a6dd0eed7f1de45a975bc6.
- Owner Node v24.19.0; Corepack pnpm v10.27.0. The initial offline frozen install with disabled lifecycle scripts FAILED because @eslint-community/regexpp v4.12.2 was missing from the store. This failure is retained as evidence, not hidden.
- Original pnpm fetch --frozen-lockfile then downloaded 116 public-registry packages to the cache, at zero monetary cost. Subsequent pnpm install --offline --frozen-lockfile --ignore-scripts SUCCEEDED. Dependency lifecycle scripts were not enabled. Network was used only to populate the dependency cache, NOT for the subsequent install or build.
- Original pnpm run build:lib PASSED (Vite v8.1.5, 9 modules, generated JS and MJS). Original pnpm run build:demo PASSED (16 modules, derived dist/index.html, hashed JS/CSS and favicon).
- All dependencies and compiled assets remain in the owner scratch tree, never in the pristine pinned donor root. Compiled assets are NOT original Git-tracked source. No build artifacts, images, gold or weights committed.

## Native GUI smoke and independent replay

- Owner Playwright Python v1.64.1 used installed Microsoft Edge headless because the Playwright bundled Chromium headless shell was absent. Original compiled Pickr demo served via an ephemeral 127.0.0.1 port, viewport 960x720, scale 1.
- Every browser request outside the exact loopback port was aborted, including the donor's external Google Fonts request.
- TWO fresh independent browser contexts each loaded the demo, found the original classic control and original picker, clicked the picker, observed the visible popup, clicked one original swatch and ASSERTED that the original color-result input changed.
- After neutralizing pointer to (0,0) and stabilizing, each context produced an in-memory decoded 960x720 RGBA frame. Independent RGBA hashes matched exactly. Both had 14 swatches and zero JavaScript page errors. No screenshots were saved or published.
- The new h174.pickr_native_smoke module reproduces this narrow owner-only smoke on explicit scratch dist paths. Synthetic CI tests check fail-closed derived assets, exact loopback URL restriction, external scripts, traversal, base URLs, missing and empty generated assets. CI never starts browsers or accesses donor/private images.

Owner-only command example: python -m h174.pickr_native_smoke --dist-dir D:\<new-owner-scratch>\source\dist --browser-channel msedge

## Non-admission and research firewall

This is ONLY one donor's build and limited native GUI smoke. It does NOT qualify full bilingual original reference actions, TRAIN/DEV, reserved unseen OOD, full historical SHA-only registry, source-family separation, or model performance. interact.js still requires pinned Yarn/Rollup workspace build; medium-zoom vanilla still requires complete pinned source export and original build. A new unrelated unseen OOD family must be reserved before training.

H173 v1/v2 remain quarantined; H173 v3 and all previous DEV/OOD are permanently consumed, never reopened or reused. H164 language and H171 rank16 vision checkpoints remain frozen. The one-shot H173 result 2/32 versus 3/32 (McNemar p=1.0) is not significant. No GPU, paid models, main merge, release, publication, human review or access to consumed images/gold/weights occurred.
