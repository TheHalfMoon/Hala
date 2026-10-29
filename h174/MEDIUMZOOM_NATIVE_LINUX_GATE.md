# H174 new original donor: medium-zoom vanilla GUI smoke

**Status: source-only Linux research candidate, NOT TRAIN/DEV/OOD admission.**

Different original native interaction family from the previous Pickr swatch
(click+color change): zooming a real original photograph in an original
Vite-built vanilla application and requiring a visible zoomed photo.

Immutable upstream: `francoischalifour/medium-zoom` at
`dccda8ab5a1f1677cf1f141d1ead4f14780f0ca4` (MIT). Original
`examples/vanilla/index.html` SHA-256
`0e85b970cc9a4f489945ff4f325c7d3380855aa7433223d24f2bd3336ab1a5ed`,
original LICENSE SHA-256
`166ba718b70c6278cce9c9758396e25e35c7f1a940d39b2c8eeaceed18cc03fb`.
These are cross-checked against the existing H174 original-source catalog,
never sourced from new guesses. Full Git archive is exported into an
isolated ephemeral scratch workspace separate from a clean original checkout
to restore missing files from the sparse owner copy without generating fake
donor assets.

The original root `bun.lock`, original Bun 1.4.4, and original vanilla
package `examples/vanilla/package.json` declare `medium-zoom@1.1.1` as
a locked dependency. The test uses `bun install --frozen-lockfile
--ignore-scripts` and the original vanilla `bun run build`; therefore
its source identity is the pinned demo Git revision with the exact original
lockfile dependency, *not* the root unreleased library 1.1.2.
No dependency or output is written to the original donor checkout.

New hosted source-only gate:
`.github/workflows/h174-mediumzoom-native-linux.yml`.
It starts original app at 127.0.0.1, blocks Service Workers, downloads,
WebSockets and off-loopback HTTP requests at the browser-context level, and
uses a real Chromium click on the original first photo. It demands a visible
`.medium-zoom-overlay`, one `.medium-zoom-image--opened`, and an exact
decoded 960×720 RGBA pixel replay across **two independent contexts**.
Frames are held only in memory. Native CI failures are real evidence and
must be fixed rather than masked.

Until the dedicated workflow runs to **success on the exact head**, there is
no medium-zoom native success claim. Even after Linux success this remains
one narrow original-app smoke; original Windows Abdulaziz Edge replay,
complete historical source SHA and donor-family audit, bilingual new TRAIN
and DEV independent references, and a separate never-seen OOD family remain
required *before training*. No old H173 media/predictions/gold accessed, no
model/GPU/optimizer, no publication/merges or paid compute.
